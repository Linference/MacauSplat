import { createHash } from "node:crypto";
import { createReadStream } from "node:fs";
import { copyFile, mkdir, readdir, readFile, stat, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const workspace = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const packageMetadata = JSON.parse(await readFile(path.join(workspace, "package.json"), "utf8"));
const version = packageMetadata.version;

if (process.platform !== "win32") {
  throw new Error(`Windows installer collection must run on Windows, not ${process.platform}.`);
}

if (!version || typeof version !== "string") {
  throw new Error("package.json does not contain a valid version.");
}

// The shipped artifact name must use this project's name. Apache-2.0 does not grant
// the upstream marks, and NOTICE/docs/COMPLIANCE.md state that this product does not
// use the upstream name, so a released file called OOOSplat-... contradicts the
// distribution's own compliance statement.
const PRODUCT_NAME = "MacauSplat";

// Tauri's NSIS bundler names the file "<productName>_<version>_<arch>-setup.exe"
// using the productName and version from the resolved tauri configuration. The
// architecture comes from the installer that was actually produced, NOT from
// process.arch: this script can run under an x64 Node.js interpreter that packaged
// an arm64 target (or the reverse), so process.arch is not authoritative.
const bundleDirectory = path.join(workspace, "src-tauri", "target", "release", "bundle", "nsis");

let entries;
try {
  entries = await readdir(bundleDirectory, { withFileTypes: true });
} catch (error) {
  throw new Error(`Cannot read the NSIS bundle directory ${bundleDirectory}: ${error.message}`);
}

// Match only the exact product and version. A loose `name.includes(version)` test
// would also accept MacauSplat_0.4.10_x64-setup.exe when package.json says 0.4.1,
// silently publishing the wrong installer.
const installerPattern = new RegExp(
  `^${PRODUCT_NAME}[-_]${version.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}[-_](?<arch>x64|arm64|aarch64|i686)-setup\\.exe$`,
);

const matches = [];
for (const entry of entries) {
  if (!entry.isFile()) continue;
  const match = installerPattern.exec(entry.name);
  if (match) {
    matches.push({ name: entry.name, arch: match.groups.arch });
  }
}

if (matches.length === 0) {
  const found = entries.filter((entry) => entry.isFile() && entry.name.endsWith(".exe")).map((entry) => entry.name);
  throw new Error(
    `No ${PRODUCT_NAME} ${version} NSIS installer was found in ${bundleDirectory}. ` +
      `Expected a file matching <product>_<version>_<arch>-setup.exe. ` +
      `Present .exe files: ${found.length > 0 ? found.join(", ") : "(none)"}. ` +
      `Run "npm run package:windows", which stages the CLI and builds the NSIS bundle.`,
  );
}

const installers = await Promise.all(
  matches.map(async (candidate) => ({
    ...candidate,
    path: path.join(bundleDirectory, candidate.name),
    modifiedAt: (await stat(path.join(bundleDirectory, candidate.name))).mtimeMs,
  })),
);
installers.sort((left, right) => right.modifiedAt - left.modifiedAt);

const selected = installers[0];
if (installers.length > 1) {
  console.warn(
    `[collect-windows-installer] ${installers.length} matching installers found; using the most recent: ${selected.name}`,
  );
}

// The bundled engines are ~505 MB. An installer far below that means the engine
// resources were not packaged, which is exactly the failure this script should
// surface rather than silently publish.
const minimumBytes = 400 * 1024 * 1024;
const selectedSize = (await stat(selected.path)).size;
if (selectedSize < minimumBytes) {
  throw new Error(
    `${selected.name} is only ${(selectedSize / 1048576).toFixed(1)} MB. ` +
      `The bundled FFmpeg/COLMAP/Brush engines alone are roughly 505 MB, so this installer is almost ` +
      `certainly missing engines/. Run "npm run setup:engines:windows" and "npm run verify:engines:windows" first.`,
  );
}

const architecture = { aarch64: "arm64", i686: "x86" }[selected.arch] ?? selected.arch;
const artifactDirectory = path.join(workspace, "dist-artifacts");
const artifactName = `${PRODUCT_NAME}-${version}-${architecture}-windows.exe`;
const artifactPath = path.join(artifactDirectory, artifactName);

await mkdir(artifactDirectory, { recursive: true });
await copyFile(selected.path, artifactPath);

const hash = createHash("sha256");
for await (const chunk of createReadStream(artifactPath)) {
  hash.update(chunk);
}
const digest = hash.digest("hex");
await writeFile(path.join(artifactDirectory, "SHA256SUMS.txt"), `${digest}  ${artifactName}\n`, "utf8");

console.log(`Windows installer: ${artifactPath}`);
console.log(`  source:   ${selected.name} (${(selectedSize / 1048576).toFixed(1)} MB, arch ${selected.arch})`);
console.log(`  sha256:   ${digest}`);
