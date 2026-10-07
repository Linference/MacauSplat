import { createHash } from "node:crypto";
import { existsSync, readFileSync, statSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { brushLock } from "./brush-runtime.mjs";

const workspace = join(dirname(fileURLToPath(import.meta.url)), "..");

function workspacePath(relativePath) {
  return join(workspace, ...relativePath.split("/"));
}

function readText(relativePath) {
  return readFileSync(workspacePath(relativePath), "utf8");
}

function assert(condition, message) {
  if (!condition) {
    throw new Error(message);
  }
}

function assertContains(text, expected, context) {
  assert(text.includes(expected), `${context} is missing '${expected}'.`);
}

function normalizedSha256(text) {
  const normalized = text.replace(/\r\n?/g, "\n").replace(/\n+$/g, "");
  return createHash("sha256").update(normalized, "utf8").digest("hex").toUpperCase();
}

const requiredFiles = [
  "LICENSE",
  "NOTICE",
  "TRADEMARK_POLICY.md",
  "GENERATED_OUTPUTS.md",
  "licenses/THIRD_PARTY_NOTICES.txt",
  "licenses/FFmpeg-LGPL-2.1.txt",
  "licenses/FFmpeg-CORRESPONDING-SOURCE.txt",
  "licenses/COLMAP-LICENSE.txt",
  "licenses/NVIDIA-CUDA-Runtime.txt",
  "licenses/Brush-LICENSE.txt",
  "licenses/PlayCanvas-MIT.txt",
  "licenses/Mediabunny-MPL-2.0.txt",
  "engines/manifest.json",
  "engines/manifest.linux.json",
  "engines/manifest.macos.json",
];

for (const relativePath of requiredFiles) {
  const path = workspacePath(relativePath);
  assert(existsSync(path) && statSync(path).isFile(), `Required license file is missing: ${relativePath}`);
}

const expectedApacheSha256 = "58D1E17FFE5109A7AE296CAAFCADFDBE6A7D176F0BC4AB01E12A689B0499D8BD";
assert(
  normalizedSha256(readText("LICENSE")) === expectedApacheSha256,
  "LICENSE is not the unmodified Apache License 2.0 text.",
);

const tauri = JSON.parse(readText("src-tauri/tauri.conf.json"));
const tauriIdentity = { productName: tauri.productName, version: tauri.version };
const packageJson = JSON.parse(readText("package.json"));
assert(packageJson.license === "Apache-2.0", "package.json must declare Apache-2.0.");
// The distributable is MacauSplat, so package.json must identify this project and
// its own repository. Reusing the upstream package identity in package.json would
// make the shipped metadata claim to be OOOSplat, which contradicts NOTICE and
// docs/COMPLIANCE.md (and Apache-2.0 Section 6 does not grant the upstream marks).
assert(packageJson.author === "Linference", "package.json must identify Linference as author.");
assert(
  packageJson.repository?.url === "git+https://github.com/Linference/MacauSplat.git",
  "package.json repository URL must identify this fork, not upstream OOOSplat.",
);
assert(
  packageJson.upstream?.url === "https://github.com/ooolabdev/ooosplat",
  "package.json must record the upstream OOOSplat project under an explicit `upstream` field.",
);
assert(
  !JSON.stringify(packageJson).includes("ooolabdev/ooosplat.git"),
  "package.json still points at the upstream OOOSplat repository as its own repository.",
);
assert(
  tauriIdentity.productName === "MacauSplat" && tauriIdentity.version,
  "Tauri product identity must be MacauSplat.",
);

const cargo = readText("src-tauri/Cargo.toml");
assert(/^license\s*=\s*"Apache-2\.0"\s*$/m.test(cargo), "Cargo.toml must declare Apache-2.0.");
const cargoAuthors = cargo.match(/^authors\s*=\s*\[([^\]]+)\]\s*$/m)?.[1] ?? "";
assert(cargoAuthors.includes('"Linference"') && cargoAuthors.includes('"ooolabdev"'), "Cargo.toml must credit the fork maintainer and upstream author.");
assert(
  /^repository\s*=\s*"https:\/\/github\.com\/Linference\/MacauSplat"\s*$/m.test(cargo),
  "Cargo.toml must identify the fork as its own repository.",
);
assert(cargo.includes('[package.metadata.upstream]') && cargo.includes('repository = "https://github.com/ooolabdev/ooosplat"'), "Cargo.toml must preserve explicit upstream provenance.");

for (const resource of [
  "../LICENSE",
  "../NOTICE",
  "../TRADEMARK_POLICY.md",
  "../GENERATED_OUTPUTS.md",
  "../licenses/THIRD_PARTY_NOTICES.txt",
  "../licenses/FFmpeg-LGPL-2.1.txt",
  "../licenses/FFmpeg-CORRESPONDING-SOURCE.txt",
  "../licenses/COLMAP-LICENSE.txt",
  "../licenses/NVIDIA-CUDA-Runtime.txt",
  "../licenses/Brush-LICENSE.txt",
  "../licenses/PlayCanvas-MIT.txt",
  "../licenses/Mediabunny-MPL-2.0.txt",
]) {
  assert(Object.hasOwn(tauri.bundle?.resources ?? {}, resource), `Tauri resources are missing ${resource}.`);
}

// LGPL-2.1 Section 6 makes the corresponding-source offer a condition of
// distributing the LGPL FFmpeg object code. The offer must ship inside every
// platform bundle, so each platform override must also map the file: a platform
// config that replaces `bundle.resources` instead of merging it would silently
// drop the offer from that installer.
for (const platformConfig of ["tauri.windows.conf.json", "tauri.linux.conf.json", "tauri.macos.conf.json"]) {
  const platform = JSON.parse(readText(`src-tauri/${platformConfig}`));
  assert(
    Object.hasOwn(platform.bundle?.resources ?? {}, "../licenses/FFmpeg-CORRESPONDING-SOURCE.txt"),
    `${platformConfig} resources are missing the LGPL corresponding-source offer.`,
  );
}

const tauriWindows = JSON.parse(readText("src-tauri/tauri.windows.conf.json"));
for (const resource of [
  "../engines/manifest.json",
  "../engines/ffmpeg/",
  "../engines/colmap/",
  "../engines/brush/",
]) {
  assert(Object.hasOwn(tauriWindows.bundle?.resources ?? {}, resource), `Windows Tauri resources are missing ${resource}.`);
}
const tauriLinux = JSON.parse(readText("src-tauri/tauri.linux.conf.json"));
assert(tauriLinux.bundle?.active === true, "Linux Tauri bundling must be enabled.");
assert(tauriLinux.bundle?.targets?.includes("deb"), "Linux Tauri targets must include deb.");
for (const resource of ["../engines/manifest.linux.json", "../engines/linux/brush/brush_app"]) {
  assert(Object.hasOwn(tauriLinux.bundle?.resources ?? {}, resource), `Linux Tauri resources are missing ${resource}.`);
}
for (const dependency of ["ffmpeg"]) {
  assert(tauriLinux.bundle?.linux?.deb?.depends?.includes(dependency), `Linux DEB dependencies are missing ${dependency}.`);
}
assert(!tauriLinux.bundle?.linux?.deb?.depends?.includes("colmap"), "Linux must not depend on unpinned system COLMAP.");
for (const resource of ["../engines/linux/colmap/"]) {
  assert(Object.hasOwn(tauriLinux.bundle?.resources ?? {}, resource), `Missing Linux COLMAP resource ${resource}`);
}
const tauriMacos = JSON.parse(readText("src-tauri/tauri.macos.conf.json"));
assert(tauriMacos.bundle?.macOS?.minimumSystemVersion === "15.0", "macOS bundle must target macOS 15.0.");
for (const resource of [
  "../engines/manifest.macos.json",
  "../engines/macos/arm64/bin/",
  "../engines/macos/arm64/lib/",
  "../engines/macos/arm64/licenses/",
  "../engines/macos/arm64/SHA256SUMS",
  "../engines/macos/arm64/BUILD-INFO.json",
  "../engines/macos/arm64/BUNDLED-COMPONENTS.json",
  "../engines/macos/arm64/colmap/",
]) {
  assert(Object.hasOwn(tauriMacos.bundle?.resources ?? {}, resource), `macOS Tauri resources are missing ${resource}.`);
}

const expectedEngines = new Map([
  ["FFmpeg / FFprobe", { license: "LGPL-2.1-or-later", file: "licenses/FFmpeg-LGPL-2.1.txt" }],
  ["COLMAP", { license: "BSD-3-Clause", file: "licenses/COLMAP-LICENSE.txt" }],
  ["Brush", { license: "Apache-2.0", file: "licenses/Brush-LICENSE.txt" }],
]);
const manifest = JSON.parse(readText("engines/manifest.json"));
const colmapRuntime = JSON.parse(readText("engines/colmap-runtime.json"));
assert(colmapRuntime.repository === "ooolabdev/ooosplat-colmap", "COLMAP runtime repository is incorrect.");
assert(colmapRuntime.releaseTag === "colmap-4.2.1-runtime.1", "COLMAP runtime release tag is incorrect.");
assert(colmapRuntime.sourceCommit === "bd1fcf654d2dd8fefa1466999c190a246f83f4b9", "COLMAP source commit is incorrect.");
for (const [platform, runtime] of Object.entries(colmapRuntime.platforms ?? {})) {
  for (const [name, value] of Object.entries({ archive: runtime.archiveSha256, inventory: runtime.integritySha256, buildInfo: runtime.buildInfoSha256 })) {
    assert(/^[a-f0-9]{64}$/i.test(value ?? ""), `${platform} COLMAP ${name} hash is not pinned.`);
  }
  assert(runtime.sourceUrl?.startsWith(`https://github.com/ooolabdev/ooosplat-colmap/releases/download/${colmapRuntime.releaseTag}/`), `${platform} COLMAP URL is not pinned to the fork Release.`);
}
assert(manifest.schemaVersion >= 2, "Engine manifest schemaVersion must include license mappings.");
assert(manifest.engines?.length === 3, "License verification expects exactly the three direct native engines.");

const thirdParty = readText("licenses/THIRD_PARTY_NOTICES.txt");
assert(!thirdParty.includes("OOOSplat 0.2.0"), "Third-party notices still contain the obsolete 0.2.0 heading.");
assert(!existsSync(workspacePath("licenses/FFmpeg-LICENSE.txt")), "The obsolete LGPLv3 FFmpeg-LICENSE.txt file must not exist.");

for (const engine of manifest.engines) {
  const expected = expectedEngines.get(engine.name);
  assert(expected, `Unexpected direct engine in manifest: ${engine.name}`);
  assert(engine.license?.startsWith(expected.license), `${engine.name} license identifier does not match ${expected.license}.`);
  assert(engine.licenseFiles?.length === 1, `${engine.name} must map to one direct license file.`);
  assert(engine.licenseFiles[0] === expected.file, `${engine.name} license file mapping is incorrect.`);
  assert(existsSync(workspacePath(expected.file)), `${engine.name} mapped license file is missing.`);
  assertContains(thirdParty, engine.name, "THIRD_PARTY_NOTICES.txt");
  assertContains(thirdParty, expected.license, "THIRD_PARTY_NOTICES.txt");
  assertContains(thirdParty, expected.file, "THIRD_PARTY_NOTICES.txt");
}

const windowsFfmpeg = manifest.engines.find((engine) => engine.name === "FFmpeg / FFprobe");
const windowsBrush = brushLock("windows");
assert(manifest.requiredFiles.find(file => file.path === "engines/brush/brush_app.exe")?.sha256 === windowsBrush.binarySha256,
  "Windows Brush required binary hash must match the pinned OOOBrush CLI.");
for (const platform of ["windows", "linux", "macos"]) {
  const brush = brushLock(platform);
  assertContains(thirdParty, brush.sourceUrl, "THIRD_PARTY_NOTICES.txt");
  assertContains(thirdParty, brush.commit, "THIRD_PARTY_NOTICES.txt");
}
assert(windowsFfmpeg, "Windows FFmpeg manifest entry is missing.");
assert(
  !windowsFfmpeg.sourceUrl?.includes("/latest/"),
  "Windows FFmpeg source URL must reference an immutable release asset, not /latest/.",
);
assertContains(thirdParty, windowsFfmpeg.actualVersion, "THIRD_PARTY_NOTICES.txt");
assertContains(thirdParty, windowsFfmpeg.sourceUrl, "THIRD_PARTY_NOTICES.txt");

const linuxManifest = JSON.parse(readText("engines/manifest.linux.json"));
assert(linuxManifest.schemaVersion >= 2, "Linux engine manifest schemaVersion must include license mappings.");
assert(linuxManifest.brush?.version === "1.0.0", "Linux Brush version is incorrect.");
assert(
  linuxManifest.brush?.sourceUrl ===
    "https://github.com/ooolabdev/OOOBrush/releases/download/ooo-v1.0.0/OOOBrush-x86_64-unknown-linux-gnu.tar.gz",
  "Linux Brush release archive is incorrect.",
);
assert(linuxManifest.brush?.license === "Apache-2.0", "Linux Brush license identifier is incorrect.");
assert(
  linuxManifest.brush?.licenseFiles?.length === 1 && linuxManifest.brush.licenseFiles[0] === "licenses/Brush-LICENSE.txt",
  "Linux Brush license file mapping is incorrect.",
);
for (const marker of [
  "Ubuntu 24.04 Alpha, Linux x86_64 release archive",
  "OOOBrush-x86_64-unknown-linux-gnu.tar.gz",
  "engines/manifest.linux.json",
]) {
  assertContains(thirdParty, marker, "THIRD_PARTY_NOTICES.txt");
}

const macosManifest = JSON.parse(readText("engines/manifest.macos.json"));
assert(macosManifest.schemaVersion >= 1, "macOS engine manifest schemaVersion is missing.");
assert(macosManifest.platform === "macos", "macOS engine manifest platform is incorrect.");
assert(macosManifest.architecture === "arm64", "macOS engine manifest must be Apple arm64 only.");
assert(macosManifest.minimumSystemVersion === "15.0", "macOS engine manifest must target macOS 15.0.");
assert(/^[A-F0-9]{40}$/.test(macosManifest.buildEnvironment?.homebrewCoreCommit), "macOS Homebrew/core build commit is not pinned.");
assert(macosManifest.buildEnvironment?.runner === "macos-15", "macOS engine runner must be pinned to macos-15.");
assert(macosManifest.buildEnvironment?.usePinnedHomebrewBottles === true, "macOS build dependencies must use pinned Homebrew bottles.");
assert(macosManifest.engines?.length === 3, "macOS manifest must contain the three direct engines.");

// The macOS engine runtime is fetched by scripts/setup-engines-macos.sh from the
// `distribution` block. Nothing used to verify this block, so a release URL that
// pointed at a tag which does not exist passed every check and made the macOS CI
// job fail at download time instead. Verify the block's shape and internal
// consistency here.
const macosDistribution = macosManifest.distribution;
assert(macosDistribution && typeof macosDistribution === "object", "macOS manifest is missing the distribution block.");
for (const key of ["tag", "archiveName", "sourceUrl", "archiveSha256File", "destination", "integrityFile"]) {
  assert(
    typeof macosDistribution[key] === "string" && macosDistribution[key].length > 0,
    `macOS distribution.${key} must be a non-empty string.`,
  );
}
for (const key of ["tag", "archiveName", "sourceUrl", "archiveSha256File", "destination", "integrityFile"]) {
  assert(macosDistribution[key] === macosDistribution[key].trim(), `macOS distribution.${key} must not be padded with whitespace.`);
}
assert(
  /^[A-Za-z0-9][A-Za-z0-9._-]*$/.test(macosDistribution.tag),
  `macOS distribution.tag is not a valid git tag: ${macosDistribution.tag}`,
);
assert(
  /^[A-Za-z0-9][A-Za-z0-9._-]*\.tar\.xz$/.test(macosDistribution.archiveName),
  `macOS distribution.archiveName must be a .tar.xz archive name: ${macosDistribution.archiveName}`,
);
const macosTagVersion = macosDistribution.tag.slice("engines-macos-arm64-".length);
assert(
  macosDistribution.archiveName.includes(macosTagVersion),
  `macOS distribution.archiveName must contain the version '${macosTagVersion}' from distribution.tag.`,
);
assert(
  macosDistribution.destination === "engines/macos/arm64",
  "macOS distribution.destination must be engines/macos/arm64 to match the Tauri resource mapping.",
);
assert(
  macosDistribution.archiveName.startsWith("macausplat-"),
  "macOS distribution.archiveName must use this project's name; released artifact names are brand elements.",
);
assert(
  macosDistribution.sourceUrl ===
    `https://github.com/Linference/MacauSplat/releases/download/${macosDistribution.tag}/${macosDistribution.archiveName}`,
  "macOS distribution.sourceUrl must resolve to this project's own release asset, built from the tag it names.",
);
assert(
  macosDistribution.archiveSha256File === `${macosDistribution.sourceUrl}.sha256`,
  "macOS distribution.archiveSha256File must be the .sha256 sibling of distribution.sourceUrl.",
);
assert(
  !macosDistribution.sourceUrl.includes("/latest/"),
  "macOS distribution.sourceUrl must reference an immutable release asset, not /latest/.",
);
assert(
  typeof macosManifest.distribution.publishedBy === "string" && macosManifest.distribution.publishedBy.length > 0,
  "macOS distribution.publishedBy must record who publishes the engine archive.",
);
assert(
  macosManifest.distribution.buildFallback === "scripts/build-engines-macos.sh",
  "macOS distribution.buildFallback must name the reproducible local build script.",
);
assert(
  macosManifest.distribution.archiveLayoutRoot === "macausplat-engines-macos-arm64",
  "macOS distribution.archiveLayoutRoot must match the directory the engine archive extracts to.",
);
// Both scripts must derive the extracted layout from this manifest field rather
// than hard-coding it, so the manifest stays the single source of truth.
for (const script of ["scripts/setup-engines-macos.sh", "scripts/build-engines-macos.sh"]) {
  assertContains(
    readText(script),
    "distribution.archiveLayoutRoot",
    `${script} (must read distribution.archiveLayoutRoot from the manifest)`,
  );
}
// scripts/setup-engines-macos.sh releases into and scripts/build-engines-macos.sh
// archives out of a directory named after distribution.archiveName's stem, and
// .github/workflows/macos-engines.yml must guard its tag against distribution.tag.
assert(
  existsSync(workspacePath("scripts/build-engines-macos.sh")),
  "macOS distribution.buildFallback script is missing.",
);
const macosEnginesWorkflow = readText(".github/workflows/macos-engines.yml");
assertContains(macosEnginesWorkflow, "distribution.tag", "macos-engines.yml");
assertContains(
  macosEnginesWorkflow,
  "engines-macos-arm64-v",
  "macos-engines.yml (the tag prefix the workflow publishes)",
);
assert(
  macosDistribution.tag.startsWith("engines-macos-arm64-v"),
  "macOS distribution.tag must use the engines-macos-arm64-v prefix that macos-engines.yml publishes.",
);

for (const engine of macosManifest.engines) {
  const expected = expectedEngines.get(engine.name);
  assert(expected, `Unexpected macOS engine: ${engine.name}`);
  assert(engine.license === expected.license, `${engine.name} macOS license identifier is incorrect.`);
  assert(engine.licenseFiles?.length === 1 && engine.licenseFiles[0] === expected.file, `${engine.name} macOS license mapping is incorrect.`);
  assert(/^[A-F0-9]{64}$/.test(engine.sourceSha256), `${engine.name} macOS source SHA-256 is invalid.`);
  assert(
    /^https:\/\//.test(engine.sourceUrl ?? ""),
    `${engine.name} macOS source URL must be an absolute https URL.`,
  );
  assert(
    engine.buildPolicy?.length > 0,
    `${engine.name} macOS buildPolicy must describe how the runtime was produced so the notices can be checked against it.`,
  );
}
for (const marker of [
  "macOS 15+ Apple Silicon arm64",
  "ffmpeg-8.1.2.tar.xz",
  "COLMAP 4.2.1 macOS arm64 CPU CLI-only",
  "OOOBrush-aarch64-apple-darwin.tar.gz",
  "engines/manifest.macos.json",
]) {
  assertContains(thirdParty, marker, "THIRD_PARTY_NOTICES.txt");
}

for (const component of ["PlayCanvas Engine", "PlayCanvas React", "PlayCanvas splat-transform (adapted algorithm)", "SuperSplat selection workflow (adapted algorithm)"]) {
  assertContains(thirdParty, component, "THIRD_PARTY_NOTICES.txt");
}
assertContains(thirdParty, "licenses/PlayCanvas-MIT.txt", "THIRD_PARTY_NOTICES.txt");
const playCanvasLicense = readText("licenses/PlayCanvas-MIT.txt");
assertContains(playCanvasLicense, "MIT License", "PlayCanvas license");
assertContains(playCanvasLicense, "Copyright (c) 2011-2026 PlayCanvas Ltd.", "PlayCanvas license");

for (const marker of [
  "Mediabunny",
  "1.52.3",
  "MPL-2.0",
  "licenses/Mediabunny-MPL-2.0.txt",
  "https://github.com/Vanilagy/mediabunny/tree/v1.52.3",
]) {
  assertContains(thirdParty, marker, "THIRD_PARTY_NOTICES.txt");
}
const mediabunnyLicense = readText("licenses/Mediabunny-MPL-2.0.txt");
assertContains(mediabunnyLicense, "Mozilla Public License Version 2.0", "Mediabunny license");
assertContains(mediabunnyLicense, "2. License Grants and Conditions", "Mediabunny license");

for (const marker of [
  "image crate",
  "0.25.10",
  "MIT OR Apache-2.0",
  "https://github.com/image-rs/image/tree/v0.25.10",
]) {
  assertContains(thirdParty, marker, "THIRD_PARTY_NOTICES.txt");
}
assertContains(readText("src-tauri/Cargo.toml"), 'image = { version = "0.25"', "Cargo.toml");

for (const marker of [
  "Tauri single-instance plugin",
  "2.4.5",
  "Apache License 2.0 OR MIT License",
  "https://crates.io/crates/tauri-plugin-single-instance/2.4.5",
]) {
  assertContains(thirdParty, marker, "THIRD_PARTY_NOTICES.txt");
}
assertContains(readText("src-tauri/Cargo.toml"), 'tauri-plugin-single-instance = "2"', "Cargo.toml");

const ffmpegLicense = readText("licenses/FFmpeg-LGPL-2.1.txt");
assertContains(ffmpegLicense, "GNU LESSER GENERAL PUBLIC LICENSE", "FFmpeg license");
assertContains(ffmpegLicense, "Version 2.1, February 1999", "FFmpeg license");
assert(!ffmpegLicense.includes("Version 3, 29 June 2007"), "FFmpeg license still contains the LGPLv3 text.");

// LGPL-2.1 Section 6 requires a written offer of the corresponding source when the
// LGPL object code is distributed without it. Shipping only the license text and a
// link to ffmpeg.org does not discharge that obligation, so the offer document and
// its packaging are enforced here.
const correspondingSource = readText("licenses/FFmpeg-CORRESPONDING-SOURCE.txt");
for (const marker of [
  "Written offer under GNU Lesser General Public License v2.1, Section 6",
  "How to obtain the corresponding source",
  "How this Object Code was built",
  "Relinking (LGPL v2.1 Section 6(b))",
  "https://github.com/Linference/MacauSplat/issues",
]) {
  assertContains(correspondingSource, marker, "FFmpeg corresponding-source offer");
}
assertContains(correspondingSource, windowsFfmpeg.sourceUrl, "FFmpeg corresponding-source offer");
assertContains(correspondingSource, windowsFfmpeg.archiveSha256, "FFmpeg corresponding-source offer");
assertContains(correspondingSource, "win64-lgpl-shared", "FFmpeg corresponding-source offer");
assertContains(correspondingSource, "variants/win64-lgpl-shared.sh", "FFmpeg corresponding-source offer");
assertContains(correspondingSource, "variants/defaults-lgpl.sh", "FFmpeg corresponding-source offer");
assertContains(correspondingSource, "LICENSE_FILE=COPYING.LGPLv3", "FFmpeg corresponding-source offer");
assertContains(correspondingSource, "build.sh", "FFmpeg corresponding-source offer");
assertContains(correspondingSource, "ffmpeg.exe -buildconf", "FFmpeg corresponding-source offer");
assertContains(correspondingSource, windowsFfmpeg.actualVersion, "FFmpeg corresponding-source offer");
// The offer must be reachable from the installer, referenced by the notices, and
// listed as a packaged resource; otherwise it exists only in the repository.
assertContains(thirdParty, "licenses/FFmpeg-CORRESPONDING-SOURCE.txt", "THIRD_PARTY_NOTICES.txt");
assert(
  Object.hasOwn(tauri.bundle?.resources ?? {}, "../licenses/FFmpeg-CORRESPONDING-SOURCE.txt"),
  "Tauri resources are missing the LGPL corresponding-source offer.",
);
// The release source archive the offer points at is the same archive the macOS
// engine manifest pins, so the two records must not drift apart.
const macosFfmpeg = macosManifest.engines.find((engine) => engine.name === "FFmpeg / FFprobe");
assert(macosFfmpeg, "macOS FFmpeg engine entry is missing.");
assertContains(
  correspondingSource,
  macosFfmpeg.sourceUrl,
  "FFmpeg corresponding-source offer (release archive URL)",
);
assertContains(
  correspondingSource,
  macosFfmpeg.sourceSha256,
  "FFmpeg corresponding-source offer (release archive SHA-256)",
);

const colmapLicense = readText("licenses/COLMAP-LICENSE.txt");
for (const marker of [
  "ETH Zurich and UNC Chapel Hill",
  "Redistributions of source code",
  "Redistributions in binary form",
  "Neither the name",
]) {
  assertContains(colmapLicense, marker, "COLMAP license");
}

const notice = readText("NOTICE");
assertContains(notice, "Copyright 2026 ooolabdev", "NOTICE");
assertContains(notice, "licenses/THIRD_PARTY_NOTICES.txt", "NOTICE");
assertContains(notice, "TRADEMARK_POLICY.md", "NOTICE");

const trademark = readText("TRADEMARK_POLICY.md");
assertContains(trademark, "does not grant a license to use the OOOSplat Marks", "Trademark policy");
assertContains(trademark, "https://github.com/ooolabdev/ooosplat/issues", "Trademark policy");

const outputs = readText("GENERATED_OUTPUTS.md");
for (const term of [
  "final.ply",
  "Apache License 2.0",
  "General Public License (GPL)",
  "Lesser General Public License (LGPL)",
  "does not assign copyright ownership",
]) {
  assertContains(outputs, term, "Generated outputs policy");
}

console.log(
  "Verified MacauSplat license metadata, the LGPL FFmpeg corresponding-source offer, " +
    "Windows/Linux/macOS engine notices and distribution manifests, PlayCanvas preview " +
    "notices, and Mediabunny video export notices.",
);
