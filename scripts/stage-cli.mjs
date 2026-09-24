#!/usr/bin/env node
/**
 * 把 CLI（splatstudio）放到 Tauri 的 externalBin 暫存位置。
 *
 * 為什麼需要這一步：
 *   MacauSplat 的自動標註流水線（inspect / fit-box / annotate / monitor / cloud）
 *   是一支獨立的命令列工具。Tauri 只會打包「主程式」，額外的執行檔必須透過
 *   `bundle.externalBin` 宣告，而且檔名必須帶目標平台的三元組後綴
 *   （例如 splatstudio-x86_64-pc-windows-msvc.exe），Tauri 安裝時才會去掉後綴
 *   放到主程式旁邊。
 *
 * 為什麼不能交給 beforeBuildCommand 之後的 cargo：
 *   Tauri 在產生安裝包之前就會檢查 externalBin 是否存在，因此必須「先」把 CLI
 *   編譯好並複製過去。這個腳本自己呼叫 cargo（有快取，通常幾秒），順序才正確。
 */
import { execFileSync } from "node:child_process";
import { copyFileSync, existsSync, mkdirSync, statSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const workspace = join(dirname(fileURLToPath(import.meta.url)), "..");
const targetDir = join(workspace, "src-tauri", "target", "release");
const stagingDir = join(workspace, "src-tauri", "binaries");

/** 取得目前工具鏈的目標三元組（例如 x86_64-pc-windows-msvc）。 */
function hostTriple() {
  const output = execFileSync("rustc", ["-vV"], { encoding: "utf8" });
  const line = output.split("\n").find((entry) => entry.startsWith("host:"));
  if (!line) throw new Error("無法從 rustc -vV 取得 host 三元組");
  return line.slice("host:".length).trim();
}

const triple = hostTriple();
const isWindows = process.platform === "win32";
const binaryName = isWindows ? "splatstudio.exe" : "splatstudio";
const stagedName = `splatstudio-${triple}${isWindows ? ".exe" : ""}`;

console.log(`[stage-cli] 目標三元組：${triple}`);

mkdirSync(stagingDir, { recursive: true });
const destination = join(stagingDir, stagedName);

// 雞生蛋問題：Tauri 的 build script 在 cargo 編譯期間就會檢查 externalBin 是否存在，
// 但該檔案正是這次 cargo 要產出的。先放一個佔位檔讓建置腳本通過，編譯完再用真的二進位覆蓋。
writeFileSync(destination, "");

// 只編譯 CLI 這一支，避免連主程式一起重編
execFileSync("cargo", ["build", "--release", "--bin", "splatstudio"], {
  cwd: join(workspace, "src-tauri"),
  stdio: "inherit",
});

const source = join(targetDir, binaryName);
if (!existsSync(source)) {
  throw new Error(`找不到編譯產物：${source}`);
}

copyFileSync(source, destination);
const size = statSync(destination).size;
if (size < 1024) {
  throw new Error(`暫存的 CLI 過小（${size} bytes），可能仍是佔位檔`);
}
console.log(`[stage-cli] 已暫存：${destination}（${(size / 1048576).toFixed(1)} MB）`);
console.log(`[stage-cli] 安裝後會與主程式並列為 ${binaryName}`);
