[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$workspace = Split-Path -Parent $PSScriptRoot

function Resolve-WorkspacePath {
    param([Parameter(Mandatory = $true)][string]$RelativePath)

    Join-Path $workspace ($RelativePath -replace "/", [IO.Path]::DirectorySeparatorChar)
}

function Read-Utf8Text {
    param([Parameter(Mandatory = $true)][string]$RelativePath)

    Get-Content -LiteralPath (Resolve-WorkspacePath $RelativePath) -Raw -Encoding UTF8
}

function Assert-True {
    param(
        [Parameter(Mandatory = $true)][bool]$Condition,
        [Parameter(Mandatory = $true)][string]$Message
    )

    if (-not $Condition) {
        throw $Message
    }
}

function Assert-Contains {
    param(
        [Parameter(Mandatory = $true)][string]$Text,
        [Parameter(Mandatory = $true)][string]$Expected,
        [Parameter(Mandatory = $true)][string]$Context
    )

    Assert-True ($Text.Contains($Expected)) "$Context is missing '$Expected'."
}

function Get-NormalizedSha256 {
    param([Parameter(Mandatory = $true)][string]$Text)

    $normalized = $Text.Replace("`r`n", "`n").Replace("`r", "`n").TrimEnd("`n")
    $bytes = [Text.Encoding]::UTF8.GetBytes($normalized)
    $sha = [Security.Cryptography.SHA256]::Create()
    try {
        return ([BitConverter]::ToString($sha.ComputeHash($bytes))).Replace("-", "")
    } finally {
        $sha.Dispose()
    }
}

$requiredFiles = @(
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
    "engines/manifest.macos.json"
)

foreach ($relativePath in $requiredFiles) {
    Assert-True (Test-Path -LiteralPath (Resolve-WorkspacePath $relativePath) -PathType Leaf) "Required license file is missing: $relativePath"
}

$apacheText = Read-Utf8Text "LICENSE"
$expectedApacheSha256 = "58D1E17FFE5109A7AE296CAAFCADFDBE6A7D176F0BC4AB01E12A689B0499D8BD"
Assert-True ((Get-NormalizedSha256 $apacheText) -eq $expectedApacheSha256) "LICENSE is not the unmodified Apache License 2.0 text."

$package = (Read-Utf8Text "package.json") | ConvertFrom-Json
Assert-True ($package.license -eq "Apache-2.0") "package.json must declare Apache-2.0."
# The distributable is MacauSplat, so package.json must identify this project and
# its own repository. Reusing the upstream identity here would make the shipped
# metadata claim to be OOOSplat, contradicting NOTICE and docs/COMPLIANCE.md.
Assert-True ($package.author -eq "Linference") "package.json must identify Linference as author."
Assert-True ($package.repository.url -eq "git+https://github.com/Linference/MacauSplat.git") "package.json repository URL must identify this fork, not upstream OOOSplat."
Assert-True ($package.upstream.url -eq "https://github.com/ooolabdev/ooosplat") "package.json must record the upstream OOOSplat project under an explicit upstream field."

$cargo = Read-Utf8Text "src-tauri/Cargo.toml"
Assert-True ([regex]::IsMatch($cargo, '(?m)^license\s*=\s*"Apache-2\.0"\s*$')) "Cargo.toml must declare Apache-2.0."
Assert-True ([regex]::IsMatch($cargo, '(?m)^authors\s*=\s*\["ooolabdev"\]\s*$')) "Cargo.toml must identify ooolabdev as author."
Assert-True ([regex]::IsMatch($cargo, '(?m)^repository\s*=\s*"https://github\.com/ooolabdev/ooosplat"\s*$')) "Cargo.toml repository URL is incorrect."

$tauri = (Read-Utf8Text "src-tauri/tauri.conf.json") | ConvertFrom-Json
Assert-True ($tauri.bundle.license -eq "Apache-2.0") "Tauri bundle license must be Apache-2.0."
Assert-True ($tauri.bundle.licenseFile -eq "../LICENSE") "Tauri bundle licenseFile must point to ../LICENSE."
Assert-True ($tauri.productName -eq "MacauSplat") "Tauri productName must be MacauSplat."
$resourceNames = @($tauri.bundle.resources.PSObject.Properties.Name)
foreach ($resource in "../LICENSE", "../NOTICE", "../TRADEMARK_POLICY.md", "../GENERATED_OUTPUTS.md", "../licenses/THIRD_PARTY_NOTICES.txt", "../licenses/FFmpeg-LGPL-2.1.txt", "../licenses/FFmpeg-CORRESPONDING-SOURCE.txt", "../licenses/COLMAP-LICENSE.txt", "../licenses/NVIDIA-CUDA-Runtime.txt", "../licenses/Brush-LICENSE.txt", "../licenses/PlayCanvas-MIT.txt", "../licenses/Mediabunny-MPL-2.0.txt") {
    Assert-True ($resourceNames -contains $resource) "Tauri resources are missing $resource."
}

# LGPL-2.1 Section 6 makes the corresponding-source offer a condition of
# distributing the LGPL FFmpeg object code, so it must be in every platform
# bundle, not only in the base config.
foreach ($platformConfig in "src-tauri/tauri.windows.conf.json", "src-tauri/tauri.linux.conf.json", "src-tauri/tauri.macos.conf.json") {
    $platformTauri = (Read-Utf8Text $platformConfig) | ConvertFrom-Json
    $platformResources = @($platformTauri.bundle.resources.PSObject.Properties.Name)
    Assert-True ($platformResources -contains "../licenses/FFmpeg-CORRESPONDING-SOURCE.txt") "$platformConfig resources are missing the LGPL corresponding-source offer."
}

$tauriWindows = (Read-Utf8Text "src-tauri/tauri.windows.conf.json") | ConvertFrom-Json
Assert-True (@($tauriWindows.bundle.resources.PSObject.Properties.Name) -contains "../engines/manifest.json") "Windows Tauri resources are missing the Windows engine manifest."
$tauriLinux = (Read-Utf8Text "src-tauri/tauri.linux.conf.json") | ConvertFrom-Json
Assert-True ($tauriLinux.bundle.active -eq $true) "Linux Tauri bundling must be enabled."
Assert-True (@($tauriLinux.bundle.targets) -contains "deb") "Linux Tauri targets must include deb."
$linuxResources = @($tauriLinux.bundle.resources.PSObject.Properties.Name)
foreach ($resource in "../engines/manifest.linux.json", "../engines/linux/brush/") {
    Assert-True ($linuxResources -contains $resource) "Linux Tauri resources are missing $resource."
}
foreach ($dependency in "ffmpeg", "colmap") {
    Assert-True (@($tauriLinux.bundle.linux.deb.depends) -contains $dependency) "Linux DEB dependencies are missing $dependency."
}
$tauriMacos = (Read-Utf8Text "src-tauri/tauri.macos.conf.json") | ConvertFrom-Json
Assert-True ($tauriMacos.bundle.macOS.minimumSystemVersion -eq "15.0") "macOS bundle must target macOS 15.0."
$macosResources = @($tauriMacos.bundle.resources.PSObject.Properties.Name)
foreach ($resource in "../engines/manifest.macos.json", "../engines/macos/arm64/") {
    Assert-True ($macosResources -contains $resource) "macOS Tauri resources are missing $resource."
}

$manifest = (Read-Utf8Text "engines/manifest.json") | ConvertFrom-Json
Assert-True ($manifest.schemaVersion -ge 2) "Engine manifest schemaVersion must include license mappings."
Assert-True ($manifest.engines.Count -eq 3) "License verification expects exactly the three direct native engines."

$expectedEngines = @{
    "FFmpeg / FFprobe" = @{
        License = "LGPL-2.1-or-later"
        File = "licenses/FFmpeg-LGPL-2.1.txt"
    }
    "COLMAP" = @{
        License = "BSD-3-Clause"
        File = "licenses/COLMAP-LICENSE.txt"
    }
    "Brush" = @{
        License = "Apache-2.0"
        File = "licenses/Brush-LICENSE.txt"
    }
}

$thirdParty = Read-Utf8Text "licenses/THIRD_PARTY_NOTICES.txt"
Assert-True (-not $thirdParty.Contains("OOOSplat 0.2.0")) "Third-party notices still contain the obsolete 0.2.0 heading."
Assert-True (-not (Test-Path -LiteralPath (Resolve-WorkspacePath "licenses/FFmpeg-LICENSE.txt"))) "The obsolete LGPLv3 FFmpeg-LICENSE.txt file must not exist."

foreach ($engine in $manifest.engines) {
    Assert-True ($expectedEngines.ContainsKey($engine.name)) "Unexpected direct engine in manifest: $($engine.name)"
    $expected = $expectedEngines[$engine.name]
    Assert-True ($engine.license.StartsWith($expected.License)) "$($engine.name) license identifier does not match $($expected.License)."
    Assert-True (@($engine.licenseFiles).Count -eq 1) "$($engine.name) must map to one direct license file."
    Assert-True ($engine.licenseFiles[0] -eq $expected.File) "$($engine.name) license file mapping is incorrect."
    Assert-True (Test-Path -LiteralPath (Resolve-WorkspacePath $expected.File) -PathType Leaf) "$($engine.name) mapped license file is missing."
    Assert-Contains $thirdParty $engine.name "THIRD_PARTY_NOTICES.txt"
    Assert-Contains $thirdParty $expected.License "THIRD_PARTY_NOTICES.txt"
    Assert-Contains $thirdParty $expected.File "THIRD_PARTY_NOTICES.txt"
}

$windowsFfmpeg = @($manifest.engines | Where-Object { $_.name -eq "FFmpeg / FFprobe" })
Assert-True ($windowsFfmpeg.Count -eq 1) "Windows FFmpeg manifest entry is missing or duplicated."
Assert-True (-not $windowsFfmpeg[0].sourceUrl.Contains("/latest/")) "Windows FFmpeg source URL must reference an immutable release asset, not /latest/."
Assert-Contains $thirdParty $windowsFfmpeg[0].actualVersion "THIRD_PARTY_NOTICES.txt"
Assert-Contains $thirdParty $windowsFfmpeg[0].sourceUrl "THIRD_PARTY_NOTICES.txt"

$linuxManifest = (Read-Utf8Text "engines/manifest.linux.json") | ConvertFrom-Json
Assert-True ($linuxManifest.schemaVersion -ge 2) "Linux engine manifest schemaVersion must include license mappings."
Assert-True ($linuxManifest.brush.version -eq "0.3.0") "Linux Brush version is incorrect."
Assert-True ($linuxManifest.brush.sourceUrl -eq "https://github.com/ArthurBrussee/brush/releases/download/v0.3.0/brush-app-x86_64-unknown-linux-gnu.tar.xz") "Linux Brush release archive is incorrect."
Assert-True ($linuxManifest.brush.license -eq "Apache-2.0") "Linux Brush license identifier is incorrect."
Assert-True (@($linuxManifest.brush.licenseFiles).Count -eq 1) "Linux Brush must map to one direct license file."
Assert-True ($linuxManifest.brush.licenseFiles[0] -eq "licenses/Brush-LICENSE.txt") "Linux Brush license file mapping is incorrect."
foreach ($marker in "Ubuntu 24.04 Alpha, Linux x86_64 release archive", "brush-app-x86_64-unknown-linux-gnu.tar.xz", "engines/manifest.linux.json") {
    Assert-Contains $thirdParty $marker "THIRD_PARTY_NOTICES.txt"
}

$macosManifest = (Read-Utf8Text "engines/manifest.macos.json") | ConvertFrom-Json
Assert-True ($macosManifest.schemaVersion -ge 1) "macOS engine manifest schemaVersion is missing."
Assert-True ($macosManifest.platform -eq "macos") "macOS engine manifest platform is incorrect."
Assert-True ($macosManifest.architecture -eq "arm64") "macOS engine manifest must be Apple arm64 only."
Assert-True ($macosManifest.minimumSystemVersion -eq "15.0") "macOS engine manifest must target macOS 15.0."
Assert-True ($macosManifest.buildEnvironment.homebrewCoreCommit -match '^[A-F0-9]{40}$') "macOS Homebrew/core build commit is not pinned."
Assert-True ($macosManifest.buildEnvironment.runner -eq "macos-15") "macOS engine runner must be pinned to macos-15."
Assert-True ($macosManifest.buildEnvironment.usePinnedHomebrewBottles -eq $true) "macOS build dependencies must use pinned Homebrew bottles."
Assert-True ($macosManifest.engines.Count -eq 3) "macOS manifest must contain the three direct engines."

# The macOS runtime is fetched by scripts/setup-engines-macos.sh from the
# `distribution` block. Nothing used to verify that block, so a release URL that
# pointed at a tag which does not exist passed every check and made the macOS CI
# job fail at download time instead.
$macosDistribution = $macosManifest.distribution
Assert-True ($null -ne $macosDistribution) "macOS manifest is missing the distribution block."
foreach ($key in "tag", "archiveName", "sourceUrl", "archiveSha256File", "destination", "integrityFile", "publishedBy", "buildFallback", "archiveLayoutRoot") {
    $value = $macosDistribution.$key
    Assert-True ($value -is [string] -and $value.Length -gt 0) "macOS distribution.$key must be a non-empty string."
}
Assert-True ($macosDistribution.tag -match '^[A-Za-z0-9][A-Za-z0-9._-]*$') "macOS distribution.tag is not a valid git tag: $($macosDistribution.tag)"
Assert-True ($macosDistribution.tag.StartsWith("engines-macos-arm64-v")) "macOS distribution.tag must use the engines-macos-arm64-v prefix that macos-engines.yml publishes."
Assert-True ($macosDistribution.archiveName -match '^[A-Za-z0-9][A-Za-z0-9._-]*\.tar\.xz$') "macOS distribution.archiveName must be a .tar.xz archive name."
Assert-True ($macosDistribution.archiveName.StartsWith("macausplat-")) "macOS distribution.archiveName must use this project's name; released artifact names are brand elements."
Assert-True ($macosDistribution.archiveName.Contains($macosDistribution.tag.Substring("engines-macos-arm64-".Length))) "macOS distribution.archiveName must contain the version from distribution.tag."
Assert-True ($macosDistribution.destination -eq "engines/macos/arm64") "macOS distribution.destination must be engines/macos/arm64 to match the Tauri resource mapping."
Assert-True ($macosDistribution.sourceUrl -eq "https://github.com/Linference/MacauSplat/releases/download/$($macosDistribution.tag)/$($macosDistribution.archiveName)") "macOS distribution.sourceUrl must resolve to this project's own release asset, built from the tag it names."
Assert-True ($macosDistribution.archiveSha256File -eq "$($macosDistribution.sourceUrl).sha256") "macOS distribution.archiveSha256File must be the .sha256 sibling of distribution.sourceUrl."
Assert-True (-not $macosDistribution.sourceUrl.Contains("/latest/")) "macOS distribution.sourceUrl must reference an immutable release asset, not /latest/."
Assert-True ($macosDistribution.buildFallback -eq "scripts/build-engines-macos.sh") "macOS distribution.buildFallback must name the reproducible local build script."
Assert-True (Test-Path -LiteralPath (Resolve-WorkspacePath "scripts/build-engines-macos.sh") -PathType Leaf) "macOS distribution.buildFallback script is missing."
$macosEnginesWorkflow = Read-Utf8Text ".github/workflows/macos-engines.yml"
Assert-Contains $macosEnginesWorkflow "distribution.tag" "macos-engines.yml"
Assert-Contains $macosEnginesWorkflow "engines-macos-arm64-v" "macos-engines.yml (the tag prefix the workflow publishes)"
Assert-Contains (Read-Utf8Text "scripts/setup-engines-macos.sh") "distribution.archiveLayoutRoot" "scripts/setup-engines-macos.sh (must read distribution.archiveLayoutRoot from the manifest)"
Assert-Contains (Read-Utf8Text "scripts/build-engines-macos.sh") "distribution.archiveLayoutRoot" "scripts/build-engines-macos.sh (must read distribution.archiveLayoutRoot from the manifest)"
Assert-True ($macosDistribution.archiveLayoutRoot -eq "macausplat-engines-macos-arm64") "macOS distribution.archiveLayoutRoot must match the directory the engine archive extracts to."
Assert-Contains (Read-Utf8Text "licenses/THIRD_PARTY_NOTICES.txt") "scripts/setup-engines-macos.sh" "THIRD_PARTY_NOTICES.txt (macOS acquisition chain)"

foreach ($engine in $macosManifest.engines) {
    Assert-True ($expectedEngines.ContainsKey($engine.name)) "Unexpected macOS engine: $($engine.name)"
    $expected = $expectedEngines[$engine.name]
    Assert-True ($engine.license -eq $expected.License) "$($engine.name) macOS license identifier is incorrect."
    Assert-True (@($engine.licenseFiles).Count -eq 1 -and $engine.licenseFiles[0] -eq $expected.File) "$($engine.name) macOS license mapping is incorrect."
    Assert-True ($engine.sourceSha256 -match '^[A-F0-9]{64}$') "$($engine.name) macOS source SHA-256 is invalid."
    Assert-True ($engine.sourceUrl -match '^https://') "$($engine.name) macOS source URL must be an absolute https URL."
    Assert-True ($null -ne $engine.buildPolicy -and $engine.buildPolicy.Length -gt 0) "$($engine.name) macOS buildPolicy must describe how the runtime was produced so the notices can be checked against it."
}
foreach ($marker in "macOS 15+ Apple Silicon arm64", "ffmpeg-8.1.2.tar.xz", "COLMAP 4.0.4 macOS arm64 CPU CLI-only", "brush-app-aarch64-apple-darwin.tar.xz", "engines/manifest.macos.json") {
    Assert-Contains $thirdParty $marker "THIRD_PARTY_NOTICES.txt"
}

foreach ($component in "PlayCanvas Engine", "PlayCanvas React", "PlayCanvas splat-transform (adapted algorithm)", "SuperSplat selection workflow (adapted algorithm)") {
    Assert-Contains $thirdParty $component "THIRD_PARTY_NOTICES.txt"
}
Assert-Contains $thirdParty "licenses/PlayCanvas-MIT.txt" "THIRD_PARTY_NOTICES.txt"
$playCanvasLicense = Read-Utf8Text "licenses/PlayCanvas-MIT.txt"
Assert-Contains $playCanvasLicense "MIT License" "PlayCanvas license"
Assert-Contains $playCanvasLicense "Copyright (c) 2011-2026 PlayCanvas Ltd." "PlayCanvas license"

foreach ($term in "Mediabunny", "1.52.3", "MPL-2.0", "licenses/Mediabunny-MPL-2.0.txt", "https://github.com/Vanilagy/mediabunny/tree/v1.52.3") {
    Assert-Contains $thirdParty $term "THIRD_PARTY_NOTICES.txt"
}
$mediabunnyLicense = Read-Utf8Text "licenses/Mediabunny-MPL-2.0.txt"
Assert-Contains $mediabunnyLicense "Mozilla Public License Version 2.0" "Mediabunny license"
Assert-Contains $mediabunnyLicense "2. License Grants and Conditions" "Mediabunny license"

foreach ($term in "image crate", "0.25.10", "MIT OR Apache-2.0", "https://github.com/image-rs/image/tree/v0.25.10") {
    Assert-Contains $thirdParty $term "THIRD_PARTY_NOTICES.txt"
}
$cargoToml = Read-Utf8Text "src-tauri/Cargo.toml"
Assert-Contains $cargoToml 'image = { version = "0.25"' "Cargo.toml"

$ffmpegLicense = Read-Utf8Text "licenses/FFmpeg-LGPL-2.1.txt"
Assert-Contains $ffmpegLicense "GNU LESSER GENERAL PUBLIC LICENSE" "FFmpeg license"
Assert-Contains $ffmpegLicense "Version 2.1, February 1999" "FFmpeg license"
Assert-True (-not $ffmpegLicense.Contains("Version 3, 29 June 2007")) "FFmpeg license still contains the LGPLv3 text."

# LGPL-2.1 Section 6 requires a written offer of the corresponding source when the
# LGPL object code is distributed without it. Shipping only the license text and a
# link to ffmpeg.org does not discharge that obligation.
$correspondingSource = Read-Utf8Text "licenses/FFmpeg-CORRESPONDING-SOURCE.txt"
foreach ($marker in "Written offer under GNU Lesser General Public License v2.1, Section 6", "How to obtain the corresponding source", "How this Object Code was built", "Relinking (LGPL v2.1 Section 6(b))", "https://github.com/Linference/MacauSplat/issues", "win64-lgpl-shared", "variants/win64-lgpl-shared.sh", "variants/defaults-lgpl.sh", "LICENSE_FILE=COPYING.LGPLv3", "build.sh", "ffmpeg.exe -buildconf") {
    Assert-Contains $correspondingSource $marker "FFmpeg corresponding-source offer"
}
$windowsFfmpeg = @($manifest.engines | Where-Object { $_.name -eq "FFmpeg / FFprobe" })[0]
Assert-Contains $correspondingSource $windowsFfmpeg.sourceUrl "FFmpeg corresponding-source offer"
Assert-Contains $correspondingSource $windowsFfmpeg.archiveSha256 "FFmpeg corresponding-source offer"
Assert-Contains $correspondingSource $windowsFfmpeg.actualVersion "FFmpeg corresponding-source offer"
Assert-Contains $thirdParty "licenses/FFmpeg-CORRESPONDING-SOURCE.txt" "THIRD_PARTY_NOTICES.txt"
Assert-True ($resourceNames -contains "../licenses/FFmpeg-CORRESPONDING-SOURCE.txt") "Tauri resources are missing the LGPL corresponding-source offer."
# The release source archive the offer names is the same archive the macOS engine
# manifest pins, so the two records must not drift apart.
$macosFfmpeg = @($macosManifest.engines | Where-Object { $_.name -eq "FFmpeg / FFprobe" })[0]
Assert-Contains $correspondingSource $macosFfmpeg.sourceUrl "FFmpeg corresponding-source offer (release archive URL)"
Assert-Contains $correspondingSource $macosFfmpeg.sourceSha256 "FFmpeg corresponding-source offer (release archive SHA-256)"

$colmapLicense = Read-Utf8Text "licenses/COLMAP-LICENSE.txt"
Assert-Contains $colmapLicense "ETH Zurich and UNC Chapel Hill" "COLMAP license"
Assert-Contains $colmapLicense "Redistributions of source code" "COLMAP license"
Assert-Contains $colmapLicense "Redistributions in binary form" "COLMAP license"
Assert-Contains $colmapLicense "Neither the name" "COLMAP license"

$notice = Read-Utf8Text "NOTICE"
Assert-Contains $notice "Copyright 2026 ooolabdev" "NOTICE"
Assert-Contains $notice "Copyright 2026 Linference" "NOTICE"
Assert-Contains $notice "licenses/THIRD_PARTY_NOTICES.txt" "NOTICE"
Assert-Contains $notice "TRADEMARK_POLICY.md" "NOTICE"
# The change list must be reproducible rather than a hand-maintained subset.
Assert-Contains $notice "git diff --name-status upstream/main...HEAD" "NOTICE (reproducible change list)"
# The statement of changes must not claim that nothing was removed: three upstream
# brand assets were deleted, and Apache-2.0 Section 4(b) requires that to be stated.
Assert-True (-not $notice.Contains("No upstream functionality was removed")) "NOTICE must not claim that nothing was removed; upstream brand assets were deleted."
foreach ($removed in "assets/app-icon.svg", "assets/app-icon-rounded.svg", "assets/readme-logo.svg") {
    Assert-Contains $notice $removed "NOTICE (removed upstream brand assets)"
}
foreach ($added in "src-tauri/src/reconstruction/autolabel.rs", "src-tauri/src/reconstruction/colmap.rs", "src-tauri/src/reconstruction/inspect.rs", "src-tauri/src/reconstruction/cloud.rs", "src-tauri/src/pipeline/monitor.rs", "scripts/stage-cli.mjs", "scripts/join-installer-parts.ps1", "docs/DESIGN.md", "docs/COMPLIANCE.md", ".github/workflows/ubuntu.yml", ".github/workflows/macos.yml", "src-tauri/tauri.linux.conf.json", "src-tauri/tauri.macos.conf.json", "src-tauri/tauri.windows.conf.json") {
    Assert-Contains $notice $added "NOTICE (change list)"
}

$trademark = Read-Utf8Text "TRADEMARK_POLICY.md"
Assert-Contains $trademark "does not grant a license to use the OOOSplat Marks" "Trademark policy"
Assert-Contains $trademark "https://github.com/ooolabdev/ooosplat/issues" "Trademark policy"

$outputs = Read-Utf8Text "GENERATED_OUTPUTS.md"
foreach ($term in "final.ply", "Apache License 2.0", "General Public License (GPL)", "Lesser General Public License (LGPL)", "does not assign copyright ownership") {
    Assert-Contains $outputs $term "Generated outputs policy"
}

Write-Host "Verified MacauSplat license metadata, the LGPL FFmpeg corresponding-source offer, Windows/Linux/macOS engine notices and distribution manifests, PlayCanvas preview notices, and Mediabunny video export notices."
