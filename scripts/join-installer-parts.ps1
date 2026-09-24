<#
.SYNOPSIS
    Join and verify the chunked MacauSplat Windows installer.

.DESCRIPTION
    Why the installer is split: the 505 MB installer cannot be transferred in one
    piece on this network (1 MB transfers work, 20 MB succeeds, but 505 MB was
    reset three times between 5 MB and 30 MB). Split into 60 MB chunks each single
    transfer takes about 30 seconds, which is inside the stable range.

    This script concatenates the *.partN chunks in numeric order, checks that the
    chunk numbering is gapless and that the merged length equals the sum of the
    chunk lengths, then compares the SHA256 and only keeps the output file when it
    matches - so a file that merely looks complete cannot be mistaken for a good
    installer.

    Verification is mandatory by design. The SHA256 is the only loss-detection
    mechanism a chunked transfer has, so when no hash is supplied this script
    keeps the merged file but states plainly that it is UNVERIFIED and exits with
    code 2 instead of reporting success.

.PARAMETER PartsDirectory
    Directory containing the downloaded chunks. Also where the output is written.

.PARAMETER Output
    Output file name. When omitted it is derived from the version in package.json,
    so this script cannot drift from the released version.

.PARAMETER ExpectedSha256
    The 64-hex-character SHA256 published for the installer.

.PARAMETER ChecksumFile
    Path to a SHA256SUMS.txt in sha256sum format ("<hash>  <file name>").
    Use -ChecksumName when the recorded name differs from the output name.

.PARAMETER ChecksumName
    File name to look up inside -ChecksumFile. Defaults to the output file name.

.PARAMETER AllowUnverified
    Explicitly accept an unverified merge. Requires -ConfirmUnverified and prints a
    warning; never claims the result is usable.

.PARAMETER ConfirmUnverified
    Second confirmation for -AllowUnverified.

.EXAMPLE
    .\join-installer-parts.ps1 -PartsDirectory "$env:USERPROFILE\Downloads" `
        -ChecksumFile "$env:USERPROFILE\Downloads\SHA256SUMS.txt"

.EXAMPLE
    .\join-installer-parts.ps1 -PartsDirectory "$env:USERPROFILE\Downloads" `
        -ExpectedSha256 3f2a...c9

.EXAMPLE
    # Offline smoke test only. Verification is skipped, so the result is UNVERIFIED.
    .\join-installer-parts.ps1 -PartsDirectory .\parts -AllowUnverified -ConfirmUnverified
#>
[CmdletBinding(DefaultParameterSetName = "ByHash")]
param(
    [Parameter(Mandatory = $true)]
    [string]$PartsDirectory,

    [string]$Output = "",

    [Parameter(ParameterSetName = "ByHash", Mandatory = $true)]
    [ValidatePattern('^[0-9a-fA-F]{64}$')]
    [string]$ExpectedSha256,

    [Parameter(ParameterSetName = "ByFile", Mandatory = $true)]
    [string]$ChecksumFile,

    [Parameter(ParameterSetName = "ByFile")]
    [string]$ChecksumName = "",

    [Parameter(ParameterSetName = "Unverified", Mandatory = $true)]
    [switch]$AllowUnverified,

    # Two switches, not one: reaching the unverified path must be a deliberate act,
    # because that path cannot detect a damaged transfer.
    [Parameter(ParameterSetName = "Unverified", Mandatory = $true)]
    [switch]$ConfirmUnverified
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

# Each chunk file name is "<anything>part<digits>[.<ext>]". The index is captured
# with \d+ so part1, part09, part001 and part10..part99 all work: a pattern limited
# to two digits cannot express "10 or more chunks", and lexical sorting alone would
# place part10 between part1 and part2.
$partPattern = '^(?<stem>.+?)[._-]?part(?<index>\d+)(?:\.(?:bin|part|exe))?$'

function Get-InstallerParts {
    param([string]$Directory)

    $groups = @{}
    foreach ($file in (Get-ChildItem -LiteralPath $Directory -File)) {
        if ($file.Name -notmatch $partPattern) { continue }
        $stem = $Matches['stem']
        $index = [int]$Matches['index']
        if (-not $groups.ContainsKey($stem)) { $groups[$stem] = @() }
        $groups[$stem] += [pscustomobject]@{ Index = $index; File = $file }
    }

    if ($groups.Count -eq 0) {
        throw "No chunk files were found in $Directory (expected names such as *.part1, *.part01 or *.part001). Download all chunks first."
    }

    # When several chunk sets are present, use the set with the most chunks and list
    # the others, so chunks from two different installers are never merged.
    $ordered = @($groups.GetEnumerator() | Sort-Object { $_.Value.Count } -Descending)
    $chosen = $ordered[0]
    if ($ordered.Count -gt 1) {
        $others = ($ordered | Select-Object -Skip 1 | ForEach-Object { $_.Key + ' (' + $_.Value.Count + ' chunks)' }) -join '; '
        Write-Host "Note: several chunk sets were found. Using '$($chosen.Key)' ($($chosen.Value.Count) chunks); ignoring: $others" -ForegroundColor Yellow
    }

    return [pscustomobject]@{
        Stem  = $chosen.Key
        Parts = @($chosen.Value | Sort-Object Index)
    }
}

function Get-DefaultOutputName {
    param([string]$Override)

    if ($Override) { return $Override }

    $packagePath = Join-Path $PSScriptRoot "..\package.json"
    if (-not (Test-Path -LiteralPath $packagePath -PathType Leaf)) {
        throw "Cannot read $packagePath to derive the output file name. Pass -Output explicitly."
    }
    $version = (Get-Content -LiteralPath $packagePath -Raw | ConvertFrom-Json).version
    if (-not $version) { throw "$packagePath has no version field. Pass -Output explicitly." }
    return "MacauSplat-$version-x64-windows.exe"
}

# Streams the file through SHA256 so a 505 MB installer is never read into memory.
function Get-StreamSha256 {
    param([string]$Path)

    $stream = [System.IO.File]::OpenRead($Path)
    try {
        $sha = [System.Security.Cryptography.SHA256]::Create()
        try { return ([System.BitConverter]::ToString($sha.ComputeHash($stream))).Replace('-', '').ToLowerInvariant() }
        finally { $sha.Dispose() }
    }
    finally { $stream.Dispose() }
}

$directory = (Resolve-Path -LiteralPath $PartsDirectory).Path
$selection = Get-InstallerParts -Directory $directory
$parts = $selection.Parts

Write-Host "Found $($parts.Count) chunks (prefix '$($selection.Stem)'):" -ForegroundColor Cyan
foreach ($part in $parts) {
    Write-Host ("  #{0,-3} {1}  {2,8:N1} MB" -f $part.Index, $part.File.Name, ($part.File.Length / 1MB))
}

# Numbering must run 1..N with no gaps: one missing chunk yields a file that looks
# complete but is corrupt.
$expectedIndexes = @(1..$parts.Count)
$actualIndexes = @($parts | ForEach-Object { $_.Index })
$missing = @($expectedIndexes | Where-Object { $actualIndexes -notcontains $_ })
if ($missing.Count -gt 0) {
    throw "Chunk numbering is not contiguous: chunk(s) $($missing -join ', ') are missing out of $($parts.Count). Download the missing chunks again."
}
if ($actualIndexes[0] -ne 1) {
    throw "Chunk numbering must start at 1, but starts at $($actualIndexes[0])."
}

# Measure-Object cannot sum a nested property such as File.Length, so total the
# chunk sizes directly.
$totalBytes = 0
foreach ($part in $parts) { $totalBytes += $part.File.Length }
if ($totalBytes -le 0) { throw "The chunks are empty; nothing to merge." }

$outputName = Get-DefaultOutputName -Override $Output
$outputPath = Join-Path $directory $outputName

# ---------------------------------------------------------------- expected hash
# Resolved BEFORE merging on purpose: if the caller asked for verification and the
# reference hash cannot be located, failing here leaves no output file behind at
# all, instead of a merged file that was never checked.
$expected = $null
$verificationSource = $null

if ($PSCmdlet.ParameterSetName -eq "ByFile") {
    $wanted = if ($ChecksumName) { $ChecksumName } else { $outputName }
    if (-not (Test-Path -LiteralPath $ChecksumFile -PathType Leaf)) {
        throw "Checksum file not found: $ChecksumFile"
    }
    $checksumPath = (Resolve-Path -LiteralPath $ChecksumFile).Path
    # Parse "hash  name" / "hash *name" directly. Select-String yields MatchInfo
    # objects, which do not split the way the raw line does.
    foreach ($rawLine in (Get-Content -LiteralPath $checksumPath)) {
        $line = $rawLine.Trim()
        if ($line -notmatch '^(?<hash>[0-9a-fA-F]{64})\s+\*?(?<name>.+)$') { continue }
        if ($Matches['name'].Trim() -ne $wanted) { continue }
        $expected = $Matches['hash'].ToLowerInvariant()
        break
    }
    if (-not $expected) {
        throw "No SHA256 record for '$wanted' was found in $checksumPath. Check -ChecksumName or use -ExpectedSha256. Nothing was merged."
    }
    $verificationSource = $checksumPath
    Write-Host "Expected SHA256 ($verificationSource): $expected" -ForegroundColor Cyan
}
elseif ($PSCmdlet.ParameterSetName -eq "ByHash") {
    $expected = $ExpectedSha256.ToLowerInvariant()
    $verificationSource = "-ExpectedSha256"
    Write-Host "Expected SHA256 ($verificationSource): $expected" -ForegroundColor Cyan
}

if (Test-Path -LiteralPath $outputPath) {
    Write-Host "Removing existing output file: $outputPath" -ForegroundColor Yellow
    Remove-Item -LiteralPath $outputPath -Force
}

Write-Host ""
Write-Host ("Merging... (total {0:N1} MB)" -f ($totalBytes / 1MB)) -ForegroundColor Cyan
# The stream variables must not be called $output/$input: $Output is a script
# parameter and a local $output silently shadows it, and [System.IO.File]::Create
# returns a FileStream that PowerShell would otherwise leak into the pipeline.
$outputStream = $null
$outputStream = [System.IO.File]::Create($outputPath)
if ($null -eq $outputStream) { throw "Cannot create the output file: $outputPath" }
try {
    foreach ($part in $parts) {
        $inputStream = $null
        $inputStream = [System.IO.File]::OpenRead($part.File.FullName)
        if ($null -eq $inputStream) { throw "Cannot read chunk: $($part.File.FullName)" }
        try { $inputStream.CopyTo($outputStream, 1MB) }
        finally { $inputStream.Dispose() }
    }
}
finally { $outputStream.Dispose() }

$size = (Get-Item -LiteralPath $outputPath).Length
if ($size -ne $totalBytes) {
    Remove-Item -LiteralPath $outputPath -Force
    throw "Merged size $size does not equal the chunk total $totalBytes. The output was deleted; download the chunks again."
}
Write-Host ("Merged: {0} ({1:N1} MB)" -f $outputName, ($size / 1MB)) -ForegroundColor Green

# ------------------------------------------------------------ verify or warn
if ($expected) {
    Write-Host ""
    Write-Host "Computing SHA256..." -ForegroundColor Cyan
    $actual = Get-StreamSha256 -Path $outputPath
    if ($actual -ne $expected) {
        Write-Host "FAILED: SHA256 mismatch. The file is corrupt or incomplete." -ForegroundColor Red
        Write-Host "  expected (source: $verificationSource): $expected"
        Write-Host "  actual:   $actual"
        Remove-Item -LiteralPath $outputPath -Force
        throw "SHA256 mismatch. The merged output was deleted; download the chunks again."
    }
    Write-Host "OK: SHA256 verified: $actual" -ForegroundColor Green
    Write-Host "  compared against: $verificationSource" -ForegroundColor Green
}
else {
    # Deliberate: with no hash supplied the script must say that nothing was
    # checked rather than let the user assume the file was verified.
    Write-Host ""
    Write-Host "WARNING: UNVERIFIED. Neither -ExpectedSha256 nor -ChecksumFile was supplied." -ForegroundColor Yellow
    Write-Host "  A chunked merge cannot detect transfer corruption, so this file may be incomplete or corrupt." -ForegroundColor Yellow
    Write-Host "  Re-run with the SHA256SUMS.txt published for the release:" -ForegroundColor Yellow
    Write-Host "    .\join-installer-parts.ps1 -PartsDirectory `"$directory`" -ChecksumFile <path to SHA256SUMS.txt>" -ForegroundColor Yellow
    Write-Host "  Output written: $outputPath (status: UNVERIFIED)" -ForegroundColor Yellow
    exit 2
}

Write-Host ""
Write-Host "Ready to run: $outputPath" -ForegroundColor Green
Write-Host "Note: the installer is not code-signed, so SmartScreen will warn. Choose 'More info' then 'Run anyway'." -ForegroundColor Yellow
