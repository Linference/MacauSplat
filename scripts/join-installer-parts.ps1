<#
.SYNOPSIS
    合併並校驗分割成多個區塊的 MacauSplat 安裝包。

.DESCRIPTION
    為什麼安裝包要切塊：505 MB 的安裝包在這個網路環境下無法一次上傳完成
    （實測 1 MB 正常、20 MB 成功，但 505 MB 連傳三次都在 5~30 MB 處被重置）。
    切成每塊 60 MB 後單次傳輸約 30 秒，落在穩定區間內。

    本腳本會依序合併 *.partNN、驗證 SHA256，並在驗證通過後才保留輸出檔——
    避免拿到一個看似完整、實際損壞的安裝包。

.EXAMPLE
    .\join-installer-parts.ps1 -PartsDirectory "$env:USERPROFILE\Downloads"
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$PartsDirectory,

    [string]$Output = "MacauSplat_0.4.1_x64-setup.exe",

    # 發行頁上公布的 SHA256；留空則只合併不校驗
    [string]$ExpectedSha256 = ""
)

$ErrorActionPreference = "Stop"

$parts = Get-ChildItem -Path $PartsDirectory -Filter "*.part[0-9][0-9]" -File |
    Sort-Object Name

if ($parts.Count -eq 0) {
    throw "在 $PartsDirectory 找不到任何 *.partNN 檔案。請先下載全部分塊。"
}

Write-Host "找到 $($parts.Count) 個分塊：" -ForegroundColor Cyan
$parts | ForEach-Object { "  {0}  {1,6:N1} MB" -f $_.Name, ($_.Length / 1MB) }

$outputPath = Join-Path (Get-Location).Path $Output
if (Test-Path $outputPath) { Remove-Item $outputPath -Force }

Write-Host "`n合併中…" -ForegroundColor Cyan
$stream = [System.IO.File]::Create($outputPath)
try {
    foreach ($part in $parts) {
        $bytes = [System.IO.File]::ReadAllBytes($part.FullName)
        $stream.Write($bytes, 0, $bytes.Length)
    }
}
finally {
    $stream.Close()
}

$size = (Get-Item $outputPath).Length
Write-Host ("合併完成：{0}（{1:N1} MB）" -f $Output, ($size / 1MB)) -ForegroundColor Green

if ($ExpectedSha256) {
    Write-Host "`n計算 SHA256 中…" -ForegroundColor Cyan
    $actual = (Get-FileHash -Path $outputPath -Algorithm SHA256).Hash.ToLower()
    $expected = $ExpectedSha256.ToLower().Trim()
    if ($actual -ne $expected) {
        Write-Host "✗ 校驗失敗！檔案可能損壞或不完整。" -ForegroundColor Red
        Write-Host "  預期：$expected"
        Write-Host "  實際：$actual"
        Remove-Item $outputPath -Force
        throw "SHA256 不符，已刪除合併結果。請重新下載分塊。"
    }
    Write-Host "✓ SHA256 校驗通過：$actual" -ForegroundColor Green
}

Write-Host "`n可以執行了：$outputPath" -ForegroundColor Green
Write-Host "提醒：安裝包未經數位簽章，SmartScreen 會警告 → 點「更多資訊」→「仍要執行」。" -ForegroundColor Yellow
