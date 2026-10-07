$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'file-hash.ps1')
$fixture = [IO.Path]::GetTempFileName()
try {
    [IO.File]::WriteAllBytes($fixture, [Text.Encoding]::UTF8.GetBytes('abc'))
    $actual = Get-LockedFileSha256 -LiteralPath $fixture
    if ($actual -ne 'BA7816BF8F01CFEA414140DE5DAE2223B00361A396177A9CB410FF61F20015AD') {
        throw 'SHA256 fixture mismatch.'
    }
    $tokens = $null
    $errors = $null
    [Management.Automation.Language.Parser]::ParseFile((Join-Path $PSScriptRoot 'setup-engines.ps1'), [ref]$tokens, [ref]$errors) | Out-Null
    if ($errors.Count -ne 0) { throw 'Engine setup script has syntax errors.' }
    Write-Host 'SHA256 known-answer fixture and engine setup syntax passed.'
} finally {
    Remove-Item -LiteralPath $fixture -ErrorAction SilentlyContinue
}
