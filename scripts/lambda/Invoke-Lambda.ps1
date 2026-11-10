[CmdletBinding(PositionalBinding = $false)]
param(
    [string]$Config = (Join-Path $env:USERPROFILE '.config\scalar-pi\lambda.json'),
    [Parameter(Position = 0, ValueFromRemainingArguments = $true)][string[]]$Arguments
)
$ErrorActionPreference = 'Stop'
$settings = Get-Content -LiteralPath $Config -Raw | ConvertFrom-Json
$previousKey = $env:LAMBDA_API_KEY
try {
    # The loader remains outside this public repository. Its value is never an argument.
    if ($Arguments -contains '--execute' -or $Arguments -contains 'inspect') {
        . $settings.key_loader
    }
    & python (Join-Path $PSScriptRoot 'cloud.py') --config $Config @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Lambda command failed (exit $LASTEXITCODE)." }
}
finally {
    $env:LAMBDA_API_KEY = $previousKey
    $previousKey = $null
}
