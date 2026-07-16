param(
    [string]$Source = "X:\OpenAI\anomaly-codex-main\projects\anthology-ai-helper\knowledge",
    [string]$Destination = "X:\OpenAI\anomaly-codex-main\projects\anthology-discord-bot\knowledge"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath $Source)) {
    throw "Source knowledge folder not found: $Source"
}

New-Item -ItemType Directory -Force -Path $Destination | Out-Null

Get-ChildItem -LiteralPath $Source -Filter "*.md" -File | ForEach-Object {
    Copy-Item -LiteralPath $_.FullName -Destination (Join-Path $Destination $_.Name) -Force
}

Write-Host "Knowledge synced:"
Write-Host "  from: $Source"
Write-Host "  to:   $Destination"

