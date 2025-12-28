param(
  [string]$Owner = "ODawgAce",
  [string]$Repo  = "IDS",
  [string]$Tag   = "latest",      # np. "v1.0.0" albo "latest"
  [string]$OutDir = "artifacts"
)

$ErrorActionPreference = "Stop"

function Ensure-Dir([string]$p) {
  if (!(Test-Path $p)) { New-Item -ItemType Directory -Path $p | Out-Null }
}

function Download-File([string]$url, [string]$dest) {
  Write-Host "Downloading: $url"
  Write-Host " -> $dest"
  try {
    Invoke-WebRequest -Uri $url -OutFile $dest -UseBasicParsing
  } catch {
    Write-Host "Invoke-WebRequest failed, trying curl..."
    curl.exe -L $url -o $dest
  }
}

function Get-ReleaseJson() {
  $api =
    if ($Tag -eq "latest") {
      "https://api.github.com/repos/$Owner/$Repo/releases/latest"
    } else {
      "https://api.github.com/repos/$Owner/$Repo/releases/tags/$Tag"
    }

  $headers = @{
    "User-Agent" = "download_models.ps1"
    "Accept"     = "application/vnd.github+json"
  }

  return Invoke-RestMethod -Uri $api -Headers $headers
}

Ensure-Dir $OutDir

$need = @(
  "preproc.joblib",
  "rf.joblib",
  "lstm.keras",
  "lstm.keras.best.keras",
  "meta.joblib"
)

Write-Host "[*] Fetching release info: $Owner/$Repo tag=$Tag"
$rel = Get-ReleaseJson

if (-not $rel.assets -or $rel.assets.Count -eq 0) {
  throw "No assets found in the release. Add model files to GitHub Release Assets first."
}

$assetsByName = @{}
foreach ($a in $rel.assets) { $assetsByName[$a.name] = $a }

$missing = @()
foreach ($f in $need) {
  if (-not $assetsByName.ContainsKey($f)) { $missing += $f }
}

if ($missing.Count -gt 0) {
  Write-Host ""
  Write-Host "Available assets:"
  $rel.assets | Select-Object name, size | Format-Table -AutoSize | Out-String | Write-Host
  throw "Missing assets in release: $($missing -join ', ')"
}

foreach ($f in $need) {
  $url = $assetsByName[$f].browser_download_url
  $dest = Join-Path $OutDir $f
  Download-File $url $dest
}

Write-Host ""
Write-Host "[OK] Models downloaded to: $OutDir"
Get-ChildItem $OutDir | Select-Object Name, Length | Format-Table -AutoSize
