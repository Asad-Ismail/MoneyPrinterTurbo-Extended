param(
    [switch]$SkipInstall,
    [switch]$ForceCpu,
    [switch]$OpenBrowser
)

$ErrorActionPreference = "Stop"

function Write-Step {
    param([string]$Message)
    Write-Host ""
    Write-Host "== $Message ==" -ForegroundColor Cyan
}

function Resolve-ProjectRoot {
    return [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
}

function Test-PythonExecutable {
    param([string]$Path)
    if (-not $Path) {
        return $false
    }
    return (Test-Path $Path)
}

function Find-HostPython {
    $candidates = New-Object System.Collections.Generic.List[string]

    foreach ($version in @("Python312", "Python311", "Python310", "Python313", "Python314")) {
        $candidate = Join-Path $env:LOCALAPPDATA "Programs\Python\$version\python.exe"
        if ((Test-Path $candidate) -and -not $candidates.Contains($candidate)) {
            $candidates.Add($candidate)
        }
    }

    try {
        $whereResults = & where.exe python 2>$null
        foreach ($line in $whereResults) {
            $candidate = $line.Trim()
            if ($candidate -and -not $candidate.Contains("WindowsApps") -and -not $candidates.Contains($candidate)) {
                $candidates.Add($candidate)
            }
        }
    }
    catch {
    }

    $commands = @((Get-Command py -ErrorAction SilentlyContinue)) | Where-Object { $_ }

    foreach ($command in $commands) {
        if ($command.Source -and -not $candidates.Contains($command.Source)) {
            $candidates.Add($command.Source)
        }
    }

    $localPrograms = Join-Path $env:LOCALAPPDATA "Programs\Python"
    if ((Test-Path $localPrograms) -and (Test-Path $localPrograms -PathType Container)) {
        Get-ChildItem -Path $localPrograms -Directory -Filter "Python*" -ErrorAction SilentlyContinue |
            Sort-Object Name -Descending |
            ForEach-Object {
                $pythonExe = Join-Path $_.FullName "python.exe"
                if (-not $candidates.Contains($pythonExe)) {
                    $candidates.Add($pythonExe)
                }
            }
    }

    foreach ($candidate in $candidates) {
        if (Test-PythonExecutable $candidate) {
            return $candidate
        }
    }

    return $null
}

function Ensure-Section {
    param(
        [System.Collections.Generic.List[string]]$Lines,
        [string]$Section
    )

    $sectionHeader = "[$Section]"
    for ($i = 0; $i -lt $Lines.Count; $i++) {
        if ($Lines[$i].Trim() -eq $sectionHeader) {
            return $i
        }
    }

    if ($Lines.Count -gt 0 -and $Lines[$Lines.Count - 1].Trim() -ne "") {
        $Lines.Add("")
    }
    $Lines.Add($sectionHeader)
    return ($Lines.Count - 1)
}

function Find-SectionEnd {
    param(
        [System.Collections.Generic.List[string]]$Lines,
        [int]$SectionIndex
    )

    for ($i = $SectionIndex + 1; $i -lt $Lines.Count; $i++) {
        $trimmed = $Lines[$i].Trim()
        if ($trimmed.StartsWith("[") -and $trimmed.EndsWith("]")) {
            return $i
        }
    }
    return $Lines.Count
}

function Upsert-KeyValue {
    param(
        [System.Collections.Generic.List[string]]$Lines,
        [string]$Section,
        [string]$Key,
        [string]$Value,
        [bool]$Force = $false,
        [string[]]$LegacyValues = @()
    )

    $sectionIndex = Ensure-Section -Lines $Lines -Section $Section
    $sectionEnd = Find-SectionEnd -Lines $Lines -SectionIndex $sectionIndex
    $prefix = "$Key ="

    for ($i = $sectionIndex + 1; $i -lt $sectionEnd; $i++) {
        $trimmed = $Lines[$i].Trim()
        if ($trimmed.StartsWith($prefix)) {
            $currentValue = $trimmed.Substring($prefix.Length).Trim()
            $shouldReplace = $Force -or [string]::IsNullOrWhiteSpace($currentValue) -or $currentValue -eq '""' -or $LegacyValues -contains $currentValue
            if ($shouldReplace) {
                $Lines[$i] = "$Key = $Value"
            }
            return
        }
    }

    $insertIndex = $sectionEnd
    if ($insertIndex -lt $Lines.Count -and $Lines[$insertIndex - 1].Trim() -ne "") {
        $Lines.Insert($insertIndex, "$Key = $Value")
    }
    else {
        $Lines.Insert($insertIndex, "$Key = $Value")
    }
}

function Ensure-ConfigDefaults {
    param(
        [string]$ConfigPath,
        [bool]$CreatedConfig,
        [bool]$ForceCpu
    )

    $lines = [System.Collections.Generic.List[string]]::new()
    if (Test-Path $ConfigPath) {
        foreach ($line in [string[]](Get-Content $ConfigPath -Encoding UTF8)) {
            $lines.Add($line)
        }
    }

    Upsert-KeyValue -Lines $lines -Section "project" -Key "video_language" -Value '"ja-JP"' -Force $CreatedConfig
    Upsert-KeyValue -Lines $lines -Section "ui" -Key "language" -Value '"ja"' -Force $CreatedConfig
    Upsert-KeyValue -Lines $lines -Section "ui" -Key "tts_server" -Value '"windows-sapi"' -Force $CreatedConfig
    Upsert-KeyValue -Lines $lines -Section "ui" -Key "voice_name" -Value '"ja-JP-NanamiNeural-Female"' -Force $CreatedConfig -LegacyValues @('""', '"en-AU-NatashaNeural-Female"')

    if ($ForceCpu -or $CreatedConfig) {
        Upsert-KeyValue -Lines $lines -Section "performance" -Key "compute_profile" -Value '"cpu-safe"' -Force $true
        Upsert-KeyValue -Lines $lines -Section "performance" -Key "enable_chatterbox" -Value "false" -Force $true
        Upsert-KeyValue -Lines $lines -Section "performance" -Key "allow_voice_clone" -Value "false" -Force $true
        Upsert-KeyValue -Lines $lines -Section "style" -Key "tts_server" -Value '"windows-sapi"' -Force $true
        Upsert-KeyValue -Lines $lines -Section "app" -Key "compute_profile" -Value '"cpu-safe"' -Force $true
        Upsert-KeyValue -Lines $lines -Section "app" -Key "enable_chatterbox" -Value "false" -Force $true
        Upsert-KeyValue -Lines $lines -Section "app" -Key "allow_voice_clone" -Value "false" -Force $true
        Upsert-KeyValue -Lines $lines -Section "whisper" -Key "device" -Value '"CPU"' -Force $true
        Upsert-KeyValue -Lines $lines -Section "whisper" -Key "compute_type" -Value '"int8"' -Force $true
    }

    Set-Content -Path $ConfigPath -Encoding UTF8 -Value $lines
}

$rootDir = Resolve-ProjectRoot
$venvDir = Join-Path $rootDir ".venv"
$venvPython = Join-Path $venvDir "Scripts\python.exe"
$requirementsPath = Join-Path $rootDir "requirements.txt"
$configPath = Join-Path $rootDir "config.toml"
$configExamplePath = Join-Path $rootDir "config.example.toml"

Write-Step "Checking Python"

$hostPython = Find-HostPython
if (-not $hostPython) {
    Write-Host "Python was not found." -ForegroundColor Red
    Write-Host "Install Python 3.11 or newer, then run run.bat again." -ForegroundColor Yellow
    Write-Host "https://www.python.org/downloads/windows/" -ForegroundColor Yellow
    exit 1
}

Write-Host "Using Python: $hostPython"

if (-not (Test-Path $venvPython)) {
    Write-Step "Creating .venv"
    & $hostPython -m venv $venvDir
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to create the virtual environment."
    }
}
else {
    Write-Step "Reusing existing .venv"
}

if (-not (Test-Path $venvPython)) {
    throw "The .venv Python executable was not found."
}

if (-not $SkipInstall) {
    Write-Step "Installing dependencies"
    & $venvPython -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to upgrade pip."
    }

    & $venvPython -m pip install -r $requirementsPath
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to install requirements.txt."
    }
}
else {
    Write-Step "Skipping dependency install"
}

$createdConfig = $false
if (-not (Test-Path $configPath)) {
    Write-Step "Creating config.toml"
    Copy-Item -Path $configExamplePath -Destination $configPath -Force
    $createdConfig = $true
}
else {
    Write-Step "Checking existing config.toml"
}

Ensure-ConfigDefaults -ConfigPath $configPath -CreatedConfig:$createdConfig -ForceCpu:$ForceCpu

Write-Step "Bootstrap complete"
Write-Host "WebUI will start next." -ForegroundColor Green
