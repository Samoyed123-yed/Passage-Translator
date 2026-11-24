# Windows PowerShell 5.1 compatible. Run from an independent clone.
[CmdletBinding()]
param(
    [switch]$NoShortcut,
    [string]$ShortcutDirectory
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$root = $PSScriptRoot
if (-not [Environment]::Is64BitOperatingSystem) { throw 'Windows x64 is required.' }
Set-Location -LiteralPath $root
foreach ($relative in @('.venv', 'models', 'runtime', 'fonts')) {
    $path = Join-Path $root $relative
    if (Test-Path -LiteralPath $path) {
        $item = Get-Item -LiteralPath $path -Force
        if ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) {
            throw "Refusing linked installation directory: $relative"
        }
    }
}
$python = $null
foreach ($candidate in @('python', 'py')) {
    if (-not (Get-Command $candidate -ErrorAction SilentlyContinue)) { continue }
    $arguments = @('-c', 'import sys,struct,json; assert sys.version_info[:2]==(3,13) and struct.calcsize(chr(80))==8; print(json.dumps(sys.executable,ensure_ascii=True))')
    if ($candidate -eq 'py') { $arguments = @('-3.13') + $arguments }
    try { $result = & $candidate @arguments 2>$null } catch { continue }
    if ($LASTEXITCODE -eq 0) { $python = [string](($result | Select-Object -Last 1) | ConvertFrom-Json); break }
}
if (-not $python) {
    throw 'Install Python 3.13 x64 first: winget install -e --id Python.Python.3.13 ; then reopen PowerShell and rerun install.ps1.'
}
$venv = Join-Path $root '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $venv)) {
    & $python -m venv (Join-Path $root '.venv')
    if ($LASTEXITCODE -ne 0) { throw 'Creating .venv failed.' }
}
& $venv -c 'import sys,struct; assert sys.version_info[:2]==(3,13) and struct.calcsize(chr(80))==8'
if ($LASTEXITCODE -ne 0) { throw 'Existing .venv must use Python 3.13 x64. Use a fresh clone.' }
& $venv -m pip install --disable-pip-version-check -r (Join-Path $root 'requirements-lock.txt')
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed; models have not been installed.' }
& $venv -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Dependency consistency check failed.' }
& $venv (Join-Path $root 'setup_models.py')
if ($LASTEXITCODE -ne 0) { throw 'Asset download or checksum verification failed.' }
& $venv -c 'import app,engine,sakura; assert not engine.model_status(), engine.model_status(); assert sakura.available()'
if ($LASTEXITCODE -ne 0) { throw 'Application import or model check failed.' }
if (-not $NoShortcut) {
    if (-not $ShortcutDirectory) { $ShortcutDirectory = [Environment]::GetFolderPath('Desktop') }
    if (-not (Test-Path -LiteralPath $ShortcutDirectory -PathType Container)) { throw 'ShortcutDirectory must exist.' }
    $link = Join-Path $ShortcutDirectory 'Passage.lnk'
    $shell = New-Object -ComObject WScript.Shell
    if (Test-Path -LiteralPath $link) {
        $old = $shell.CreateShortcut($link)
        if ($old.TargetPath -ne (Join-Path $root 'Launch.vbs')) {
            throw 'An existing Passage shortcut points elsewhere. Rerun with -NoShortcut or -ShortcutDirectory; it was not overwritten.'
        }
    }
    $shortcut = $shell.CreateShortcut($link)
    $shortcut.TargetPath = Join-Path $root 'Launch.vbs'
    $shortcut.WorkingDirectory = $root
    $shortcut.IconLocation = (Join-Path $root 'passage.ico') + ',0'
    $shortcut.Description = 'Passage - offline comic translation'
    $shortcut.Save()
}
Write-Host 'Installation complete. Double-click Launch.vbs or Start-Passage.cmd.'
