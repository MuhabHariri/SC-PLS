param(
    [ValidateSet('cu118','cu128','cpu')][string]$Build = 'cu128',
    [ValidateSet('3.11','3.12')][string]$PythonVersion = '3.11'
)
$ErrorActionPreference = 'Stop'
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
$ProjectPython = Join-Path $RepoRoot '.venv\Scripts\python.exe'
if (-not (Test-Path $ProjectPython)) {
    & py "-$PythonVersion" -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Install the requested Python version, or pass -PythonVersion 3.12.' }
}
& $ProjectPython -m pip install -r "requirements-$Build.txt"
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
& $ProjectPython -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Dependency check failed.' }
& $ProjectPython run.py verify --load-model --device cpu
if ($LASTEXITCODE -ne 0) { throw 'Asset or checkpoint verification failed.' }
& $ProjectPython -m unittest discover -s tests -v
if ($LASTEXITCODE -ne 0) { throw 'Tests failed.' }
Write-Host 'Setup completed. Run preflight before the full evaluation.'
