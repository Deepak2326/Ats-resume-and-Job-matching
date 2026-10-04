# ==========================================================================
# Run the full automated test suite (Phase 1 + Phase 2 + Phase 1-2 e2e gate).
#
# Same AppLocker workaround as run_app.ps1: use the allow-listed uv-managed
# interpreter with the project venv's site-packages on PYTHONPATH.
# ==========================================================================
$root = $PSScriptRoot
$env:PYTHONPATH = Join-Path $root '.venv\Lib\site-packages'
$py = 'C:\Users\Deepak\AppData\Roaming\uv\python\cpython-3.11-windows-x86_64-none\python.exe'
Set-Location $root

& $py tests\test_phase1.py
if ($LASTEXITCODE -ne 0) { Write-Host 'PHASE 1 TESTS FAILED'; exit 1 }
& $py tests\test_phase2.py
if ($LASTEXITCODE -ne 0) { Write-Host 'PHASE 2 TESTS FAILED'; exit 1 }
& $py tests\test_e2e_phase12.py
if ($LASTEXITCODE -ne 0) { Write-Host 'E2E TESTS FAILED'; exit 1 }
& $py tests\test_phase3.py
if ($LASTEXITCODE -ne 0) { Write-Host 'PHASE 3 TESTS FAILED'; exit 1 }
& $py tests\test_phase3_ui.py
if ($LASTEXITCODE -ne 0) { Write-Host 'PHASE 3 UI TESTS FAILED'; exit 1 }
& $py tests\test_phase4.py
if ($LASTEXITCODE -ne 0) { Write-Host 'PHASE 4 TESTS FAILED'; exit 1 }
& $py tests\test_phase4_ui.py
if ($LASTEXITCODE -ne 0) { Write-Host 'PHASE 4 UI TESTS FAILED'; exit 1 }
& $py tests\test_auth.py
if ($LASTEXITCODE -ne 0) { Write-Host 'AUTH TESTS FAILED'; exit 1 }
Write-Host "`nALL TEST SUITES PASSED"
