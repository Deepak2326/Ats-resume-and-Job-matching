# ==========================================================================
# Launch the ATS Resume Analyzer & Job Recommendation Platform.
#
# NOTE (this machine): the Application Control policy (AppLocker) only allows
# the uv-managed base interpreter to execute, so we run it directly and point
# PYTHONPATH at the project venv's site-packages (.venv\Lib\site-packages).
# On a normal machine, plain `streamlit run app.py` works.
# ==========================================================================
$root = $PSScriptRoot
$env:PYTHONPATH = Join-Path $root '.venv\Lib\site-packages'
$py = 'C:\Users\Deepak\AppData\Roaming\uv\python\cpython-3.11-windows-x86_64-none\python.exe'
# Load all-MiniLM-L6-v2 from the local HF cache without network revalidation
# (this machine's proxy triggers slow SSL-retry storms on every HEAD request).
# The model is already cached; if the cache is ever wiped, matcher.py
# automatically falls back to the TF-IDF backend.
$env:HF_HUB_OFFLINE = '1'
Set-Location $root
& $py -m streamlit run app.py
