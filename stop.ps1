param([int]$Port = 5000)

$ErrorActionPreference = 'Stop'
$projectDirectory = $PSScriptRoot
$pythonPath = Join-Path $projectDirectory '.venv\Scripts\python.exe'
$identityCode = 'import sys; from pathlib import Path; from rag.runtime import project_identity; print(project_identity(Path(sys.argv[1])))'
Push-Location -LiteralPath $projectDirectory
try {
    $expectedIdentity = & $pythonPath -X utf8 -c $identityCode $projectDirectory
    if ($LASTEXITCODE -ne 0) { throw 'Cannot read project identity.' }
    $listeners = @(Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue)
    if ($listeners.Count -gt 0) {
        $status = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/api/status" -TimeoutSec 5
        if ($status.service -ne 'lingxing-rag' -or $status.project_id -ne $expectedIdentity.Trim()) {
            throw 'This port belongs to another application; nothing was stopped.'
        }
        foreach ($processId in ($listeners.OwningProcess | Select-Object -Unique)) {
            $webProcess = Get-CimInstance Win32_Process -Filter "ProcessId = $processId"
            if ($webProcess.CommandLine -notmatch 'run\.py') {
                throw 'Unexpected server process; nothing was stopped.'
            }
            Stop-Process -Id $processId
            Wait-Process -Id $processId -Timeout 10 -ErrorAction SilentlyContinue
        }
        Write-Output 'Lingxing RAG web server stopped.'
    }
    & $pythonPath -X utf8 local_mysql.py stop
    if ($LASTEXITCODE -ne 0) { throw 'Database shutdown failed.' }
} finally {
    Pop-Location
}
