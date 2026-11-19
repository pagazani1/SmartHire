Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PWD'; .\.venv\Scripts\Activate.ps1; uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000"
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PWD'; npm --prefix frontend run dev"
Start-Sleep -Seconds 3
Start-Process "http://localhost:5173"
