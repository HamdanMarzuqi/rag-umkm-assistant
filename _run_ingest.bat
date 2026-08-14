@echo off
cd /d "E:\Dokumenku_2025\UNTUK PORTOFOLIO WEBDEV\PROJEKKU\rag-umkm-assistant"
set PYTHONPATH=
".venv\Scripts\python.exe" -m src.ingest > ingest_log.txt 2>&1
echo RC=%errorlevel% >> ingest_log.txt
