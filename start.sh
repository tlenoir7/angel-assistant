#!/bin/sh
exec uvicorn web_app_fastapi:app --host 0.0.0.0 --port $PORT
