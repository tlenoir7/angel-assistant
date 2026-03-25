import os
import uvicorn

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("web_app_fastapi:asgi_app", host="0.0.0.0", port=port)
