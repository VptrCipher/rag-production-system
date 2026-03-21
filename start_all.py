import os
import subprocess
import sys
import time

from config import get_settings


def start_system():
    settings = get_settings()

    print("🚀 Starting Premium RAG Production System...")

    # 1. Start Arize Phoenix in a separate process if enabled
    phoenix_process = None
    if os.getenv("PHOENIX_ENABLE", "true").lower() == "true":
        print("🔍 Launching Arize Phoenix Trace Viewer...")
        phoenix_process = subprocess.Popen(
            [sys.executable, "scripts/phoenix_setup.py"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        # Give it a few seconds to initialize
        time.sleep(3)
        print("✅ Phoenix Trace Viewer should be active at http://localhost:6006")

    # 2. Start the FastAPI backend
    print(f"🌐 Starting FastAPI server at http://{settings.api_host}:{settings.api_port}")
    try:
        # We use subprocess for uvicorn to handle the reload properly if desired,
        # but here we'll just run it directly for simplicity if NOT using reload.
        import uvicorn

        uvicorn.run(
            "api.main:app",
            host=settings.api_host,
            port=settings.api_port,
            reload=True,
            log_level=settings.log_level.lower(),
        )
    except KeyboardInterrupt:
        print("\nStopping system...")
    finally:
        if phoenix_process:
            phoenix_process.terminate()
            print("Cleanup: Phoenix process terminated.")


if __name__ == "__main__":
    start_system()
