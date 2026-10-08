import os
import subprocess
import sys


def main() -> None:
    port = os.getenv("PORT", "8000")
    print(f"Application: http://127.0.0.1:{port}")
    try:
        result = subprocess.run([
            sys.executable,
            "-m",
            "uvicorn",
            "backend.app:app",
            "--port",
            port,
            "--no-access-log",
        ])
    except KeyboardInterrupt:
        return
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()


