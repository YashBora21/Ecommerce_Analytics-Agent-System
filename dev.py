import os
import subprocess
import sys
import time


def main() -> None:
    backend_port = os.getenv("BACKEND_PORT", "8000")
    frontend_port = os.getenv("FRONTEND_PORT", "5500")
    commands = [
        [
            sys.executable,
            "-m",
            "uvicorn",
            "backend.app:app",
            "--reload",
            "--port",
            backend_port,
        ],
        [
            sys.executable,
            "-m",
            "http.server",
            frontend_port,
            "--directory",
            "frontend",
        ],
    ]
    processes = [subprocess.Popen(command) for command in commands]
    print(
        f"Backend: http://127.0.0.1:{backend_port}\n"
        f"Frontend: http://127.0.0.1:{frontend_port}\n"
        "Press Ctrl+C to stop both."
    )

    try:
        while all(process.poll() is None for process in processes):
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            process.wait()

    failed = next((process.returncode for process in processes if process.returncode), 0)
    raise SystemExit(failed)


if __name__ == "__main__":
    main()
