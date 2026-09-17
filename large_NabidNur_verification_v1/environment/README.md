# Environment

The solution is pure Python standard library: no third-party packages, no GPU, no CUDA, no network.

- Preferred: Docker, from the package root: `docker build -f environment/Dockerfile -t nabidnur/cuhkx-large-vqa:v1 .`
  The image is based on `python:3.11.9-slim-bookworm` and contains the package; its `WORKDIR` is the package root.
- Without Docker: any Linux x86_64 machine with CPython 3.8 or newer (`python3` on PATH) and `bash`.
  Setup commands from a clean machine: none beyond installing Python 3 (`apt-get install -y python3`).

`requirements.txt` is intentionally empty (comments only).
