# CUHK-X Large Model Track - team "Nabid Nur".
# The solution uses only the Python standard library (no third-party packages, no GPU, no network).
FROM python:3.11.9-slim-bookworm

ENV PYTHONHASHSEED=0 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /workspace/large_NabidNur_verification_v1
COPY . /workspace/large_NabidNur_verification_v1

# Build from the package root:
#   docker build -f environment/Dockerfile -t nabidnur/cuhkx-large-vqa:v1 .
# Run (GPU flags are accepted but not needed):
#   docker run --rm --network none -v <data_dir>:/data:ro -v <output_dir>:/output \
#       nabidnur/cuhkx-large-vqa:v1 bash inference.sh /data /output/submission.csv
CMD ["bash", "inference.sh", "/data", "/output/submission.csv"]
