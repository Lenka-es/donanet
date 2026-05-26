FROM ultralytics/ultralytics:latest

WORKDIR /app

# Copy project files
COPY pyproject.toml .
COPY donanet.py .

# Install additional CLI dependencies
# (ultralytics, torch, torchvision already provided by base image)
RUN pip install --no-cache-dir \
    "typer[all]>=0.12.0" \
    "rich>=14.0.0" \
    "pillow>=10.0.0" \
    "pyyaml>=6.0" \
    "scikit-learn>=1.4.0" \
    "tqdm>=4.66.0"

# Volume mount points
RUN mkdir -p /app/dataset /app/weights /app/runs

ENTRYPOINT ["python", "donanet.py"]
