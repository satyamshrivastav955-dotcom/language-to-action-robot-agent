# Reproduce the benchmark and the replays without installing anything locally.
#
#   docker build -t robot-agent .
#   docker run --rm -v "$PWD/results:/app/results" robot-agent bench.run --episodes 10 --workers 4
#   docker run --rm -v "$PWD/results:/app/results" -v "$PWD/web:/app/web" robot-agent bench.report results/ablation.json --md results/ablation.md --json web/data/results.json
#
# Rendering is CPU-only (OSMesa), so no GPU or display is needed.
FROM python:3.12-slim

ENV MUJOCO_GL=osmesa \
    PYOPENGL_PLATFORM=osmesa \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends libosmesa6 libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY agent ./agent
COPY bench ./bench
COPY configs ./configs
COPY dashboard ./dashboard
COPY env ./env
COPY perception ./perception
COPY tools ./tools
COPY adapters ./adapters
COPY run_agent.py app.py ./
COPY tests ./tests
RUN pip install -e ".[dashboard,dev]"

# `docker run robot-agent <module> <args>` runs `python -m <module> <args>`.
ENTRYPOINT ["python", "-m"]
CMD ["pytest", "-q", "-m", "not physics"]
