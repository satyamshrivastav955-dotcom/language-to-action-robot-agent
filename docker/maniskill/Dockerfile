# ManiSkill3 benchmark image: CPU simulation + software Vulkan rendering (lavapipe),
# so it needs no GPU and runs the same on Windows (Docker Desktop/WSL2), Linux and CI.
FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3-pip python3-dev build-essential \
        libvulkan1 mesa-vulkan-drivers vulkan-tools libgl1 libglib2.0-0 libxext6 libsm6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*
RUN pip3 install torch --index-url https://download.pytorch.org/whl/cpu
RUN pip3 install "mani_skill>=3.0.0" mplib==0.1.1 opencv-python-headless
# mplib 0.1.1 segfaults under NumPy 2; keep NumPy 1.x.
RUN pip3 install "numpy<2"
WORKDIR /work
COPY maniskill_bench ./maniskill_bench
ENV XDG_RUNTIME_DIR=/tmp
ENV VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json
ENTRYPOINT ["python3", "-m"]
CMD ["maniskill_bench.run", "--help"]
