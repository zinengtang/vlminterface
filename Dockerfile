
# FROM ghcr.io/nvidia/driver:7c5f8932-550.144.03-ubuntu24.04
FROM nvidia/cuda:12.5.0-devel-ubuntu22.04
# System
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/San_Francisco
RUN apt-get update && apt-get install -y \
  ffmpeg git vim curl software-properties-common grep \
  libglew-dev x11-xserver-utils xvfb wget unzip \
  libglu1-mesa libxi6 libxcursor1 libxinerama1 \
  libxrandr2 libxxf86vm1 libasound2 libdbus-1-3 \
  && apt-get clean

# Python (DMLab needs <=3.11)
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_ROOT_USER_ACTION=ignore
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update && apt-get install -y python3.11-dev python3.11-venv && apt-get clean
RUN python3.11 -m venv /venv --upgrade-deps
ENV PATH="/venv/bin:$PATH"
RUN pip install -U pip setuptools

# Envs
RUN wget -O - https://gist.githubusercontent.com/danijar/ca6ab917188d2e081a8253b3ca5c36d3/raw/install-dmlab.sh | sh
RUN pip install ale_py==0.9.0 autorom[accept-rom-license]==0.6.1
RUN pip install procgen_mirror
RUN pip install crafter
RUN pip install dm_control
RUN pip install memory_maze
ENV MUJOCO_GL=egl
RUN apt-get update && apt-get install -y openjdk-8-jdk && apt-get clean
RUN pip install https://github.com/danijar/minerl/releases/download/v0.4.4-patched/minerl_mirror-0.4.4-cp311-cp311-linux_x86_64.whl
RUN chown -R 1000:root /venv/lib/python3.11/site-packages/minerl
RUN pip install dm-meltingpot
# VirtualHome and dependencies
# RUN pip install opencv-python
# Download VirtualHome Unity Simulator (Linux x86-64 version)
# Create directory structure first
# RUN mkdir -p /app/simulation/unity_simulator

# Requirements
RUN pip install jax[cuda]==0.5.0
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Source
RUN mkdir -p /app
WORKDIR /app
COPY . .

# Download VirtualHome executable after copying source
# This ensures the simulation directory exists in the right place
# RUN wget -O /tmp/virtualhome_unity.zip "http://virtual-home.org//release/simulator/v2.0/v2.3.0/linux_exec.zip" && \
#     unzip /tmp/virtualhome_unity.zip -d /app/simulation/unity_simulator/ && \
#     rm /tmp/virtualhome_unity.zip && \
#     find /app/simulation/unity_simulator -name "*.x86_64" -exec chmod +x {} \;

# RUN pip install "git+https://github.com/zinengtang/virtualhome.git"

# Set environment variable for VirtualHome executable
# Adjust the path based on the actual extracted structure
# ENV VHOME_EXECUTABLE=/app/simulation/unity_simulator/v2.3.0_Linux/VirtualHome.x86_64

RUN chown -R 1000:root .
# RUN chown -R 1000:root /app/simulation/unity_simulator

ENV TRANSFORMERS_CACHE=/root/.cache/huggingface/transformers

ENTRYPOINT ["sh", "entrypoint.sh"]