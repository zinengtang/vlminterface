
# FROM ghcr.io/nvidia/driver:7c5f8932-550.144.03-ubuntu24.04
FROM nvidia/cuda:12.5.0-devel-ubuntu22.04
# System
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/San_Francisco
RUN apt-get update && apt-get install -y \
  ffmpeg git vim curl wget unzip tmux software-properties-common grep \
  libglew-dev x11-xserver-utils xvfb \
  libglu1-mesa libxi6 libxcursor1 libxinerama1 \
  libxrandr2 libxxf86vm1 libasound2 libdbus-1-3 \
  xserver-xephyr xserver-xorg libxi-dev libxext-dev \
  openjdk-17-jdk openjdk-8-jdk pciutils build-essential pkg-config \
  libgl1-mesa-dev apt-transport-https gnupg lsb-release \
  && rm -rf /var/lib/apt/lists/*


ENV NVM_DIR=/root/.nvm
RUN curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash \
  && . "$NVM_DIR/nvm.sh" \
  && nvm install 18.18.0 \
  && nvm alias default 18.18.0 \
  && nvm use 18.18.0
# Make node/npm available for all shells (incl. Docker RUN / ENTRYPOINT)
ENV PATH="$NVM_DIR/versions/node/v18.18.0/bin:$PATH"

# Python (DMLab needs <=3.11)
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_ROOT_USER_ACTION=ignore
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update && apt-get install -y python3.11-dev python3.11-venv && apt-get clean
RUN python3.11 -m venv /venv --upgrade-deps
ENV PATH="/venv/bin:$PATH"
RUN pip install -U pip setuptools

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
  && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

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

# Requirements
RUN pip install jax[cuda]==0.5.0
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

RUN pip install socketio
# Source
RUN mkdir -p /app
WORKDIR /app
COPY . .

RUN chown -R 1000:root .

ENV TRANSFORMERS_CACHE=/root/.cache/huggingface/transformers

ENTRYPOINT ["sh", "entrypoint.sh"]