set -e

# ldconfig

echo 'PIP freeze (subset):'
pip freeze | grep nvidia
pip freeze | grep jax

echo GCP instance:
echo "Name:     $(curl -s -H 'Metadata-Flavor: Google' http://metadata.google.internal/computeMetadata/v1/instance/name || echo NA)"
echo "Hostname: $(curl -s -H 'Metadata-Flavor: Google' http://metadata.google.internal/computeMetadata/v1/instance/hostname || echo NA)"
echo "ID:       $(curl -s -H 'Metadata-Flavor: Google' http://metadata.google.internal/computeMetadata/v1/instance/id || echo NA)"
echo "Zone:     $(curl -s -H 'Metadata-Flavor: Google' http://metadata.google.internal/computeMetadata/v1/instance/zone || echo NA)"
echo

echo GPUs:
nvidia-smi --query-gpu=gpu_name,memory.total,driver_version --format=csv || true
echo

export XLA_FLAGS="--xla_gpu_enable_cudnn_fmha=false --xla_gpu_deterministic_ops=true"
export TF_CUDNN_USE_AUTOTUNE=0
export TF_CUDNN_DETERMINISTIC=1
export XLA_FLAGS="--xla_gpu_autotune_level=0"

# xvfb-run -a -s '-screen 0 1024x768x24 -ac +extension GLX +render -noreset' "$@" bash scripts/omni.sh
# xvfb-run -a -s '-screen 0 1024x768x24 -ac +extension GLX +render -noreset' "$@" bash scripts/atari_montezumarevenge.sh
# xvfb-run -a -s '-screen 0 1024x768x24 -ac +extension GLX +render -noreset' "$@" bash scripts/mineland.sh
# xvfb-run -a -s '-screen 0 1024x768x24 -ac +extension GLX +render -noreset' "$@" bash scripts/mineland_test.sh
# xvfb-run -a -s '-screen 0 1024x768x24 -ac +extension GLX +render -noreset' "$@" bash scripts/minecraft_blueprint.sh
# xvfb-run -a -s '-screen 0 1024x768x24 -ac +extension GLX +render -noreset' "$@" bash scripts/app.sh
# xvfb-run -a -s '-screen 0 1024x768x24 -ac +extension GLX +render -noreset' "$@" bash scripts/profile.sh
xvfb-run -a -s '-screen 0 1024x768x24 -ac +extension GLX +render -noreset' "$@" bash scripts/overcooked.sh
# xvfb-run -a -s '-screen 0 1024x768x24 -ac +extension GLX +render -noreset' "$@" bash scripts/crafter_vlm.sh
# xvfb-run -a -s '-screen 0 1024x768x24 -ac +extension GLX +render -noreset' "$@" bash scripts/meltingpot.sh
# bash scripts/minecraft.sh
# xvfb-run "$@"
