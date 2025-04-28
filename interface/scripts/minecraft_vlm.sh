# rm -rf ~/logdir/dreamer/minecraft_climb_vlm
xvfb-run -a -s '-screen 0 1024x768x24 -ac +extension GLX +render -noreset' "$@" \
python3 dreamerv3/main.py \
    --configs minecraft \
    --task minecraft_climb \
    --agent.use_vlm True \
    --logdir ~/logdir/dreamer/minecraft_climb_vlm \
    --jax.policy_devices=0,1 --jax.train_devices=0,1