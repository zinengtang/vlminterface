xvfb-run -a -s '-screen 0 1024x768x24 -ac +extension GLX +render -noreset' "$@" \
python3 dreamerv3/main.py \
    --configs minecraft \
    --task minecraft_diamond \
    --logdir ~/logdir/dreamer/minecraft_diamond_v1 \
    --agent.opt.lr 4e-5 \
    --run.envs 8 \
    --jax.policy_devices=0 --jax.train_devices=0 \
    --agent.use_vlm True