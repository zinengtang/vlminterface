# rm -rf /root/logdir/dreamer/minecraft_climb/
xvfb-run -a -s '-screen 0 1024x768x24 -ac +extension GLX +render -noreset' "$@" \
python3 dreamerv3/main.py \
    --configs minecraft \
    --task minecraft_climb \
    --logdir ~/logdir/dreamer/minecraft_climb \
    --agent.opt.lr 1e-4 \
    --jax.policy_devices=0 --jax.train_devices=0