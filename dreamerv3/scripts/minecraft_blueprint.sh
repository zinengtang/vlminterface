# rm -rf /root/logdir/dreamer/minecraft_diamond/
xvfb-run -a -s '-screen 0 1024x768x24 -ac +extension GLX +render -noreset' "$@" \
python3 dreamerv3/main.py \
    --configs minecraft \
    --task minecraft_blueprint \
    --logdir ~/logdir/dreamer/minecraft_blueprint_eiffeltower \
    --agent.opt.lr 1e-4 \
    --jax.policy_devices=0 --jax.train_devices=0