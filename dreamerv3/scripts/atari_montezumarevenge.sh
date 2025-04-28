# rm -rf /root/logdir/dreamer/minecraft_diamond/
xvfb-run -a -s '-screen 0 1024x768x24 -ac +extension GLX +render -noreset' "$@" \
python3 dreamerv3/main.py \
    --configs atari \
    --task atari_montezuma_revenge \
    --logdir ~/logdir/dreamer/atari_montezumarevenge \
    --agent.opt.lr 1e-4 \
    --jax.policy_devices=0 --jax.train_devices=0