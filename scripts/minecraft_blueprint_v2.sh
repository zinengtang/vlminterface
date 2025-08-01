sudo rm -rf ~/logdir/docker/dreamer/minecraft_blueprint_single/
xvfb-run -a -s '-screen 0 1024x768x24 -ac +extension GLX +render -noreset' "$@" \
python3 dreamerv3/main.py \
    --configs minecraft \
    --task minecraft_blueprint \
    --logdir ~/logdir/dreamer/minecraft_blueprint_v1 \
    --agent.opt.lr 4e-5 \
    --agent.use_vlm True \
    --run.envs 32 \
    --jax.policy_devices=0 --jax.train_devices=0 \
    --agent.enc.simple.depth 128 \
    --agent.enc.simple.mults 2,3,4,6,8 \
    --agent.enc.simple.kernel 3 \
    --agent.enc.simple.strided True \
    --jax.debug False --run.debug False