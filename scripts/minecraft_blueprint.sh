sudo rm -rf ~/logdir/docker/dreamer/minecraft_blueprint_clip_global_local_224/
xvfb-run -a -s '-screen 0 1024x768x24 -ac +extension GLX +render -noreset' "$@" \
python3 dreamerv3/main.py \
    --configs minecraft \
    --task minecraft_blueprint \
    --logdir ~/logdir/dreamer/minecraft_blueprint_clip_global_local_224 \
    --agent.opt.lr 4e-5 \
    --agent.use_vlm True \
    --run.envs 16 \
    --jax.policy_devices=0 --jax.train_devices=0 \
    --agent.enc.simple.depth 128 \
    --agent.enc.simple.mults 2,3,4,6,8 \
    --agent.enc.simple.kernel 3 \
    --agent.enc.simple.strided True
    # --jax.debug True --run.debug True