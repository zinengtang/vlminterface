python3 dreamerv3/main.py \
    --configs mindcraft \
    --task mindcraft_cook \
    --logdir ~/logdir/dreamer/mindcraft_cook \
    --run.envs 2 \
    --jax.policy_devices=0 --jax.train_devices=0 \
    --agent.use_vlm True
    # --jax.debug True --run.debug True