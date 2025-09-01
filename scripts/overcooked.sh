python3 dreamerv3/main.py \
    --configs overcooked \
    --task overcooked_all \
    --logdir ~/logdir/dreamer/overcooked_oneagent_v1_multienv_multiplayer \
    --run.envs 8 \
    --jax.policy_devices=0 --jax.train_devices=0 \
    --agent.use_vlm True
    # --jax.debug True --run.debug True