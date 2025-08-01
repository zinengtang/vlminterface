python3 dreamerv3/main.py \
    --configs overcooked \
    --task overcooked_aall \
    --logdir ~/logdir/dreamer/overcooked_all\
    --jax.policy_devices=0 --jax.train_devices=0 \
    --agent.use_vlm True