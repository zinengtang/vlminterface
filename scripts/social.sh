python3 dreamerv3/main.py \
    --configs social \
    --task social_harvest \
    --logdir ~/logdir/dreamer/social_harvest \
    --jax.policy_devices=0 --jax.train_devices=0 \
    --agent.use_vlm True