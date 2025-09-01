python3 dreamerv3/main.py \
    --configs crafter \
    --task crafter_reward \
    --logdir ~/logdir/dreamer/crafter_v1 \
    --agent.use_vlm True \
    --jax.policy_devices=0 --jax.train_devices=0