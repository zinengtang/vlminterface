python3 dreamerv3/main.py \
    --configs virtualhome \
    --task virtualhome_standard \
    --logdir ~/logdir/dreamer/virtualhome_standard\
    --jax.policy_devices=0 --jax.train_devices=0 \
    --agent.use_vlm True