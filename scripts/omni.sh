rm -rf ~/logdir/dreamer/omni_test
python3 dreamerv3/main.py \
    --configs crafter \
    --task crafter_reward \
    --logdir ~/logdir/dreamer/omni_test \
    --jax.policy_devices=0 --jax.train_devices=0