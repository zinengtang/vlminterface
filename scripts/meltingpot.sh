python3 dreamerv3/main.py \
    --configs meltingpot \
    --task meltingpot_prisoners_dilemma_in_the_matrix__arena \
    --logdir ~/logdir/dreamer/meltingpot_prisoners_dilemma_in_the_matrix__arena \
    --jax.policy_devices=0 --jax.train_devices=0 \
    --run.envs 8 \
    --agent.use_vlm True