rm -rf /root/MineLand/mineland/sim/server/world
python3 dreamerv3/main.py \
    --configs mineland \
    --task mineland_techtree_1_wooden_sword_with_64_oak_planks \
    --logdir ~/logdir/dreamer/mineland_techtree_1_wooden_sword_with_64_oak_planks \
    --run.envs 2 \
    --jax.policy_devices=0 --jax.train_devices=0 \
    --agent.use_vlm True
    # --jax.debug True --run.debug True