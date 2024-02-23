export PYTHONPATH="/home/kylehatch/Desktop/hidql/calvin-sim/external/susie:$PYTHONPATH"
export PYTHONPATH="/home/kylehatch/Desktop/hidql/calvin-sim/external/jaxrl_m:$PYTHONPATH"
export PYTHONPATH="/home/kylehatch/Desktop/hidql/calvin-sim/external/urdfpy:$PYTHONPATH"
export PYTHONPATH="/home/kylehatch/Desktop/hidql/calvin-sim/external/networkx:$PYTHONPATH"

# export DIFFUSION_MODEL_CHECKPOINT="/home/kylehatch/Desktop/hidql/susie-calvin-checkpoints/diffusion_model/jax_model/public_checkpoint/only_checkpoint/params_ema"
# # export DIFFUSION_MODEL_CHECKPOINT="/home/kylehatch/Desktop/hidql/susie-calvin-checkpoints/diffusion_model/pytorch_model/checkpoint-60000"
# # export DIFFUSION_MODEL_CHECKPOINT="/home/kylehatch/Desktop/hidql/susie-calvin-checkpoints/diffusion_model/pytorch_model/checkpoint-10000"
# # export DIFFUSION_MODEL_CHECKPOINT="/home/kylehatch/Desktop/hidql/susie-calvin-checkpoints/diffusion_model/pytorch_model/checkpoint-80000"
# # export DIFFUSION_MODEL_CHECKPOINT="/home/kylehatch/Desktop/hidql/susie-calvin-checkpoints/diffusion_model/pytorch_model/checkpoint-15000"
# # export DIFFUSION_MODEL_CHECKPOINT="/home/kylehatch/Desktop/hidql/susie-calvin-checkpoints/diffusion_model/jax_model/test1_remote128_2023.12.25_01.41.12/15000/params_ema"

# # export DIFFUSION_MODEL_CHECKPOINT="/home/kylehatch/Desktop/hidql/susie-calvin-checkpoints/diffusion_model/jax_model/test1_remote400_2024.01.03_19.36.28/15000/params_ema"
# # export CUDA_VISIBLE_DEVICES=0
# # export DIFFUSION_MODEL_CHECKPOINT="/home/kylehatch/Desktop/hidql/susie-calvin-checkpoints/diffusion_model/jax_model/test1_remote400_2024.01.03_19.36.28/40000/params_ema"
# # export CUDA_VISIBLE_DEVICES=1
# # export DIFFUSION_MODEL_CHECKPOINT="/home/kylehatch/Desktop/hidql/susie-calvin-checkpoints/diffusion_model/jax_model/test1_remote400smth_2024.01.03_22.02.29/40000/params_ema"
# # export CUDA_VISIBLE_DEVICES=2

# # export DIFFUSION_MODEL_CHECKPOINT="/home/kylehatch/Desktop/hidql/susie-calvin-checkpoints/diffusion_model/jax_model/test1_remote400smth_2024.01.03_22.02.29/5000/params_ema"
# export CUDA_VISIBLE_DEVICES=0

# export GC_POLICY_CHECKPOINT="/home/kylehatch/Desktop/hidql/susie-calvin-checkpoints/gc_policy/public_checkpoint/only_checkpoint"
# # export GC_POLICY_CHECKPOINT="/home/kylehatch/Desktop/hidql/susie-calvin-checkpoints/gc_policy/gcbc_diffusion_policy_remote_20240120_074721/checkpoint_638000"
# export NUM_EVAL_SEQUENCES=10

export DIFFUSION_MODEL_CHECKPOINT="/home/kylehatch/Desktop/hidql/susie-calvin-checkpoints/diffusion_model/jax_model/public_checkpoint/only_checkpoint/params_ema"
export GC_POLICY_CHECKPOINT="/home/kylehatch/Desktop/hidql/susie-calvin-checkpoints/gc_policy/public_checkpoint/only_checkpoint"
# export GC_POLICY_CHECKPOINT="/home/kylehatch/Desktop/hidql/susie-calvin-checkpoints/gc_policy/gcbc_20240121_001822/checkpoint_1538000"
# export GC_POLICY_CHECKPOINT="/home/kylehatch/Desktop/hidql/susie-calvin-checkpoints/gc_policy/gciql_20240121_012712/checkpoint_1538000"
export GC_VF_CHECKPOINT="/home/kylehatch/Desktop/hidql/susie-calvin-checkpoints/gc_policy/gciql_20240121_012712/checkpoint_1538000"

export WANDB_API_KEY="65915e3ae3752bc3ddc4b7eef1b066067b9d1cb1"
export WANDB_ENTITY="tri"

# export DIFFUSION_MODEL_CHECKPOINT="/opt/ml/code/susie-calvin-checkpoints/diffusion_model/jax_model/public_checkpoint/only_checkpoint/params_ema"
# # export GC_POLICY_CHECKPOINT="/opt/ml/code/susie-calvin-checkpoints/gc_policy/public_checkpoint/only_checkpoint"
# export GC_POLICY_CHECKPOINT="/opt/ml/code/susie-calvin-checkpoints/gc_policy/gcbc_20240121_001822/checkpoint_1538000"


export CUDA_VISIBLE_DEVICES=0
export NUM_EVAL_SEQUENCES=10

hard_coded_goal_images_path="/home/kylehatch/Desktop/hidql/oracle_goals/jax_model/public_checkpoint/only_checkpoint/public_checkpoint/only_checkpoint/no_vf/no_checkpoint/50_denoising_steps/1_samples/tmpensb/2024.02.13_14.51.30/ep2/real_images.npy"
hard_coded_goal_idxs="19,39,59,78,78,78"


if [[ "$DIFFUSION_MODEL_CHECKPOINT" == *"jax"* ]]; then
    diffusion_model_framework="jax"
elif [[ "$DIFFUSION_MODEL_CHECKPOINT" == *"pytorch"* ]]; then
    diffusion_model_framework="pytorch"
else 
    echo "Unsupported diffusion model framework: $diffusion_model_framework"
    exit 125
fi

echo "diffusion_model_framework: $diffusion_model_framework"

export DEBUG=1

echo $PYTHONPATH

# python3 -u calvin_models/calvin_agent/evaluation/evaluate_policy_subgoal_diffusion.py \
# python3 -u evaluate_policy_subgoal_diffusion_singletask.py \
python3 -u evaluate_policy_subgoal_diffusion.py \
--dataset_path mini_dataset \
--custom_model 1 \
--diffusion_model_checkpoint_path $DIFFUSION_MODEL_CHECKPOINT \
--gc_policy_checkpoint_path $GC_POLICY_CHECKPOINT \
--gc_vf_checkpoint_path $GC_VF_CHECKPOINT \
--diffusion_model_framework $diffusion_model_framework \
--use_temporal_ensembling 1 \
--num_denoising_steps 50 \
--save_to_s3 0 \
--s3_save_uri "s3://kyle-sagemaker-training-outputs/eval-ouputs" \
--num_samples 1 \
# --hard_coded_goal_idxs $hard_coded_goal_idxs \
# --hard_coded_goal_images_path $hard_coded_goal_images_path \


# changes 
# 0.03 --> 0.035 in calvin_env/conf/tasks/new_playtable_tasks.yaml
# NUM_SEQUENCES 1 vs 10 for DEBUG mode
# pass in hard_coded_goal_images_path
# K=20 vs 30 (self.subgoal_max)