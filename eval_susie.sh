

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
# export GC_POLICY_CHECKPOINT="/home/kylehatch/Desktop/hidql/susie-calvin-checkpoints/gc_policy/public_checkpoint/only_checkpoint"
# export GC_POLICY_CHECKPOINT="/home/kylehatch/Desktop/hidql/susie-calvin-checkpoints/gc_policy/gcbc_20240121_001822/checkpoint_1538000"
export GC_POLICY_CHECKPOINT="/home/kylehatch/Desktop/hidql/susie-calvin-checkpoints/gc_policy/gciql_20240121_012712/checkpoint_1538000"

export WANDB_API_KEY="65915e3ae3752bc3ddc4b7eef1b066067b9d1cb1"
export WANDB_ENTITY="tri"

# export DIFFUSION_MODEL_CHECKPOINT="/opt/ml/code/susie-calvin-checkpoints/diffusion_model/jax_model/public_checkpoint/only_checkpoint/params_ema"
# # export GC_POLICY_CHECKPOINT="/opt/ml/code/susie-calvin-checkpoints/gc_policy/public_checkpoint/only_checkpoint"
# export GC_POLICY_CHECKPOINT="/opt/ml/code/susie-calvin-checkpoints/gc_policy/gcbc_20240121_001822/checkpoint_1538000"


# export CUDA_VISIBLE_DEVICES=0
export NUM_EVAL_SEQUENCES=1



if [[ "$DIFFUSION_MODEL_CHECKPOINT" == *"jax"* ]]; then
    diffusion_model_framework="jax"
elif [[ "$DIFFUSION_MODEL_CHECKPOINT" == *"pytorch"* ]]; then
    diffusion_model_framework="pytorch"
else 
    echo "Unsupported diffusion model framework: $diffusion_model_framework"
    exit 125
fi

echo "diffusion_model_framework: $diffusion_model_framework"


if [[ "$GC_POLICY_CHECKPOINT" == *"gcbc_diffusion"* ]]; then
    agent_type="gc_ddpm_bc"
elif [[ "$GC_POLICY_CHECKPOINT" == *"public"* ]]; then
    agent_type="gc_ddpm_bc"
elif [[ "$GC_POLICY_CHECKPOINT" == *"gcbc"* ]]; then
    agent_type="gc_bc"
elif [[ "$GC_POLICY_CHECKPOINT" == *"gciql2"* ]]; then
    agent_type="gc_iql2"
elif [[ "$GC_POLICY_CHECKPOINT" == *"gciql3"* ]]; then
    agent_type="gc_iql3"
elif [[ "$GC_POLICY_CHECKPOINT" == *"gciql4"* ]]; then
    agent_type="gc_iql4"
elif [[ "$GC_POLICY_CHECKPOINT" == *"gciql5"* ]]; then
    agent_type="gc_iql5"
elif [[ "$GC_POLICY_CHECKPOINT" == *"gciql"* ]]; then
    agent_type="gc_iql"
else 
    echo "Unsupported agent_type in $GC_POLICY_CHECKPOINT"
    exit 125
fi

echo "agent_type: $agent_type"


export DEBUG=1

# python3 -u calvin_models/calvin_agent/evaluation/evaluate_policy_subgoal_diffusion.py \
python3 -u evaluate_policy_subgoal_diffusion.py \
--dataset_path mini_dataset \
--custom_model 1 \
--diffusion_model_checkpoint_path $DIFFUSION_MODEL_CHECKPOINT \
--gc_policy_checkpoint_path $GC_POLICY_CHECKPOINT \
--diffusion_model_framework $diffusion_model_framework \
--agent_type $agent_type \
--use_temporal_ensembling 1 \
--num_denoising_steps 50 \
--save_to_s3 0 \
--s3_save_uri "s3://kyle-sagemaker-training-outputs/eval-ouputs"