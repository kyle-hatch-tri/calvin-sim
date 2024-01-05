

# export DIFFUSION_MODEL_CHECKPOINT="/home/kylehatch/Desktop/hidql/susie-calvin-checkpoints/diffusion_model/jax_model/public_checkpoint"
# export DIFFUSION_MODEL_CHECKPOINT="/home/kylehatch/Desktop/hidql/susie-calvin-checkpoints/diffusion_model/pytorch_model/checkpoint-60000"
# export DIFFUSION_MODEL_CHECKPOINT="/home/kylehatch/Desktop/hidql/susie-calvin-checkpoints/diffusion_model/pytorch_model/checkpoint-10000"
# export DIFFUSION_MODEL_CHECKPOINT="/home/kylehatch/Desktop/hidql/susie-calvin-checkpoints/diffusion_model/pytorch_model/checkpoint-80000"
# export DIFFUSION_MODEL_CHECKPOINT="/home/kylehatch/Desktop/hidql/susie-calvin-checkpoints/diffusion_model/pytorch_model/checkpoint-15000"
# export DIFFUSION_MODEL_CHECKPOINT="/home/kylehatch/Desktop/hidql/susie-calvin-checkpoints/diffusion_model/jax_model/test1_remote128_2023.12.25_01.41.12/15000/params_ema"

export DIFFUSION_MODEL_CHECKPOINT="/home/kylehatch/Desktop/hidql/susie-calvin-checkpoints/diffusion_model/jax_model/test1_remote400_2024.01.03_19.36.28/15000/params_ema"
export CUDA_VISIBLE_DEVICES=0
# export DIFFUSION_MODEL_CHECKPOINT="/home/kylehatch/Desktop/hidql/susie-calvin-checkpoints/diffusion_model/jax_model/test1_remote400_2024.01.03_19.36.28/40000/params_ema"
# export CUDA_VISIBLE_DEVICES=1
# export DIFFUSION_MODEL_CHECKPOINT="/home/kylehatch/Desktop/hidql/susie-calvin-checkpoints/diffusion_model/jax_model/test1_remote400smth_2024.01.03_22.02.29/40000/params_ema"
# export CUDA_VISIBLE_DEVICES=2

# export DIFFUSION_MODEL_CHECKPOINT="/home/kylehatch/Desktop/hidql/susie-calvin-checkpoints/diffusion_model/jax_model/test1_remote400smth_2024.01.03_22.02.29/5000/params_ema"
# export CUDA_VISIBLE_DEVICES=0

export GC_POLICY_CHECKPOINT="/home/kylehatch/Desktop/hidql/susie-calvin-checkpoints/gc_policy"
export NUM_EVAL_SEQUENCES=100


if [[ "$DIFFUSION_MODEL_CHECKPOINT" == *"jax"* ]]; then
    diffusion_model_framework="jax"
elif [[ "$DIFFUSION_MODEL_CHECKPOINT" == *"pytorch"* ]]; then
    diffusion_model_framework="pytorch"
else 
    echo "Unsupported diffusion model framework: $diffusion_model_framework"
    exit 125
fi

echo "diffusion_model_framework: $diffusion_model_framework"


python3 -u calvin_models/calvin_agent/evaluation/evaluate_policy_subgoal_diffusion.py \
--dataset_path mini_dataset \
--custom_model \
--diffusion_model_framework $diffusion_model_framework




# --eval_log_dir  \


# parser.add_argument(
#     "--train_folder", type=str, help="If calvin_agent was used to train, specify path to the log dir."
# )
# parser.add_argument(
#     "--checkpoints",
#     type=str,
#     default=None,
#     help="Comma separated list of epochs for which checkpoints will be loaded",
# )
# parser.add_argument(
#     "--checkpoint",
#     type=str,
#     default=None,
#     help="Path of the checkpoint",
# )
# parser.add_argument(
#     "--last_k_checkpoints",
#     type=int,
#     help="Specify the number of checkpoints you want to evaluate (starting from last). Only used for calvin_agent.",
# )

# # arguments for loading custom model or custom language embeddings
# parser.add_argument(
#     "--custom_model", action="store_true", help="Use this option to evaluate a custom model architecture."
# )

# parser.add_argument("--debug", action="store_true", help="Print debug info and visualize environment.")

# parser.add_argument("--eval_log_dir", default=None, type=str, help="Where to log the evaluation results.")

# parser.add_argument("--device", default=0, type=int, help="CUDA device")