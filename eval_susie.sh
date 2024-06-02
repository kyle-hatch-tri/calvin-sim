export PYTHONPATH="/home/kylehatch/Desktop/hidql/calvin-sim/external/susie:$PYTHONPATH"
export PYTHONPATH="/home/kylehatch/Desktop/hidql/calvin-sim/external/jaxrl_m:$PYTHONPATH"
export PYTHONPATH="/home/kylehatch/Desktop/hidql/calvin-sim/external/urdfpy:$PYTHONPATH"
export PYTHONPATH="/home/kylehatch/Desktop/hidql/calvin-sim/external/networkx:$PYTHONPATH"
export WANDB_API_KEY="65915e3ae3752bc3ddc4b7eef1b066067b9d1cb1"
export WANDB_ENTITY=tri 

export CUDA_VISIBLE_DEVICES=0


S3_SAVE_URI=s3://kyle-sagemaker-training-outputs/eval-outputs

export NUM_DENOISING_STEPS=50




current_timestamp=$(date +"%Y_%m_%d_%H_%M_%S")
mkdir -p results/stdouts/$current_timestamp

n_seeds=1

# # CALVIN
# export NUM_EVAL_SEQUENCES=100
# export EP_LEN=360
# export SUBGOAL_MAX=20
# export NUM_SAMPLES=1
# export DIFFUSION_MODEL_CHECKPOINT=/home/kylehatch/Desktop/hidql/susie-calvin-checkpoints/susie_test/test1_400smthlibs1_2024.02.22_19.59.48/40000/params_ema
# export GC_POLICY_CHECKPOINT=/home/kylehatch/Desktop/hidql/susie-calvin-checkpoints/susie_low_level/calvin/gcdiffusion/default/seed_2/20240227_191322/checkpoint_50000
# agent_config_string="calvin_gcdiffusion_noactnorm-sagemaker"

# # liberosplit1
# export NUM_EVAL_SEQUENCES=100
# export EP_LEN=360
# export NUM_SAMPLES=1
# export DIFFUSION_MODEL_CHECKPOINT=/home/kylehatch/Desktop/hidql/susie-calvin-checkpoints/susie_test/test1_400smthlibs1_2024.02.22_19.59.48/40000/params_ema
# export GC_POLICY_CHECKPOINT=/home/kylehatch/Desktop/hidql/susie-calvin-checkpoints/susie_low_level/liberosplit1/gcdiffusion/default/seed_0/20240228_001204/checkpoint_500000
# agent_config_string="liberosplit1_gcdiffusion_noactnorm-sagemaker"


# # liberosplit2
# export NUM_EVAL_SEQUENCES=100
# export EP_LEN=360
# export SUBGOAL_MAX=30
# export NUM_SAMPLES=1
# export DIFFUSION_MODEL_CHECKPOINT=/home/kylehatch/Desktop/hidql/susie-calvin-checkpoints/susie_test/test1_400smthlibs2_2024.02.23_14.20.24/40000/params_ema
# export GC_POLICY_CHECKPOINT=/home/kylehatch/Desktop/hidql/susie-calvin-checkpoints/susie_low_level/liberosplit2/gcdiffusion/default/seed_0/20240301_015613/checkpoint_500000
# agent_config_string="liberosplit2_gcdiffusion_noactnorm-sagemaker"

# # liberosplit2 single task 
# export NUM_EVAL_SEQUENCES=50
# export EP_LEN=500
# export SUBGOAL_MAX=30
# export NUM_SAMPLES=1
# export DIFFUSION_MODEL_CHECKPOINT=/home/kylehatch/Desktop/hidql/susie-calvin-checkpoints/susie_test/test1_400smthlibs2_2024.02.23_14.20.24/40000/params_ema
# # export GC_POLICY_CHECKPOINT=/home/kylehatch/Desktop/hidql/susie-calvin-checkpoints/susie_low_level/liberosplit2/gcdiffusion/default/seed_0/20240301_015613/checkpoint_500000
# export GC_POLICY_CHECKPOINT=/home/kylehatch/Desktop/hidql/susie-calvin-checkpoints/susie_low_level/liberosplit2/gcdiffusion/auggoaldiff/seed_0/20240228_024329/checkpoint_150000
# agent_config_string="liberosplit2_gcdiffusion_noactnorm-sagemaker"
# SINGLE_TASK=1
# export DEBUG=0

# ORACLE_GOALS_DIR="/home/kylehatch/Desktop/hidql/data/libero_data_processed_split2"
# # ORACLE_GOALS_TYPE="generated"
# ORACLE_GOALS_TYPE="dataset_true"

# # liberosplit3
# export NUM_EVAL_SEQUENCES=100
# export EP_LEN=500
# export SUBGOAL_MAX=30
# export NUM_SAMPLES=1
# export DIFFUSION_MODEL_CHECKPOINT=/home/kylehatch/Desktop/hidql/susie-calvin-checkpoints/susie_test/test1_400smthlib_2024.02.21_06.44.06/40000/params_ema
# export GC_POLICY_CHECKPOINT=/home/kylehatch/Desktop/hidql/susie-calvin-checkpoints/susie_low_level/liberosplit3/gcdiffusion/auggoaldiff/seed_0/20240306_064653/checkpoint_150000
# agent_config_string="liberosplit3_gcdiffusion_noactnorm-sagemaker"

# # liberosplit3val90
# export NUM_EVAL_SEQUENCES=100
# export EP_LEN=360
# export SUBGOAL_MAX=30
# export NUM_SAMPLES=1
# export DIFFUSION_MODEL_CHECKPOINT=/home/kylehatch/Desktop/hidql/susie-calvin-checkpoints/susie_test/test1_400smthlib_2024.02.21_06.44.06/40000/params_ema
# export GC_POLICY_CHECKPOINT=/home/kylehatch/Desktop/hidql/susie-calvin-checkpoints/susie_low_level/liberosplit3/gcdiffusion/auggoaldiff/seed_0/20240306_064653/checkpoint_150000
# agent_config_string="liberosplit3val90_gcdiffusion_noactnorm-sagemaker"


# # CALVIN SINGLE TASK
# export NUM_EVAL_SEQUENCES=10
# export EP_LEN=360
# export SUBGOAL_MAX=20
# export NUM_SAMPLES=1
# export DIFFUSION_MODEL_CHECKPOINT=/home/kylehatch/Desktop/hidql/susie-calvin-checkpoints/susie_test/public_model/checkpoint_only/params_ema
# export GC_POLICY_CHECKPOINT=/home/kylehatch/Desktop/hidql/susie-calvin-checkpoints/susie_low_level/calvin/gcdiffusion/auggoaldiff/seed_0/20240227_194024/checkpoint_150000
# agent_config_string="calvin_gcdiffusion_noactnorm-sagemaker-auggoaldiff"
# SINGLE_TASK=1


# ORACLE_GOALS_DIR="/home/kylehatch/Desktop/hidql/calvin-sim/results/single_task/calvin/public_model/checkpoint_only/calvin/gcdiffusion/auggoaldiff/seed_0/20240227_194024/checkpoint_150000/fullsusie_collect_strictthresholds_2024.03.26_11.47.26"
# # ORACLE_GOALS_TYPE="generated"
# ORACLE_GOALS_TYPE="retrospective_true"

# export DEBUG=0

# # CALVIN SINGLE TASK Chat GPT
# export NUM_EVAL_SEQUENCES=10
# export EP_LEN=360
# export SUBGOAL_MAX=20
# export SINGLE_TASK=1
# export DIFFUSION_MODEL_CHECKPOINT=/home/kylehatch/Desktop/hidql/susie-calvin-checkpoints/susie_test/public_model/checkpoint_only/params_ema
# export GC_POLICY_CHECKPOINT=/home/kylehatch/Desktop/hidql/susie-calvin-checkpoints/susie_low_level/calvin/gcdiffusion/auggoaldiff/seed_0/20240227_194024/checkpoint_150000
# agent_config_string="calvin_gcdiffusion_noactnorm-sagemaker-auggoaldiff"
# NUM_SAMPLES=4
# FILTERING_METHOD="chat_gpt_dummy"

# export DEBUG=0

# # CALVIN SINGLE TASK Chat GPT
# export NUM_EVAL_SEQUENCES=10
# export EP_LEN=360
# export SUBGOAL_MAX=20
# export SINGLE_TASK=1
# export DIFFUSION_MODEL_CHECKPOINT=/home/kylehatch/Desktop/hidql/susie-calvin-checkpoints/susie_test/public_model/checkpoint_only/params_ema
# export GC_POLICY_CHECKPOINT=/home/kylehatch/Desktop/hidql/susie-calvin-checkpoints/susie_low_level/calvinlcbc/lcdiffusion/auggoaldiff/seed_0/20240410_010011/checkpoint_150000
# agent_config_string="calvinlcbc_lcdiffusion_noactnorm-sagemaker-auggoaldiff"
# NUM_SAMPLES=1
# FILTERING_METHOD="chat_gpt_dummy"
# FLAT_POLICY=1

# export DEBUG=1



# # CALVIN SINGLE TASK Chat GPT
# export NUM_EVAL_SEQUENCES=25
# export EP_LEN=360
# export SUBGOAL_MAX=20
# export SINGLE_TASK=1
# export DIFFUSION_MODEL_CHECKPOINT=/home/kylehatch/Desktop/hidql/susie-calvin-checkpoints/susie_test/public_model/checkpoint_only/params_ema
# export GC_POLICY_CHECKPOINT=/home/kylehatch/Desktop/hidql/susie-calvin-checkpoints/susie_low_level/calvin/gcdiffusion/auggoaldiff/seed_0/20240227_194024/checkpoint_150000
# agent_config_string="calvin_gcdiffusion_noactnorm-sagemaker-auggoaldiff"
# NUM_SAMPLES=8
# FILTERING_METHOD="chat_gpt_pairwise"
# FLAT_POLICY=0

# export DEBUG=0


# # CALVIN Chat GPT
# export NUM_EVAL_SEQUENCES=2
# export SINGLE_TASK=1
# # export NUM_EVAL_SEQUENCES=100
# # export SINGLE_TASK=0
# export EP_LEN=360
# export SUBGOAL_MAX=20
# export DIFFUSION_MODEL_CHECKPOINT=/home/kylehatch/Desktop/hidql/susie-calvin-checkpoints/susie_test/public_model/checkpoint_only/params_ema
# export GC_POLICY_CHECKPOINT=/home/kylehatch/Desktop/hidql/susie-calvin-checkpoints/susie_low_level/calvin/gcdiffusion/auggoaldiff/seed_0/20240227_194024/checkpoint_150000
# agent_config_string="calvin_gcdiffusion_noactnorm-sagemaker-auggoaldiff"
# # NUM_SAMPLES=8
# # FILTERING_METHOD="chat_gpt_pairwise"
# NUM_SAMPLES=16
# # FILTERING_METHOD="chat_gpt_dummy"
# FILTERING_METHOD="human_dummy"
# FLAT_POLICY=0
# PROMPT_W=7.5
# CONTEXT_W=1.5

# export DEBUG=1


# # CALVIN Chat GPT
# export NUM_EVAL_SEQUENCES=20
# export SINGLE_TASK=1
# # export NUM_EVAL_SEQUENCES=100
# # export SINGLE_TASK=0
# export EP_LEN=360
# export SUBGOAL_MAX=20
# export DIFFUSION_MODEL_CHECKPOINT=/home/kylehatch/Desktop/hidql/susie-calvin-checkpoints/susie_test/test1_sagemaker400lbs2currdrop0_2_2024.04.29_01.00.04/40000/params_ema
# export GC_POLICY_CHECKPOINT=/home/kylehatch/Desktop/hidql/susie-calvin-checkpoints/susie_low_level/calvin/gcdiffusion/auggoaldiff/seed_0/20240227_194024/checkpoint_150000
# agent_config_string="calvin_gcdiffusion_noactnorm-sagemaker-auggoaldiff"
# # NUM_SAMPLES=8
# # FILTERING_METHOD="chat_gpt_pairwise"
# NUM_SAMPLES=1
# # FILTERING_METHOD="chat_gpt_dummy"
# FILTERING_METHOD="human_dummy"
# FLAT_POLICY=0
# PROMPT_W=7.5
# CONTEXT_W=1.5

# export DEBUG=0

# # Horizon 5
# export NUM_EVAL_SEQUENCES=100
# export SINGLE_TASK=0
# export EP_LEN=360
# export SUBGOAL_MAX=5
# export DIFFUSION_MODEL_CHECKPOINT=/home/kylehatch/Desktop/hidql/susie-calvin-checkpoints/susie_test/test1_sagemakerhorizon5_2024.04.25_01.45.24/40000/params_ema
# export GC_POLICY_CHECKPOINT=/home/kylehatch/Desktop/hidql/susie-calvin-checkpoints/susie_low_level/calvin/gcdiffusion/auggoaldiff/seed_0/20240227_194024/checkpoint_150000
# agent_config_string="calvin_gcdiffusion_noactnorm-sagemaker-auggoaldiff"
# NUM_SAMPLES=1
# FILTERING_METHOD="none"
# FLAT_POLICY=0

# export DEBUG=1



# # CALVIN SINGLE TASK Chat GPT
# export NUM_EVAL_SEQUENCES=25
# export EP_LEN=360
# export SUBGOAL_MAX=20
# export SINGLE_TASK=0
# export DIFFUSION_MODEL_CHECKPOINT=/home/kylehatch/Desktop/hidql/susie-calvin-checkpoints/susie_test/public_model/checkpoint_only/params_ema
# export GC_POLICY_CHECKPOINT=/home/kylehatch/Desktop/hidql/susie-calvin-checkpoints/susie_low_level/calvin/gcdiffusion/auggoaldiff/seed_0/20240227_194024/checkpoint_150000
# agent_config_string="calvin_gcdiffusion_noactnorm-sagemaker-auggoaldiff"
# NUM_SAMPLES=1
# FILTERING_METHOD="none"
# FLAT_POLICY=0
# PROMPT_W=7.5
# CONTEXT_W=1.5

# export DEBUG=1

# # CALVIN SINGLE TASK Chat GPT
# export NUM_EVAL_SEQUENCES=20
# export EP_LEN=360
# export SUBGOAL_MAX=20
# export SINGLE_TASK=1
# export DIFFUSION_MODEL_CHECKPOINT=/home/kylehatch/Desktop/hidql/susie-calvin-checkpoints/susie_test/public_model/checkpoint_only/params_ema
# export GC_POLICY_CHECKPOINT=/home/kylehatch/Desktop/hidql/susie-calvin-checkpoints/susie_low_level/calvin/gcdiffusion/auggoaldiff/seed_0/20240227_194024/checkpoint_150000
# export HIGH_LEVEL_VF_CHECKPOINT=/home/kylehatch/Desktop/hidql/susie-calvin-checkpoints/susie_low_level/calvinlcbc/lcgcprogressvf/auggoaldiff/seed_0/20240510_005751/checkpoint_100000
# agent_config_string="calvin_gcdiffusion_noactnorm-sagemaker-auggoaldiff"
# vf_agent_config_string="calvinlcbc_lcgcprogressvf_noactnorm-auggoaldiff"
# NUM_SAMPLES=16
# FILTERING_METHOD="high_level_vf"
# # NUM_SAMPLES=4
# # FILTERING_METHOD="human_dummy"
# FLAT_POLICY=0
# PROMPT_W=7.5
# CONTEXT_W=1.5

# export DEBUG=0
# export CUDA_VISIBLE_DEVICES=2

# CALVIN SINGLE TASK Chat GPT
export NUM_EVAL_SEQUENCES=100
export EP_LEN=360
export SUBGOAL_MAX=5
export SINGLE_TASK=0
export DIFFUSION_MODEL_CHECKPOINT=/home/kylehatch/Desktop/hidql/susie-calvin-checkpoints/susie_test/test1_sagemakerhorizon5_2024.04.25_01.45.24/40000/params_ema
export GC_POLICY_CHECKPOINT=/home/kylehatch/Desktop/hidql/susie-calvin-checkpoints/susie_low_level/calvin/gcdiffusion/auggoaldiff/seed_0/20240227_194024/checkpoint_150000
# export HIGH_LEVEL_VF_CHECKPOINT=/home/kylehatch/Desktop/hidql/susie-calvin-checkpoints/susie_low_level/calvinlcbc/lcgcprogressvf/auggoaldiff/seed_0/20240510_005751/checkpoint_100000
export HIGH_LEVEL_VF_CHECKPOINT=/home/kylehatch/Desktop/hidql/susie-calvin-checkpoints/susie_low_level/calvinlcbc/lcgcprogressvf/auggoaldiffgoaldelta5long/seed_0/20240520_230555/checkpoint_100000
agent_config_string="calvin_gcdiffusion_noactnorm-sagemaker-auggoaldiff"
vf_agent_config_string="calvinlcbc_lcgcprogressvf_noactnorm-auggoaldiff"
NUM_SAMPLES=4
FILTERING_METHOD="high_level_vf"
FLAT_POLICY=0
PROMPT_W=7.5
CONTEXT_W=1.5

export DEBUG=0
export CUDA_VISIBLE_DEVICES=1


# # liberosplit2 single task high level vf filtering
# export NUM_EVAL_SEQUENCES=20
# export EP_LEN=500
# export SUBGOAL_MAX=30
# export SINGLE_TASK=0
# export DIFFUSION_MODEL_CHECKPOINT=/home/kylehatch/Desktop/hidql/susie-calvin-checkpoints/susie_test/test1_400smthlibs2_2024.02.23_14.20.24/40000/params_ema
# export GC_POLICY_CHECKPOINT=/home/kylehatch/Desktop/hidql/susie-calvin-checkpoints/susie_low_level/liberosplit2/gcdiffusion/auggoaldiff/seed_0/20240228_024329/checkpoint_150000
# export HIGH_LEVEL_VF_CHECKPOINT=/home/kylehatch/Desktop/hidql/susie-calvin-checkpoints/susie_low_level/liberosplit2/lcgcprogressvf/auggoaldiff/seed_0/20240510_010341/checkpoint_100000
# agent_config_string="liberosplit2_gcdiffusion_noactnorm-sagemaker"
# vf_agent_config_string="liberosplit2_lcgcprogressvf_noactnorm-auggoaldiff"
# export NUM_SAMPLES=16
# FILTERING_METHOD="high_level_vf"
# FLAT_POLICY=0
# PROMPT_W=7.5
# CONTEXT_W=1.5

# export DEBUG=0
# export CUDA_VISIBLE_DEVICES=2

# # liberosplit2 single task high level vf filtering
# export NUM_EVAL_SEQUENCES=20
# export EP_LEN=500
# export SUBGOAL_MAX=30
# export SINGLE_TASK=0
# export DIFFUSION_MODEL_CHECKPOINT=/home/kylehatch/Desktop/hidql/susie-calvin-checkpoints/susie_test/test1_400smthlibs2_2024.02.23_14.20.24/40000/params_ema
# export GC_POLICY_CHECKPOINT=/home/kylehatch/Desktop/hidql/susie-calvin-checkpoints/susie_low_level/liberosplit2/gcdiffusion/auggoaldiff/seed_0/20240228_024329/checkpoint_150000
# export LOW_LEVEL_VF_CHECKPOINT=/home/kylehatch/Desktop/hidql/susie-calvin-checkpoints/susie_low_level/liberosplit2/gciql/auggoaldiffhparams5/seed_0/20240513_190251/checkpoint_100000
# agent_config_string="liberosplit2_gcdiffusion_noactnorm-sagemaker"
# vf_agent_config_string="liberosplit2_gciql_noactnorm-hparams5-auggoaldiff"
# # export NUM_SAMPLES=16
# export NUM_SAMPLES=1
# FILTERING_METHOD="low_level_vf"
# FLAT_POLICY=0
# PROMPT_W=7.5
# CONTEXT_W=1.5

# # export DEBUG=0
# export DEBUG=1
# export CUDA_VISIBLE_DEVICES=2


# # liberosplit2 single task 
# # export NUM_EVAL_SEQUENCES=20
# export NUM_EVAL_SEQUENCES=10
# export EP_LEN=500
# export SUBGOAL_MAX=30
# export SINGLE_TASK=0
# export DIFFUSION_MODEL_CHECKPOINT=/home/kylehatch/Desktop/hidql/susie-calvin-checkpoints/susie_test/test1_400smthlibs2_2024.02.23_14.20.24/40000/params_ema
# export GC_POLICY_CHECKPOINT=/home/kylehatch/Desktop/hidql/susie-calvin-checkpoints/susie_low_level/liberosplit2/gcdiffusion/auggoaldiff/seed_0/20240228_024329/checkpoint_150000
# agent_config_string="liberosplit2_gcdiffusion_noactnorm-sagemaker"
# export NUM_SAMPLES=16
# FILTERING_METHOD="human"
# FLAT_POLICY=0
# PROMPT_W=7.5
# CONTEXT_W=1.5

# export DEBUG=0


# # liberosplit2 single task 
# # export NUM_EVAL_SEQUENCES=20
# export NUM_EVAL_SEQUENCES=20
# export EP_LEN=500
# export SUBGOAL_MAX=30
# export SINGLE_TASK=0
# # export DIFFUSION_MODEL_CHECKPOINT=/home/kylehatch/Desktop/hidql/susie-calvin-checkpoints/susie_test/test1_400smthlibs2_2024.02.23_14.20.24/40000/params_ema
# export DIFFUSION_MODEL_CHECKPOINT=/home/kylehatch/Desktop/hidql/susie-calvin-checkpoints/susie_test/test1_sagemaker400lbs2promptdrop0_2_2024.04.29_00.59.42/40000/params_ema
# export GC_POLICY_CHECKPOINT=/home/kylehatch/Desktop/hidql/susie-calvin-checkpoints/susie_low_level/liberosplit2/gcdiffusion/auggoaldiff/seed_0/20240228_024329/checkpoint_150000
# agent_config_string="liberosplit2_gcdiffusion_noactnorm-sagemaker"
# export NUM_SAMPLES=1
# FILTERING_METHOD="human_dummy"
# FLAT_POLICY=0
# PROMPT_W=7.5
# CONTEXT_W=1.5

# export DEBUG=0



echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

python3 -u evaluate_policy_subgoal_diffusion.py \
--agent_config_string $agent_config_string \
--vf_agent_config_string $vf_agent_config_string \
--dataset_path mini_dataset \
--diffusion_model_checkpoint_path "file://$DIFFUSION_MODEL_CHECKPOINT" \
--gc_policy_checkpoint_path "file://$GC_POLICY_CHECKPOINT" \
--diffusion_model_framework jax \
--save_to_s3 0 \
--s3_save_uri $S3_SAVE_URI \
--use_temporal_ensembling 1 \
--num_denoising_steps 50 \
--single_task $SINGLE_TASK \
--num_samples $NUM_SAMPLES \
--filtering_method $FILTERING_METHOD \
--flat_policy $FLAT_POLICY \
--prompt_w $PROMPT_W \
--context_w $CONTEXT_W

# for ((i=0; i<$n_seeds; i++)); do
#     # export CUDA_VISIBLE_DEVICES=$i  
#     export CUDA_VISIBLE_DEVICES=2
#     echo "[$i] CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

#     python3 -u evaluate_policy_subgoal_diffusion.py \
#     --agent_config_string $agent_config_string \
#     --vf_agent_config_string $vf_agent_config_string \
#     --dataset_path mini_dataset \
#     --diffusion_model_checkpoint_path "file://$DIFFUSION_MODEL_CHECKPOINT" \
#     --gc_policy_checkpoint_path "file://$GC_POLICY_CHECKPOINT" \
#     --diffusion_model_framework jax \
#     --save_to_s3 0 \
#     --s3_save_uri $S3_SAVE_URI \
#     --use_temporal_ensembling 1 \
#     --num_denoising_steps 50 \
#     --single_task $SINGLE_TASK \
#     --num_samples $NUM_SAMPLES \
#     --filtering_method $FILTERING_METHOD \
#     --flat_policy $FLAT_POLICY \
#     --prompt_w $PROMPT_W \
#     --context_w $CONTEXT_W \
#     2>&1 | tee "results/stdouts/$current_timestamp/stdout_and_sterr_$i.txt" 

#         # --oracle_goals_dir $ORACLE_GOALS_DIR \
#     # --oracle_goals_type $ORACLE_GOALS_TYPE \

#     sleep 1
# done 


# wait 



