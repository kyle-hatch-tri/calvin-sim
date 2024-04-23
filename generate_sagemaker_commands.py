AWS_ACCESS_KEY_ID="ASIARZ3C2ZCORVQTLKWP"
AWS_SECRET_ACCESS_KEY="D3uiCcRHa0T69wxDs1qt1EcawIOB3G6YxGA+xGk/"
AWS_SESSION_TOKEN="IQoJb3JpZ2luX2VjEIT//////////wEaCXVzLWVhc3QtMSJGMEQCICEh1SdkMbKwMw70iy3iLXpwfewU844gUWT+/acFGis6AiApmjL26AR7/vM3e7diJ30StDu35sOR8Mp5ZZx9vdi7lSqJAwhsEAQaDDEyNDIyNDQ1Njg2MSIM4QmeU/HmPSkvxhWXKuYCHXACjYuB7d/MuVHqmnoeBUge9Y/V7qrhJO115ClQs8YrgxF8k5XN5JeHum+6r8g3kvslcS21Ri+1EWBAjujFSx29yEgYdAn5D82zK0XKz/TO7E5vEMVQ8fdVYtKNq5AcOWak1s1a/T1kJLhaoUoSxsniL9D4G1Ltx1ewFJr6bZEnsBWS7nMb/3W7R610by+hYj1MTFBSV0jke9QeZJ+2QWTYS2M5gVFmH/c4kKMFOBcagyQvMXl0RzLNvE7tkphnbP8aOcQxjNlGt6zQO0XlwQSUx2vdXeuUIswSY/hh6pHxv1gG0GSot6f3NtQ6vdF5xXtm58+z3JzlCDc7QQsglsbpGMjFEbRepm0DFjlk1yx+AIPLlWRc5fbWzBB0nHHjGnN8a2DVoSWXdJvysgzGGruEf/IJg4p3wYvIHK1PfvxG0DUYRMFqjky2VzQOQjV3fbhCcKIZT8dLJPAUGAgsELngxOAf/TCDmuCuBjqnAa/g0Wf8Qx159AhSEwJAccHHmbivBxTea9ojocLVaJzCG4LpIqcwusMk2eNNppCGyp0Pf1gKIYLxC+x86Fc3egM5DDkXRvEHo0R26iV4uYplALS+cQ0htD+H0GmU+u1hmiqnBKlHGNPc8kibBPFK27tV6OczTIW3/AuHLQ1SYPA7rlqjTMy+U+dM1ZYuuZedSfUax6N3Ta2zarx8Vcm3DS6w/GS+M1mG"

DIFFUSION_MODEL_CHECKPOINT = "s3://kyle-sagemaker-training-outputs/local_checkpoints/diffusion_model/jax_model/public_checkpoint/only_checkpoint/params_ema"

# # GC_POLICY_CHECKPOINTS = """
# # s3://kyle-sagemaker-training-outputs/local_checkpoints/gc_policy/public_checkpoint/only_checkpoint
# # s3://kyle-sagemaker-training-outputs/susie_gc_low_level/gcbc_diffusion_20240121_214528/checkpoint_50000
# # s3://kyle-sagemaker-training-outputs/susie_gc_low_level/gcbc_20240121_001822/checkpoint_50000
# # s3://kyle-sagemaker-training-outputs/susie_gc_low_level/gciql_20240121_012712/checkpoint_50000
# # s3://kyle-sagemaker-training-outputs/susie_gc_low_level/gciql2_20240121_005452/checkpoint_50000
# # s3://kyle-sagemaker-training-outputs/susie_gc_low_level/gciql3_20240121_005455/checkpoint_50000
# # s3://kyle-sagemaker-training-outputs/susie_gc_low_level/gciql4_20240121_012657/checkpoint_50000
# # s3://kyle-sagemaker-training-outputs/susie_gc_low_level/gciql5_20240121_005509/checkpoint_50000
# # """
# GC_POLICY_CHECKPOINTS = """
# s3://kyle-sagemaker-training-outputs/susie_gc_low_level/gcbc_20240121_001822/checkpoint_50000
# s3://kyle-sagemaker-training-outputs/susie_gc_low_level/gcbc_20240121_001822/checkpoint_50000
# s3://kyle-sagemaker-training-outputs/susie_gc_low_level/gcbc_20240121_001822/checkpoint_50000
# s3://kyle-sagemaker-training-outputs/susie_gc_low_level/gcbc_20240121_001822/checkpoint_50000
# s3://kyle-sagemaker-training-outputs/susie_gc_low_level/gcbc_20240121_001822/checkpoint_50000
# s3://kyle-sagemaker-training-outputs/susie_gc_low_level/gcbc_20240121_001822/checkpoint_50000
# s3://kyle-sagemaker-training-outputs/susie_gc_low_level/gcbc_20240121_001822/checkpoint_50000
# s3://kyle-sagemaker-training-outputs/susie_gc_low_level/gcbc_20240121_001822/checkpoint_50000
# """

# GC_POLICY_CHECKPOINTS = """
# s3://kyle-sagemaker-training-outputs/susie_gc_low_level/gcbc_20240121_001822/checkpoint_50000
# s3://kyle-sagemaker-training-outputs/susie_gc_low_level/gcbc_20240121_001822/checkpoint_50000
# s3://kyle-sagemaker-training-outputs/susie_gc_low_level/gcbc_20240121_001822/checkpoint_50000
# s3://kyle-sagemaker-training-outputs/susie_gc_low_level/gcbc_20240121_001822/checkpoint_50000
# s3://kyle-sagemaker-training-outputs/susie_gc_low_level/gcbc_20240121_001822/checkpoint_50000
# s3://kyle-sagemaker-training-outputs/susie_gc_low_level/gcbc_20240121_001822/checkpoint_50000
# s3://kyle-sagemaker-training-outputs/susie_gc_low_level/gcbc_20240121_001822/checkpoint_50000
# s3://kyle-sagemaker-training-outputs/susie_gc_low_level/gcbc_20240121_001822/checkpoint_50000
# """

# GC_VF_CHECKPONTS = """
# s3://kyle-sagemaker-training-outputs/susie_gc_low_level/gciql5_20240121_005509/checkpoint_50000
# s3://kyle-sagemaker-training-outputs/susie_gc_low_level/gciql5_20240121_005509/checkpoint_50000
# s3://kyle-sagemaker-training-outputs/susie_gc_low_level/gciql5_20240121_005509/checkpoint_50000
# s3://kyle-sagemaker-training-outputs/susie_gc_low_level/gciql5_20240121_005509/checkpoint_50000
# s3://kyle-sagemaker-training-outputs/susie_gc_low_level/gciql5_20240121_005509/checkpoint_50000
# s3://kyle-sagemaker-training-outputs/susie_gc_low_level/gciql5_20240121_005509/checkpoint_50000
# s3://kyle-sagemaker-training-outputs/susie_gc_low_level/gciql5_20240121_005509/checkpoint_50000
# s3://kyle-sagemaker-training-outputs/susie_gc_low_level/gciql5_20240121_005509/checkpoint_50000
# """




# GC_POLICY_CHECKPOINTS = """
# s3://kyle-sagemaker-training-outputs/susie_gc_low_level/gcbc_diffusion_b2048_20240131_014354/checkpoint_50000/
# s3://kyle-sagemaker-training-outputs/susie_gc_low_level/gcbc_diffusion_b4096_20240131_014541/checkpoint_50000/
# s3://kyle-sagemaker-training-outputs/susie_gc_low_level/gcbc_diffusion_b4096_20240131_014541/checkpoint_500000/
# s3://kyle-sagemaker-training-outputs/susie_gc_low_level/gcbc_diffusion_b8192_20240131_173254/checkpoint_50000/
# s3://kyle-sagemaker-training-outputs/susie_gc_low_level/gcbc_diffusion_noactnorm_20240201_014718/checkpoint_50000/
# s3://kyle-sagemaker-training-outputs/susie_gc_low_level/gcbc_diffusion_noactnorm_20240201_014718/checkpoint_500000/
# s3://kyle-sagemaker-training-outputs/susie_gc_low_level/gcbc_diffusion_noactnorm_b2048_20240201_014619/checkpoint_50000/
# s3://kyle-sagemaker-training-outputs/susie_gc_low_level/gcbc_diffusion_noactnorm_b4096_20240201_014758/checkpoint_50000/
# """


# GC_VF_CHECKPONTS = """
# s3://kyle-sagemaker-training-outputs/susie_gc_low_level/gciql5_20240121_005509/checkpoint_50000
# s3://kyle-sagemaker-training-outputs/susie_gc_low_level/gciql5_20240121_005509/checkpoint_50000
# s3://kyle-sagemaker-training-outputs/susie_gc_low_level/gciql5_20240121_005509/checkpoint_50000
# s3://kyle-sagemaker-training-outputs/susie_gc_low_level/gciql5_20240121_005509/checkpoint_50000
# s3://kyle-sagemaker-training-outputs/susie_gc_low_level/gciql5_20240121_005509/checkpoint_50000
# s3://kyle-sagemaker-training-outputs/susie_gc_low_level/gciql5_20240121_005509/checkpoint_50000
# s3://kyle-sagemaker-training-outputs/susie_gc_low_level/gciql5_20240121_005509/checkpoint_50000
# s3://kyle-sagemaker-training-outputs/susie_gc_low_level/gciql5_20240121_005509/checkpoint_50000
# """

# GC_POLICY_CHECKPOINTS = """
# s3://kyle-sagemaker-training-outputs/susie_gc_low_level/gcbc_diffusion_nam_20240201_014718/checkpoint_60000/
# s3://kyle-sagemaker-training-outputs/susie_gc_low_level/gcbc_diffusion_nam_b2048_20240201_014619/checkpoint_50000/
# s3://kyle-sagemaker-training-outputs/susie_gc_low_level/gcbc_diffusion_nam_b4096_20240201_014758/checkpoint_50000/
# s3://kyle-sagemaker-training-outputs/susie_gc_low_level/gcbc_diffusion_nam_20240201_014718/checkpoint_490000/
# s3://kyle-sagemaker-training-outputs/susie_gc_low_level/gcbc_diffusion_nam_b2048_20240201_014619/checkpoint_500000/
# s3://kyle-sagemaker-training-outputs/susie_gc_low_level/gcbc_diffusion_nam_b4096_20240201_014758/checkpoint_390000/
# """
# GC_POLICY_CHECKPOINTS = """
# s3://kyle-sagemaker-training-outputs/susie_gc_low_level/gcbc_diffusion_nam_20240201_014718/checkpoint_60000
# s3://kyle-sagemaker-training-outputs/susie_gc_low_level/gcbc_diffusion_nam_20240201_014718/checkpoint_600000
# s3://kyle-sagemaker-training-outputs/susie_gc_low_level/gcbc_diffusion_nam_20240201_014718/checkpoint_1020000
# s3://kyle-sagemaker-training-outputs/susie_gc_low_level/gcbc_diffusion_nam_20240201_014718/checkpoint_1600000
# s3://kyle-sagemaker-training-outputs/susie_gc_low_level/gcbc_diffusion_nam_20240201_014718/checkpoint_1940000
# s3://kyle-sagemaker-training-outputs/susie_gc_low_level/gcbc_diffusion_nam_20240201_014718/checkpoint_20000
# s3://kyle-sagemaker-training-outputs/susie_gc_low_level/gcbc_diffusion_nam_20240201_014718/checkpoint_140000
# s3://kyle-sagemaker-training-outputs/susie_gc_low_level/gcbc_diffusion_nam_20240201_014718/checkpoint_40000
# """
GC_POLICY_CHECKPOINTS = """
s3://kyle-sagemaker-training-outputs/susie_gc_low_level/diffusion_ng/seed_42/20240225_053306/checkpoint_50000/
s3://kyle-sagemaker-training-outputs/susie_gc_low_level/diffusion_ng/seed_42/20240225_053306/checkpoint_100000/
s3://kyle-sagemaker-training-outputs/susie_gc_low_level/diffusion_ng/seed_42/20240225_053306/checkpoint_150000/
s3://kyle-sagemaker-training-outputs/susie_gc_low_level/diffusion_ng/seed_42/20240225_053306/checkpoint_200000/
s3://kyle-sagemaker-training-outputs/susie_gc_low_level/diffusion_ng/seed_42/20240225_053306/checkpoint_250000/
s3://kyle-sagemaker-training-outputs/susie_gc_low_level/diffusion_ng/seed_42/20240225_053306/checkpoint_300000/
s3://kyle-sagemaker-training-outputs/susie_gc_low_level/diffusion_ng/seed_42/20240225_053306/checkpoint_500000/
s3://kyle-sagemaker-training-outputs/susie_gc_low_level/diffusion_ng/seed_42/20240225_053306/checkpoint_600000/
"""


GC_VF_CHECKPONTS = """
s3://kyle-sagemaker-training-outputs/susie_gc_low_level/gciql5_20240121_005509/checkpoint_50000
s3://kyle-sagemaker-training-outputs/susie_gc_low_level/gciql5_20240121_005509/checkpoint_50000
s3://kyle-sagemaker-training-outputs/susie_gc_low_level/gciql5_20240121_005509/checkpoint_50000
s3://kyle-sagemaker-training-outputs/susie_gc_low_level/gciql5_20240121_005509/checkpoint_50000
s3://kyle-sagemaker-training-outputs/susie_gc_low_level/gciql5_20240121_005509/checkpoint_50000
s3://kyle-sagemaker-training-outputs/susie_gc_low_level/gciql5_20240121_005509/checkpoint_50000
s3://kyle-sagemaker-training-outputs/susie_gc_low_level/gciql5_20240121_005509/checkpoint_50000
s3://kyle-sagemaker-training-outputs/susie_gc_low_level/gciql5_20240121_005509/checkpoint_50000
"""

gc_policy_checkpoints_string = ",".join(GC_POLICY_CHECKPOINTS.strip('\n').split('\n'))
gc_vf_checkpoints_string = ",".join(GC_VF_CHECKPONTS.strip('\n').split('\n'))


command_str = f"""
export AWS_ACCESS_KEY_ID=\"{AWS_ACCESS_KEY_ID}\"
export AWS_SECRET_ACCESS_KEY=\"{AWS_SECRET_ACCESS_KEY}\"
export AWS_SESSION_TOKEN=\"{AWS_SESSION_TOKEN}\"


./update_docker.sh
./upload_docker.sh
python3 -u sagemaker_launch.py \\
--entry_point evaluate_policy_subgoal_diffusion.py \\
--user kylehatch \\
--input-source s3 \\
--base-job-name calvin-sim \\
--save_to_s3 1 \\
--wandb-api-key 65915e3ae3752bc3ddc4b7eef1b066067b9d1cb1 \\
--diffusion_model_checkpoint {DIFFUSION_MODEL_CHECKPOINT} \\
--gc_policy_checkpoint {gc_policy_checkpoints_string} \\
--gc_vf_checkpoint {gc_vf_checkpoints_string} \\
--num_eval_sequences 100 \\
--num_denoising_steps 50 \\
--num_samples 1 \\
--instance_type ml.p4d.24xlarge 


./update_docker.sh
python3 -u sagemaker_launch.py \\
--entry_point evaluate_policy_subgoal_diffusion.py \\
--user kylehatch \\
--input-source s3 \\
--base-job-name calvin-sim \\
--save_to_s3 1 \\
--wandb-api-key 65915e3ae3752bc3ddc4b7eef1b066067b9d1cb1 \\
--diffusion_model_checkpoint {DIFFUSION_MODEL_CHECKPOINT} \\
--gc_policy_checkpoint {gc_policy_checkpoints_string} \\
--gc_vf_checkpoint {gc_vf_checkpoints_string} \\
--num_eval_sequences 100 \\
--num_denoising_steps 50 \\
--num_samples 1 \\
--local \\
--debug 1
"""

print(command_str)


"""
TO DOs

/home/kylehatch/Desktop/hidql/calvin-sim/calvin_models/calvin_agent/evaluation/diffusion_gc_policy.py
normalize_actions=False if "noactnorm" in self.agent_type else True # TODO better way to handle this
Better way to handle all the kwargs. Copy of the gcbc_train_config.py 



./update_docker.sh
python3 -u sagemaker_launch.py \\
--entry_point evaluate_policy_subgoal_diffusion.py \\
--user kylehatch \\
--input-source s3 \\
--base-job-name calvin-sim \\
--save_to_s3 0 \\
--wandb-api-key 65915e3ae3752bc3ddc4b7eef1b066067b9d1cb1 \\
--diffusion_model_checkpoint {DIFFUSION_MODEL_CHECKPOINT} \\
--gc_policy_checkpoint {gc_policy_checkpoints_string} \\
--gc_vf_checkpoint {gc_vf_checkpoints_string} \\
--num_eval_sequences 100 \\
--num_denoising_steps 50 \\
--num_samples 1 \\
--local \\
--debug 1
"""


