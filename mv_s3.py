import os 
from glob import glob 
import boto3
import time
import subprocess
import awswrangler as wr
from smart_open import smart_open

bucket_name = 'kyle-sagemaker-training-outputs'

# s3 = boto3.resource('s3')
# bucket = s3.Bucket(bucket_name)
# for obj in bucket.objects.filter(Prefix='eval-outputs/jax_model-public_checkpoint-only_checkpoint/'): 
#     copy_source = dict(Bucket=bucket_name, Key=obj.key)

#     try:
#         dst_key = os.path.join("eval-outputs/jax_model/public_checkpoint", obj.key.split("/")[2].split("-")[0], obj.key.split("/")[2].split("-")[1], "no_vf", "no_checkpoint", obj.key.split("/")[3], "1_samples", *obj.key.split("/")[4:])
#         print(f"copying \"{obj.key}\" to \"{dst_key}\"...")
#         bucket.copy(copy_source, dst_key)
#     except:
#         print(obj.key)
#         import ipdb; ipdb.set_trace()
# s3 = boto3.resource('s3')
# bucket = s3.Bucket(bucket_name)
# for obj in bucket.objects.filter(Prefix='susie_gc_low_level/'): 
#     copy_source = dict(Bucket=bucket_name, Key=obj.key)

#     try:
#         name = obj.key.split("/")[1].split("_")[0]
#         wandb_id = "_".join(obj.key.split("/")[1].split("_")[1:])
#         dst_key = os.path.join("susie_gc_low_level",  name, "seed_42", wandb_id, *obj.key.split("/")[2:])
#         print(f"copying \"{obj.key}\" to \"{dst_key}\"...")
#         # bucket.copy(copy_source, dst_key)
#     except:
#         print("obj.key:", obj.key)


s3 = boto3.resource('s3')
bucket = s3.Bucket(bucket_name)
for obj in bucket.objects.filter(Prefix='susie_gc_low_level/'): 
    copy_source = dict(Bucket=bucket_name, Key=obj.key)

    if "diffusion" in obj.key:

        try:
            import ipdb; ipdb.set_trace()
            name = obj.key.split("/")[1].split("_")[0]
            wandb_id = "_".join(obj.key.split("/")[1].split("_")[1:])
            dst_key = os.path.join("susie_gc_low_level",  name, "seed_42", wandb_id, *obj.key.split("/")[2:])
            print(f"copying \"{obj.key}\" to \"{dst_key}\"...")
            # bucket.copy(copy_source, dst_key)
        except:
            print("obj.key:", obj.key)


# s3 = boto3.resource('s3')
# bucket = s3.Bucket(bucket_name)
# for obj in bucket.objects.filter(Prefix='eval-outputs/jax_model/public_checkpoint/'): 
#     if "1_samples" in obj.key and "no_vf" not in obj.key:
#         print(obj.key)

"s3://kyle-sagemaker-training-outputs/eval-outputs/jax_model/public_checkpoint/gcbc_20240121_001822/checkpoint_50000/gciql5_20240121_005509/checkpoint_50000/50_denoising_steps/16_samples/notmpensb/2024.02.01_01.15.33/"

"s3://kyle-sagemaker-training-outputs/eval-outputs/jax_model/public_checkpoint/public_checkpoint/only_checkpoint/gciql5_20240121_005509/checkpoint_50000/50_denoising_steps/16_samples/tmpensb/2024.02.01_01.56.35/"


"""
aws s3 mv --recursive s3://kyle-sagemaker-training-outputs/eval-outputs/jax_model/public_checkpoint/gcbc_diffusion_b2048_20240131_014354/checkpoint_50000/gciql5_20240121_005509/checkpoint_50000/50_denoising_steps/1_samples s3://kyle-sagemaker-training-outputs/eval-outputs/jax_model/public_checkpoint/gcbc_diffusion_b2048_20240131_014354/checkpoint_50000/no_vf/no_checkpoint/50_denoising_steps/1_samples 
aws s3 mv --recursive s3://kyle-sagemaker-training-outputs/eval-outputs/jax_model/public_checkpoint/gcbc_diffusion_b4096_20240131_014541/checkpoint_50000/gciql5_20240121_005509/checkpoint_50000/50_denoising_steps/1_samples s3://kyle-sagemaker-training-outputs/eval-outputs/jax_model/public_checkpoint/gcbc_diffusion_b4096_20240131_014541/checkpoint_50000/no_vf/no_checkpoint/50_denoising_steps/1_samples
aws s3 mv --recursive s3://kyle-sagemaker-training-outputs/eval-outputs/jax_model/public_checkpoint/gcbc_diffusion_b8192_20240131_173254/checkpoint_50000/gciql5_20240121_005509/checkpoint_50000/50_denoising_steps/1_samples s3://kyle-sagemaker-training-outputs/eval-outputs/jax_model/public_checkpoint/gcbc_diffusion_b8192_20240131_173254/checkpoint_50000/no_vf/no_checkpoint/50_denoising_steps/1_samples


aws s3 mv --recursive s3://kyle-sagemaker-training-outputs/eval-outputs/jax_model/public_checkpoint/gcbc_diffusion_b2048_20240131_014354/checkpoint_500000/gciql5_20240121_005509/checkpoint_50000/50_denoising_steps/1_samples s3://kyle-sagemaker-training-outputs/eval-outputs/jax_model/public_checkpoint/gcbc_diffusion_b2048_20240131_014354/checkpoint_500000/no_vf/no_checkpoint/50_denoising_steps/1_samples 


aws s3 mv --recursive s3://kyle-sagemaker-training-outputs/susie_gc_low_level/gcbc_diffusion_20240121_214528/ s3://kyle-sagemaker-training-outputs/susie_gc_low_level/diffusion/seed_42/20240121_214528/ --dryrun

aws s3 cp --recursive s3://kyle-sagemaker-training-outputs/susie_gc_low_level/gcbc_20240121_001822/ s3://kyle-sagemaker-training-outputs/susie_gc_low_level/gcbc/seed_42/20240121_001822/
aws s3 cp --recursive s3://kyle-sagemaker-training-outputs/susie_gc_low_level/gcbc_diffusion_b2048_20240131_014354/ s3://kyle-sagemaker-training-outputs/susie_gc_low_level/gcdiffusionb2048/seed_42/20240131_014354/
aws s3 cp --recursive s3://kyle-sagemaker-training-outputs/susie_gc_low_level/gcbc_diffusion_b4096_20240131_014541/ s3://kyle-sagemaker-training-outputs/susie_gc_low_level/gcdiffusionb4096/seed_42/20240131_014541/
aws s3 cp --recursive s3://kyle-sagemaker-training-outputs/susie_gc_low_level/gcbc_diffusion_b8192_20240131_173254/ s3://kyle-sagemaker-training-outputs/susie_gc_low_level/gcdiffusionb8192/seed_42/20240131_173254/
aws s3 cp --recursive s3://kyle-sagemaker-training-outputs/susie_gc_low_level/gcbc_diffusion_nam_20240201_014718/ s3://kyle-sagemaker-training-outputs/susie_gc_low_level/gcdiffusionnam/seed_42/20240201_014718/
aws s3 cp --recursive s3://kyle-sagemaker-training-outputs/susie_gc_low_level/gcbc_diffusion_nam_b2048_20240201_014619/ s3://kyle-sagemaker-training-outputs/susie_gc_low_level/gcdiffusionnamb2048/seed_42/20240201_014619/
aws s3 cp --recursive s3://kyle-sagemaker-training-outputs/susie_gc_low_level/gcbc_diffusion_nam_b4096_20240201_014758/ s3://kyle-sagemaker-training-outputs/susie_gc_low_level/gcdiffusionnamb4096/seed_42/20240201_014758/
aws s3 cp --recursive s3://kyle-sagemaker-training-outputs/susie_gc_low_level/gcbc_diffusion_policy_remote_20240120_074721/ s3://kyle-sagemaker-training-outputs/susie_gc_low_level/gcdiffusionremote/seed_42/20240120_074721/
aws s3 cp --recursive s3://kyle-sagemaker-training-outputs/susie_gc_low_level/gciql_20240121_012712/ s3://kyle-sagemaker-training-outputs/susie_gc_low_level/gciql/seed_42/20240121_012712/
aws s3 cp --recursive s3://kyle-sagemaker-training-outputs/susie_gc_low_level/gciql2_20240121_005452/ s3://kyle-sagemaker-training-outputs/susie_gc_low_level/gciql2/seed_42/20240121_005452/
aws s3 cp --recursive s3://kyle-sagemaker-training-outputs/susie_gc_low_level/gciql3_20240121_005455/ s3://kyle-sagemaker-training-outputs/susie_gc_low_level/gciql3/seed_42/20240121_005455/
aws s3 cp --recursive s3://kyle-sagemaker-training-outputs/susie_gc_low_level/gciql4_20240121_012657/ s3://kyle-sagemaker-training-outputs/susie_gc_low_level/gciql4/seed_42/20240121_012657/
aws s3 cp --recursive s3://kyle-sagemaker-training-outputs/susie_gc_low_level/gciql5_20240121_005509/ s3://kyle-sagemaker-training-outputs/susie_gc_low_level/gciql5/seed_42/20240121_005509/


"""