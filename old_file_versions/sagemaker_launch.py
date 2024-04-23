import argparse
import os
from datetime import datetime

import boto3
import sagemaker
# from sagemaker.pytorch import PyTorch
from sagemaker.tensorflow import TensorFlow
from sagemaker.inputs import FileSystemInput


def get_job_name(base):
    now = datetime.now()
    now_ms_str = f'{now.microsecond // 1000:03d}'
    date_str = f"{now.strftime('%Y-%m-%d-%H-%M-%S')}-{now_ms_str}"
    job_name = '_'.join([base, date_str])
    return job_name


def launch(args):

    print("args.entry_point:", args.entry_point)

    if args.wandb_api_key is None:
        wandb_api_key = os.environ.get('WANDB_API_KEY', None)
        assert wandb_api_key is not None, 'Please provide wandb api key either via --wandb-api-key or env variable WANDB_API_KEY.'
        args.wandb_api_key = wandb_api_key

    if args.local:
        assert args.instance_count == 1, f'Local mode requires 1 instance, now get {args.instance_count}.'
        assert args.input_source not in {'lustre'}
        args.sagemaker_session = sagemaker.LocalSession()
    else:
        assert args.input_source not in {'local'}    
        args.sagemaker_session = sagemaker.Session()


    multi_mode = "," in args.gc_policy_checkpoint

    if multi_mode:
        gc_policy_checkpoints = []
        for single_gc_policy_checkpoint in args.gc_policy_checkpoint.split(","):
            if single_gc_policy_checkpoint[-1] != "/":
                single_gc_policy_checkpoint += "/"
            gc_policy_checkpoints.append(single_gc_policy_checkpoint)


        gc_vf_checkpoints = []
        for single_gc_vf_checkpoint in args.gc_vf_checkpoint.split(","):
            if single_gc_vf_checkpoint[-1] != "/":
                single_gc_vf_checkpoint += "/"
            gc_vf_checkpoints.append(single_gc_vf_checkpoint)
    else:
        if args.gc_policy_checkpoint[-1] != "/":
            args.gc_policy_checkpoint += "/"

        if args.gc_vf_checkpoint[-1] != "/":
            args.gc_vf_checkpoint += "/"


    if args.diffusion_model_checkpoint[-1] != "/":
        args.diffusion_model_checkpoint += "/"


    if args.input_source == 'local':
        input_mode = 'File'
        if multi_mode:
            training_inputs = {"h-" + "-".join(args.diffusion_model_checkpoint.split("/")[-5:-2]):'file://' + args.diffusion_model_checkpoint}
            for single_gc_policy_checkpoint in gc_policy_checkpoints:
                training_inputs["l-" + "-".join(single_gc_policy_checkpoint.split("/")[-5:-1])] = 'file://' + single_gc_policy_checkpoint

            for single_gc_vf_checkpoint in gc_vf_checkpoints:
                training_inputs["v-" + "-".join(single_gc_vf_checkpoint.split("/")[-3:-1])] = 'file://' + single_gc_vf_checkpoint
        else:
            training_inputs = {"h-" + "-".join(args.diffusion_model_checkpoint.split("/")[-5:-2]):'file://' + args.diffusion_model_checkpoint,
                            "l-" + "-".join(args.gc_policy_checkpoint.split("/")[-5:-1]):'file://' + args.gc_policy_checkpoint,
                            "v-" + "-".join(args.gc_vf_checkpoint.split("/")[-3:-1]):'file://' + args.gc_vf_checkpoint}
            
    elif args.input_source == 'lustre':
        input_mode = 'File'
        train_fs = FileSystemInput(
            file_system_id='fs-02831553b25f26b1c', ###TODO
            file_system_type='FSxLustre',
            directory_path='/onhztbev', ###TODO
            file_system_access_mode='ro'
        )
    elif args.input_source == 's3':
        input_mode = 'FastFile'
        if multi_mode:
            training_inputs = {"h-" + "-".join(args.diffusion_model_checkpoint.split("/")[-5:-2]):args.diffusion_model_checkpoint}
            for single_gc_policy_checkpoint in gc_policy_checkpoints:
                training_inputs["l-" + "-".join(single_gc_policy_checkpoint.split("/")[-5:-1])] = single_gc_policy_checkpoint
            for single_gc_vf_checkpoint in gc_vf_checkpoints:
                training_inputs["v-" + "-".join(single_gc_vf_checkpoint.split("/")[-3:-1])] = single_gc_vf_checkpoint # TODO also change this to -5:-1?
        else:
            training_inputs = {"h-" + "-".join(args.diffusion_model_checkpoint.split("/")[-5:-2]):args.diffusion_model_checkpoint,
                            "l-" + "-".join(args.gc_policy_checkpoint.split("/")[-5:-1]):args.gc_policy_checkpoint,
                            "v-" + "-".join(args.gc_vf_checkpoint.split("/")[-3:-1]):args.gc_vf_checkpoint}
    else:
        raise ValueError(f'Invalid input source {args.input_source}')

    role = 'arn:aws:iam::124224456861:role/service-role/SageMaker-SageMakerAllAccess'
    role_name = role.split(['/'][-1])

    session = boto3.session.Session()
    region = session.region_name


    if "jax" in args.diffusion_model_checkpoint:
        diffusion_model_framework="jax"
    elif "pytorch" in args.diffusion_model_checkpoint:
        diffusion_model_framework="pytorch"
    else:
        raise ValueError(f"Unsupported diffusion model framework: $diffusion_model_framework, args.args.diffusion_model_checkpoint: {args.diffusion_model_checkpoint}")

    hyperparameters = {}


    subnets = [
        # 'subnet-07bf42d7c9cb929e4',
        # 'subnet-0f72615fd9bd3c717', 
        # 'subnet-0a29e4f1a47443e28', 
        # 'subnet-06e0db77592be2b36',

        'subnet-05f1115c7d6ccbd07',
        'subnet-03c7c7be28e923670',
        'subnet-0a29e4f1a47443e28',
        'subnet-06e0db77592be2b36',
        'subnet-0dd3f8c4ce7e0ae4c',
        'subnet-02a6ddd2a60a8e048',
        'subnet-060ad40beeb7f24b4',
        'subnet-036abdaead9798455',
        'subnet-07ada213d5ef651bb',
        'subnet-0e260ba29726b9fbb',
        'subnet-08468a58663b2b173',
        'subnet-0ecead4af60b3306f',
        'subnet-09b3b182287e9aa29',
        'subnet-07bf42d7c9cb929e4',
        'subnet-0f72615fd9bd3c717',
        'subnet-0578590f6bd9a5dde',
        # 'subnet-03550978b510a6d55',
        # 'subnet-0449e12487555a62a',
        # 'subnet-0a930733cdb95ffc9',
        # 'subnet-07d4dd2bc160d9df6',
        # 'subnet-016ef5d3e0df2ab9d',
        # 'subnet-0a2097ea05c20b45e',
        # 'subnet-0310cca5c76f96899',
        # 'subnet-05a638cfbfae73305',
        # 'subnet-03853f6eef9dbd13f',
    ]


    security_group_ids = [
        'sg-0afb9fb0e79a54061', 
        'sg-0333993fea1aeb948', 
        'sg-0c4b828f4023a04cc',
    ]



    job_name = get_job_name(args.base_job_name)

    if args.local:
        image_uri = f'{args.base_job_name}:latest' 
    else:
        image_uri = f'124224456861.dkr.ecr.us-east-1.amazonaws.com/{args.base_job_name}:latest'
    
    output_path = os.path.join(f's3://tri-ml-sandbox-16011-us-east-1-datasets/sagemaker/{args.user}/bridge_data_v2/', job_name)

    checkpoint_s3_uri = None if args.local else output_path
    checkpoint_local_path = None if args.local else '/opt/ml/checkpoints'
    code_location = output_path

    base_job_name = args.base_job_name.replace("_", "-")
    instance_count = args.instance_count
    entry_point = args.entry_point
    sagemaker_session = args.sagemaker_session

    instance_type = 'local_gpu' if args.local else args.instance_type 
    keep_alive_period_in_seconds = 0
    max_run = 60 * 60 * 24 * 5




    if multi_mode:
        gc_policy_checkpoint_paths = []
        for single_gc_policy_checkpoint in gc_policy_checkpoints:
            gc_policy_checkpoint_paths.append("-".join(single_gc_policy_checkpoint.split("/")[-5:-1]))
        gc_policy_checkpoint_path = ",".join(gc_policy_checkpoint_paths)


        gc_vf_checkpoint_paths = []
        for single_gc_vf_checkpoint in gc_vf_checkpoints:
            gc_vf_checkpoint_paths.append("-".join(single_gc_vf_checkpoint.split("/")[-3:-1]))
        gc_vf_checkpoint_path = ",".join(gc_vf_checkpoint_paths)

    else:
        gc_policy_checkpoint_path = "-".join(args.gc_policy_checkpoint.split("/")[-5:-1])
        gc_vf_checkpoint_path = "-".join(args.gc_vf_checkpoint.split("/")[-3:-1])
    

    environment = {
        'WANDB_API_KEY': args.wandb_api_key,
        'WANDB_ENTITY': "tri",
        'DIFFUSION_MODEL_CHECKPOINT':"/opt/ml/input/data/h",
        'GC_POLICY_CHECKPOINT':"/opt/ml/input/data/l",
        'GC_VF_CHECKPOINT':"/opt/ml/input/data/v",
        'NUM_EVAL_SEQUENCES':args.num_eval_sequences,

        # "CUDA_VISIBLE_DEVICES":"1",
        # "XLA_PYTHON_CLIENT_PREALLOCATE":"false",

        'DIFFUSION_MODEL_FRAMEWORK': diffusion_model_framework,
        'DIFFUSION_MODEL_CHECKPOINT_PATH':"-".join(args.diffusion_model_checkpoint.split("/")[-5:-2]),
        "GC_POLICY_CHECKPOINT_PATH": gc_policy_checkpoint_path,
        "GC_VF_CHECKPOINT_PATH": gc_vf_checkpoint_path,
        "SAVE_TO_S3":args.save_to_s3,
        "S3_SAVE_URI":args.s3_save_uri,

        "DEBUG":args.debug,
        "NUM_DENOISING_STEPS":args.num_denoising_steps,
        "NUM_SAMPLES":args.num_samples,
        # "USE_TEMPORAL_ENSEMBLING":args.use_temporal_ensembling
    }

    print("environment:", environment)

    distribution = {
        'smdistributed': {
            'dataparallel': {
                    'enabled': True,
            },
        },
    }

    print()
    print()
    print('#############################################################')
    print(f'SageMaker Execution Role:       {role}')
    print(f'The name of the Execution role: {role_name[-1]}')
    print(f'AWS region:                     {region}')
    print(f'Entry point:                    {entry_point}')
    print(f'Image uri:                      {image_uri}')
    print(f'Job name:                       {job_name}')
    # print(f'Configuration file:             {config}')
    print(f'Instance count:                 {instance_count}')
    print(f'Input mode:                     {input_mode}')
    print('#############################################################')
    print()
    print()

    if args.enable_ddp:
        estimator = TensorFlow(
            base_job_name=base_job_name,
            entry_point=entry_point,
            hyperparameters=hyperparameters,
            role=role,
            image_uri=image_uri,
            instance_count=instance_count,
            instance_type=instance_type,
            environment=environment,
            sagemaker_session=sagemaker_session,
            subnets=subnets,
            security_group_ids=security_group_ids,
            keep_alive_period_in_seconds=keep_alive_period_in_seconds,
            max_run=max_run,
            input_mode=input_mode,
            job_name=job_name,
            checkpoint_s3_uri=checkpoint_s3_uri,
            checkpoint_local_path=checkpoint_local_path,
            code_location=code_location,
            distribution=distribution,
        )
    else:
        estimator = TensorFlow(
        base_job_name=base_job_name,
        entry_point=entry_point,
        hyperparameters=hyperparameters,
        role=role,
        image_uri=image_uri,
        instance_count=instance_count,
        instance_type=instance_type,
        environment=environment,
        sagemaker_session=sagemaker_session,
        subnets=subnets,
        security_group_ids=security_group_ids,
        keep_alive_period_in_seconds=keep_alive_period_in_seconds,
        max_run=max_run,
        input_mode=input_mode,
        job_name=job_name,
        checkpoint_s3_uri=checkpoint_s3_uri,
        checkpoint_local_path=checkpoint_local_path,
        code_location=code_location,
    )
    estimator.fit(inputs=training_inputs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local', action='store_true', default=False)
    # parser.add_argument('--debug', type=int, default=0)
    # parser.add_argument('--save_to_s3', type=int, default=1)
    parser.add_argument('--debug', type=str, default="0")
    parser.add_argument('--save_to_s3', type=str, default="1")
    # parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--base-job-name', type=str, required=True)
    parser.add_argument('--user', type=str, required=True, help='supported users under the IT-predefined bucket.')
    parser.add_argument('--s3_save_uri', type=str, default="s3://kyle-sagemaker-training-outputs/eval-outputs")
    parser.add_argument('--diffusion_model_checkpoint', type=str)
    parser.add_argument('--gc_policy_checkpoint', type=str)
    parser.add_argument('--gc_vf_checkpoint', type=str)
    # parser.add_argument('--num_eval_sequences', type=int)
    parser.add_argument('--num_eval_sequences', type=str)
    parser.add_argument('--wandb-api-key', type=str, default=None)
    parser.add_argument('--input-source', choices=['s3', 'lustre', 'local'], default='lustre')
    parser.add_argument('--instance-count', type=int, default=1)
    # parser.add_argument('--num_denoising_steps', type=int, default=200)
    parser.add_argument('--num_denoising_steps', type=str, default="200")
    parser.add_argument('--num_samples', type=str, default="1")
    # parser.add_argument('--use_temporal_ensembling', type=str, default="1")
    parser.add_argument('--entry_point', type=str, default='scripts/train.py'),
    parser.add_argument('--enable_ddp', action='store_true', default=False)
    parser.add_argument('--instance_type', type=str, default="ml.p4de.24xlarge"),
    args = parser.parse_args()

    launch(args)

"""
=================== Local debug ===================

./update_docker.sh
python3 -u sagemaker_launch.py \
--entry_point evaluate_policy_subgoal_diffusion.py \
--user kylehatch \
--input-source local \
--base-job-name calvin-sim \
--save_to_s3 1 \
--wandb-api-key 65915e3ae3752bc3ddc4b7eef1b066067b9d1cb1 \
--diffusion_model_checkpoint /home/kylehatch/Desktop/hidql/susie-calvin-checkpoints/diffusion_model/jax_model/public_checkpoint/only_checkpoint/params_ema \
--gc_policy_checkpoint /home/kylehatch/Desktop/hidql/susie-calvin-checkpoints/gc_policy/public_checkpoint/only_checkpoint,/home/kylehatch/Desktop/hidql/susie-calvin-checkpoints/gc_policy/gcbc_20240121_001822/checkpoint_1538000,/home/kylehatch/Desktop/hidql/susie-calvin-checkpoints/gc_policy/gciql_20240121_012712/checkpoint_1538000 \
--num_eval_sequences 100 \
--num_denoising_steps 50 \
--use_temporal_ensembling 1 \
--local \
--debug 1 




./update_docker.sh
python3 -u sagemaker_launch.py \
--entry_point evaluate_policy_subgoal_diffusion.py \
--user kylehatch \
--input-source s3 \
--base-job-name calvin-sim \
--save_to_s3 1 \
--wandb-api-key 65915e3ae3752bc3ddc4b7eef1b066067b9d1cb1 \
--diffusion_model_checkpoint s3://kyle-sagemaker-training-outputs/local_checkpoints/diffusion_model/jax_model/public_checkpoint/only_checkpoint/params_ema \
--gc_policy_checkpoint s3://kyle-sagemaker-training-outputs/local_checkpoints/gc_policy/public_checkpoint/only_checkpoint,s3://kyle-sagemaker-training-outputs/susie_gc_low_level/gciql4_20240121_012657/checkpoint_50000 \
--num_eval_sequences 100 \
--num_denoising_steps 50 \
--use_temporal_ensembling 1 \
--local \
--debug 1 


=================== Remote ===================
./update_docker.sh
./upload_docker.sh
python3 -u sagemaker_launch.py \
--entry_point evaluate_policy_subgoal_diffusion.py \
--user kylehatch \
--input-source s3 \
--base-job-name calvin-sim \
--save_to_s3 1 \
--wandb-api-key 65915e3ae3752bc3ddc4b7eef1b066067b9d1cb1 \
--diffusion_model_checkpoint s3://kyle-sagemaker-training-outputs/local_checkpoints/diffusion_model/jax_model/public_checkpoint/only_checkpoint/params_ema \
--gc_policy_checkpoint s3://kyle-sagemaker-training-outputs/local_checkpoints/gc_policy/public_checkpoint/only_checkpoint,s3://kyle-sagemaker-training-outputs/local_checkpoints/gc_policy/gcbc_diffusion_policy_remote_20240120_074721/checkpoint_638000,s3://kyle-sagemaker-training-outputs/susie_gc_low_level/gciql4_20240121_012657/checkpoint_50000 \
--num_eval_sequences 100 \
--num_denoising_steps 200 \
--use_temporal_ensembling 1 \
--instance_type ml.p4d.24xlarge \
--debug 1 


Only w/ temporal ensembling
s3://kyle-sagemaker-training-outputs/local_checkpoints/gc_policy/public_checkpoint/only_checkpoint
s3://kyle-sagemaker-training-outputs/susie_gc_low_level/gcbc_diffusion_20240121_214528/checkpoint_50000/
s3://kyle-sagemaker-training-outputs/susie_gc_low_level/gcbc_diffusion_20240121_214528/checkpoint_500000/
s3://kyle-sagemaker-training-outputs/susie_gc_low_level/gcbc_diffusion_20240121_214528/checkpoint_1000000/
s3://kyle-sagemaker-training-outputs/susie_gc_low_level/gcbc_diffusion_20240121_214528/checkpoint_2000000/

W/ and w/o temporal ensembling
s3://kyle-sagemaker-training-outputs/susie_gc_low_level/gcbc_20240121_001822/checkpoint_50000
s3://kyle-sagemaker-training-outputs/susie_gc_low_level/gcbc_20240121_001822/checkpoint_500000
s3://kyle-sagemaker-training-outputs/susie_gc_low_level/gcbc_20240121_001822/checkpoint_1000000
s3://kyle-sagemaker-training-outputs/susie_gc_low_level/gcbc_20240121_001822/checkpoint_2000000

s3://kyle-sagemaker-training-outputs/susie_gc_low_level/gciql_20240121_012712/checkpoint_50000
s3://kyle-sagemaker-training-outputs/susie_gc_low_level/gciql_20240121_012712/checkpoint_500000
s3://kyle-sagemaker-training-outputs/susie_gc_low_level/gciql_20240121_012712/checkpoint_1000000
s3://kyle-sagemaker-training-outputs/susie_gc_low_level/gciql_20240121_012712/checkpoint_2000000

s3://kyle-sagemaker-training-outputs/susie_gc_low_level/gciql2_20240121_005452/checkpoint_50000
s3://kyle-sagemaker-training-outputs/susie_gc_low_level/gciql2_20240121_005452/checkpoint_500000
s3://kyle-sagemaker-training-outputs/susie_gc_low_level/gciql2_20240121_005452/checkpoint_1000000
s3://kyle-sagemaker-training-outputs/susie_gc_low_level/gciql2_20240121_005452/checkpoint_2000000

s3://kyle-sagemaker-training-outputs/susie_gc_low_level/gciql3_20240121_005455/checkpoint_50000
s3://kyle-sagemaker-training-outputs/susie_gc_low_level/gciql3_20240121_005455/checkpoint_500000
s3://kyle-sagemaker-training-outputs/susie_gc_low_level/gciql3_20240121_005455/checkpoint_1000000
s3://kyle-sagemaker-training-outputs/susie_gc_low_level/gciql3_20240121_005455/checkpoint_2000000

s3://kyle-sagemaker-training-outputs/susie_gc_low_level/gciql4_20240121_012657/checkpoint_50000
s3://kyle-sagemaker-training-outputs/susie_gc_low_level/gciql4_20240121_012657/checkpoint_500000
s3://kyle-sagemaker-training-outputs/susie_gc_low_level/gciql4_20240121_012657/checkpoint_1000000
s3://kyle-sagemaker-training-outputs/susie_gc_low_level/gciql4_20240121_012657/checkpoint_2000000

s3://kyle-sagemaker-training-outputs/susie_gc_low_level/gciql5_20240121_005509/checkpoint_50000
s3://kyle-sagemaker-training-outputs/susie_gc_low_level/gciql5_20240121_005509/checkpoint_500000
s3://kyle-sagemaker-training-outputs/susie_gc_low_level/gciql5_20240121_005509/checkpoint_1000000
s3://kyle-sagemaker-training-outputs/susie_gc_low_level/gciql5_20240121_005509/checkpoint_2000000



First, do early, mid, late my checkpoints. diff only w/ ensemble, others w/ and w/o 
Tomorrow, take best performing runs, and do w 200 denoising steps 


# Public checkpoint and my diffusion policy checkpoints Only w/ temporal ensembling
python3 -u sagemaker_launch.py \
--entry_point evaluate_policy_subgoal_diffusion.py \
--user kylehatch \
--input-source s3 \
--base-job-name calvin-sim \
--save_to_s3 1 \
--wandb-api-key 65915e3ae3752bc3ddc4b7eef1b066067b9d1cb1 \
--diffusion_model_checkpoint s3://kyle-sagemaker-training-outputs/local_checkpoints/diffusion_model/jax_model/public_checkpoint/only_checkpoint/params_ema/ \
--gc_policy_checkpoint s3://kyle-sagemaker-training-outputs/local_checkpoints/gc_policy/public_checkpoint/only_checkpoint,s3://kyle-sagemaker-training-outputs/susie_gc_low_level/gcbc_diffusion_20240121_214528/checkpoint_50000/,s3://kyle-sagemaker-training-outputs/susie_gc_low_level/gcbc_diffusion_20240121_214528/checkpoint_500000/,s3://kyle-sagemaker-training-outputs/susie_gc_low_level/gcbc_diffusion_20240121_214528/checkpoint_1000000/,s3://kyle-sagemaker-training-outputs/susie_gc_low_level/gcbc_diffusion_20240121_214528/checkpoint_2000000/ \
--num_eval_sequences 100 \
--num_denoising_steps 50 \
--use_temporal_ensembling 1 \
--instance_type ml.p4d.24xlarge


# GCBC and IQL w/ temporal ensemble
python3 -u sagemaker_launch.py \
--entry_point evaluate_policy_subgoal_diffusion.py \
--user kylehatch \
--input-source s3 \
--base-job-name calvin-sim \
--save_to_s3 1 \
--wandb-api-key 65915e3ae3752bc3ddc4b7eef1b066067b9d1cb1 \
--diffusion_model_checkpoint s3://kyle-sagemaker-training-outputs/local_checkpoints/diffusion_model/jax_model/public_checkpoint/only_checkpoint/params_ema \
--gc_policy_checkpoint s3://kyle-sagemaker-training-outputs/susie_gc_low_level/gcbc_20240121_001822/checkpoint_50000,s3://kyle-sagemaker-training-outputs/susie_gc_low_level/gcbc_20240121_001822/checkpoint_500000,s3://kyle-sagemaker-training-outputs/susie_gc_low_level/gcbc_20240121_001822/checkpoint_1000000,s3://kyle-sagemaker-training-outputs/susie_gc_low_level/gcbc_20240121_001822/checkpoint_2000000,s3://kyle-sagemaker-training-outputs/susie_gc_low_level/gciql_20240121_012712/checkpoint_50000,s3://kyle-sagemaker-training-outputs/susie_gc_low_level/gciql_20240121_012712/checkpoint_500000,s3://kyle-sagemaker-training-outputs/susie_gc_low_level/gciql_20240121_012712/checkpoint_1000000,s3://kyle-sagemaker-training-outputs/susie_gc_low_level/gciql_20240121_012712/checkpoint_2000000 \
--num_eval_sequences 100 \
--num_denoising_steps 50 \
--use_temporal_ensembling 1 \
--instance_type ml.p4d.24xlarge 


# IQL 2 and IQL 3 w/ temporal ensemble
python3 -u sagemaker_launch.py \
--entry_point evaluate_policy_subgoal_diffusion.py \
--user kylehatch \
--input-source s3 \
--base-job-name calvin-sim \
--save_to_s3 1 \
--wandb-api-key 65915e3ae3752bc3ddc4b7eef1b066067b9d1cb1 \
--diffusion_model_checkpoint s3://kyle-sagemaker-training-outputs/local_checkpoints/diffusion_model/jax_model/public_checkpoint/only_checkpoint/params_ema \
--gc_policy_checkpoint s3://kyle-sagemaker-training-outputs/susie_gc_low_level/gciql2_20240121_005452/checkpoint_50000,s3://kyle-sagemaker-training-outputs/susie_gc_low_level/gciql2_20240121_005452/checkpoint_500000,s3://kyle-sagemaker-training-outputs/susie_gc_low_level/gciql2_20240121_005452/checkpoint_1000000,s3://kyle-sagemaker-training-outputs/susie_gc_low_level/gciql2_20240121_005452/checkpoint_2000000,s3://kyle-sagemaker-training-outputs/susie_gc_low_level/gciql3_20240121_005455/checkpoint_50000,s3://kyle-sagemaker-training-outputs/susie_gc_low_level/gciql3_20240121_005455/checkpoint_500000,s3://kyle-sagemaker-training-outputs/susie_gc_low_level/gciql3_20240121_005455/checkpoint_1000000,s3://kyle-sagemaker-training-outputs/susie_gc_low_level/gciql3_20240121_005455/checkpoint_2000000 \
--num_eval_sequences 100 \
--num_denoising_steps 50 \
--use_temporal_ensembling 0 \
--instance_type ml.p4d.24xlarge


# IQL 4 and IQL 5 w/ temporal ensemble
python3 -u sagemaker_launch.py \
--entry_point evaluate_policy_subgoal_diffusion.py \
--user kylehatch \
--input-source s3 \
--base-job-name calvin-sim \
--save_to_s3 1 \
--wandb-api-key 65915e3ae3752bc3ddc4b7eef1b066067b9d1cb1 \
--diffusion_model_checkpoint s3://kyle-sagemaker-training-outputs/local_checkpoints/diffusion_model/jax_model/public_checkpoint/only_checkpoint/params_ema \
--gc_policy_checkpoint s3://kyle-sagemaker-training-outputs/susie_gc_low_level/gciql4_20240121_012657/checkpoint_50000,s3://kyle-sagemaker-training-outputs/susie_gc_low_level/gciql4_20240121_012657/checkpoint_500000,s3://kyle-sagemaker-training-outputs/susie_gc_low_level/gciql4_20240121_012657/checkpoint_1000000,s3://kyle-sagemaker-training-outputs/susie_gc_low_level/gciql4_20240121_012657/checkpoint_2000000,s3://kyle-sagemaker-training-outputs/susie_gc_low_level/gciql5_20240121_005509/checkpoint_50000,s3://kyle-sagemaker-training-outputs/susie_gc_low_level/gciql5_20240121_005509/checkpoint_500000,s3://kyle-sagemaker-training-outputs/susie_gc_low_level/gciql5_20240121_005509/checkpoint_1000000,s3://kyle-sagemaker-training-outputs/susie_gc_low_level/gciql5_20240121_005509/checkpoint_2000000 \
--num_eval_sequences 100 \
--num_denoising_steps 50 \
--use_temporal_ensembling 0 \
--instance_type ml.p4d.24xlarge









Installing collected packages: urllib3, tblib, smdebug-rulesconfig, schema, rpds-py, ppft, pox, platformdirs, pkgutil-resolve-name, jmespath, dill, cloudpickle, referencing, multiprocess, botocore, s3transfer, pathos, jsonschema-specifications, docker, jsonschema, boto3, sagemaker
  Attempting uninstall: urllib3
    Found existing installation: urllib3 2.0.7
    Uninstalling urllib3-2.0.7:
      Successfully uninstalled urllib3-2.0.7
  Attempting uninstall: cloudpickle
    Found existing installation: cloudpickle 3.0.0
    Uninstalling cloudpickle-3.0.0:
      Successfully uninstalled cloudpickle-3.0.0
Successfully installed boto3-1.34.29 botocore-1.34.29 cloudpickle-2.2.1 dill-0.3.8 docker-7.0.0 jmespath-1.0.1 jsonschema-4.21.1 jsonschema-specifications-2023.12.1 multiprocess-0.70.16 pathos-0.3.2 pkgutil-resolve-name-1.3.10 platformdirs-4.1.0 pox-0.3.4 ppft-1.7.6.8 referencing-0.33.0 rpds-py-0.17.1 s3transfer-0.10.0 sagemaker-2.205.0 schema-0.7.5 smdebug-rulesconfig-1.0.1 tblib-2.0.0 urllib3-1.26.18
"""