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



    if args.input_source == 'local':
        input_mode = 'File'
        training_inputs = {"calvin_data":f'file:///home/kylehatch/Desktop/hidql/calvin/dataset/',
                           "high_level":'file:///' + args.diffusion_model_checkpoint.strip("/") + f"/params_ema"} 
    elif args.input_source == 's3': 
        input_mode = 'FastFile'
        training_inputs = {"calvin_data":"s3://susie-data/calvin_data/",
                           "high_level":args.diffusion_model_checkpoint.strip("/") + f"/params_ema"}
    else:
        raise ValueError(f'Invalid input source {args.input_source}')

    role = 'arn:aws:iam::124224456861:role/service-role/SageMaker-SageMakerAllAccess'
    role_name = role.split(['/'][-1])

    session = boto3.session.Session()
    region = session.region_name


    hyperparameters = {}


    subnets = [
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
    # entry_point = args.entry_point
    sagemaker_session = args.sagemaker_session

    instance_type = 'local_gpu' if args.local else args.instance_type 
    keep_alive_period_in_seconds = 0
    max_run = 60 * 60 * 24 * 5

    environment = {
        # 'PYTHONPATH': '/opt/ml/code:/opt/ml/code/externals/datasets:/opt/ml/code/external/dlimp', 
        'WANDB_API_KEY': args.wandb_api_key,
        'WANDB_ENTITY': "tri",
        # "DIFFUSION_MODEL_CHECKPOINT":"/opt/ml/input/data/diffusion_checkpoint/params_ema/",
        "MACHINE_START_IDX":args.machine_start_idx,
        "MACHINE_N_IDXS":args.machine_n_idxs,
        "NUM_SAMPLES":args.n_samples,
        "S3_SAVE_URI":args.s3_save_uri.strip("/"), # + "/",
        "DEBUG":args.debug,
    }



    distribution = {
        'smdistributed': {
            'dataparallel': {
                    'enabled': False,
            },
        },
    }

    print()
    print()
    print('#############################################################')
    print(f'SageMaker Execution Role:       {role}')
    print(f'The name of the Execution role: {role_name[-1]}')
    print(f'AWS region:                     {region}')
    # print(f'Entry point:                    {entry_point}')
    print(f'Image uri:                      {image_uri}')
    print(f'Job name:                       {job_name}')
    # print(f'Configuration file:             {config}')
    print(f'Instance count:                 {instance_count}')
    print(f'Input mode:                     {input_mode}')
    print('#############################################################')
    print()
    print()
    
    estimator = TensorFlow(
        base_job_name=base_job_name,
        # entry_point=entry_point,
        entry_point="train",
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
        # output_path=output_path,
        checkpoint_s3_uri=checkpoint_s3_uri,
        checkpoint_local_path=checkpoint_local_path,
        code_location=code_location,
        distribution=distribution,
    )

    estimator.fit(inputs=training_inputs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local', action='store_true', default=False)
    parser.add_argument('--base-job-name', type=str, required=True)
    parser.add_argument('--user', type=str, required=True, help='supported users under the IT-predefined bucket.')
    
    parser.add_argument('--debug', type=str, default="0"),
    parser.add_argument('--wandb-api-key', type=str, default=None)
    parser.add_argument('--input-source', choices=['s3', 'lustre', 'local'], default='lustre')
    parser.add_argument('--diffusion_model_checkpoint', type=str, default="ml.p4de.24xlarge"),
    parser.add_argument('--machine_start_idx', type=str, default="ml.p4de.24xlarge"),
    parser.add_argument('--machine_n_idxs', type=str, default="ml.p4de.24xlarge"),
    parser.add_argument('--n_samples', type=str, default="ml.p4de.24xlarge"),
    parser.add_argument('--s3_save_uri', type=str, default="s3://kyle-sagemaker-training-outputs")


    parser.add_argument('--instance-count', type=int, default=1)
    # parser.add_argument('--entry_point', type=str, default='scripts/train.py'),
    parser.add_argument('--instance_type', type=str, default="ml.p4de.24xlarge"),
    
    args = parser.parse_args()


    launch(args)

"""
./update_docker.sh
./upload_docker.sh
python3 -u sagemaker_launch_generated_goals.py \
--user kylehatch \
--base-job-name calvin-sim \
--wandb-api-key 65915e3ae3752bc3ddc4b7eef1b066067b9d1cb1 \
--machine_start_idx 0 \
--machine_n_idxs 1600 \
--n_samples 4 \
--s3_save_uri "s3://susie-data/calvin_data_processed/language_conditioned_4_samples" \
--diffusion_model_checkpoint s3://kyle-sagemaker-training-outputs/susie_test/test1_400smthlib_2024.02.21_06.44.06/40000/ \
--input-source s3 \
--instance_type ml.p4d.24xlarge 


./update_docker.sh
./upload_docker.sh
python3 -u sagemaker_launch_generated_goals.py \
--user kylehatch \
--base-job-name calvin-sim \
--wandb-api-key 65915e3ae3752bc3ddc4b7eef1b066067b9d1cb1 \
--machine_start_idx 0 \
--machine_n_idxs 16 \
--n_samples 16 \
--s3_save_uri "s3://susie-data/calvin_data_processed/language_conditioned_16_samples" \
--diffusion_model_checkpoint s3://kyle-sagemaker-training-outputs/susie_test/test1_400smthlib_2024.02.21_06.44.06/40000/ \
--input-source s3 \
--instance_type ml.p4d.24xlarge 


P4D instance
Also somehow only took a few minutes to download the entire dataset 
batch size: 2. time: 3.14

batch size: 16. time: 9.468 


./update_docker.sh
python3 -u sagemaker_launch_generated_goals.py \
--user kylehatch \
--base-job-name calvin-sim \
--wandb-api-key 65915e3ae3752bc3ddc4b7eef1b066067b9d1cb1 \
--machine_start_idx 0 \
--machine_n_idxs 6 \
--n_samples 2 \
--s3_save_uri "s3://susie-data/calvin_data_processed/language_conditioned_2_samples" \
--diffusion_model_checkpoint /home/kylehatch/Desktop/hidql/susie-calvin-checkpoints/susie_test/test1_400smthlib_2024.02.21_06.44.06/40000/ \
--input-source local \
--local \
--debug 1

batch size: 2. time: 4.224




========= calvin =========
./update_docker.sh
python3 -u sagemaker_launch.py \
--user kylehatch \
--base-job-name bridge_data_v2 \
--wandb_proj_name susie_low_level \
--wandb-api-key 65915e3ae3752bc3ddc4b7eef1b066067b9d1cb1 \
--config experiments/configs/susie/calvin/configs/gcbc_train_config.py:calvin_gcdiffusion_noactnorm-sagemaker-auggoaldiff \
--calvin_dataset_config experiments/configs/susie/calvin/configs/gcbc_data_config.py:all \
--algo gcdiffusion \
--description default \
--seeds 0 1 2  \
--log_to_wandb 1 \
--debug 1 \
--save_to_s3 1 \
--input-source local \
--local 

./update_docker.sh
./upload_docker.sh
python3 -u sagemaker_launch.py \
--user kylehatch \
--base-job-name bridge_data_v2 \
--wandb_proj_name susie_low_level \
--wandb-api-key 65915e3ae3752bc3ddc4b7eef1b066067b9d1cb1 \
--config experiments/configs/susie/calvin/configs/gcbc_train_config.py:calvin_gcdiffusion_noactnorm-sagemaker-auggoaldiff \
--calvin_dataset_config experiments/configs/susie/calvin/configs/gcbc_data_config.py:all \
--algo gcdiffusion \
--description auggoaldiff \
--seeds 0 1 2 3 \
--log_to_wandb 1 \
--debug 0 \
--save_to_s3 1 \
--input-source s3 \
--instance_type ml.p4d.24xlarge 

========= liberosplit1 =========

./update_docker.sh
python3 -u sagemaker_launch.py \
--user kylehatch \
--base-job-name bridge_data_v2 \
--wandb_proj_name susie_low_level \
--wandb-api-key 65915e3ae3752bc3ddc4b7eef1b066067b9d1cb1 \
--config experiments/configs/susie/calvin/configs/gcbc_train_config.py:liberosplit1_gcdiffusion_noactnorm-sagemaker-auggoaldiff \
--calvin_dataset_config experiments/configs/susie/calvin/configs/gcbc_data_config.py:libero \
--algo gcdiffusion \
--description auggoaldiff \
--seeds 0 1 2 \
--log_to_wandb 1 \
--debug 1 \
--save_to_s3 1 \
--input-source local \
--local 

./update_docker.sh
./upload_docker.sh
python3 -u sagemaker_launch.py \
--user kylehatch \
--base-job-name bridge_data_v2 \
--wandb_proj_name susie_low_level \
--wandb-api-key 65915e3ae3752bc3ddc4b7eef1b066067b9d1cb1 \
--config experiments/configs/susie/calvin/configs/gcbc_train_config.py:liberosplit1_gcdiffusion_noactnorm-sagemaker-auggoaldiff \
--calvin_dataset_config experiments/configs/susie/calvin/configs/gcbc_data_config.py:libero \
--algo gcdiffusion \
--description auggoaldiff \
--seeds 0 1 2 3 \
--log_to_wandb 1 \
--debug 0 \
--save_to_s3 1 \
--input-source s3 \
--instance_type ml.p4d.24xlarge 


========= liberosplit2 =========

./update_docker.sh
./upload_docker.sh
python3 -u sagemaker_launch.py \
--user kylehatch \
--base-job-name bridge_data_v2 \
--wandb_proj_name susie_low_level \
--wandb-api-key 65915e3ae3752bc3ddc4b7eef1b066067b9d1cb1 \
--config experiments/configs/susie/calvin/configs/gcbc_train_config.py:liberosplit2_gcdiffusion_noactnorm-sagemaker-auggoaldiff \
--calvin_dataset_config experiments/configs/susie/calvin/configs/gcbc_data_config.py:libero \
--algo gcdiffusion \
--description auggoaldiff \
--seeds 0 1 2 3 \
--log_to_wandb 1 \
--debug 0 \
--save_to_s3 1 \
--input-source s3 \
--instance_type ml.p4d.24xlarge 


========= liberosplit3 =========

./update_docker.sh
./upload_docker.sh
python3 -u sagemaker_launch.py \
--user kylehatch \
--base-job-name bridge_data_v2 \
--wandb_proj_name susie_low_level \
--wandb-api-key 65915e3ae3752bc3ddc4b7eef1b066067b9d1cb1 \
--config experiments/configs/susie/calvin/configs/gcbc_train_config.py:liberosplit3_gcdiffusion_noactnorm-sagemaker-auggoaldiff \
--calvin_dataset_config experiments/configs/susie/calvin/configs/gcbc_data_config.py:libero \
--algo gcdiffusion \
--description auggoaldiff \
--seeds 0 1 2 3 \
--log_to_wandb 1 \
--debug 0 \
--save_to_s3 1 \
--input-source s3 \
--instance_type ml.p4d.24xlarge 



========= liberosplit4 =========

./update_docker.sh
./upload_docker.sh
python3 -u sagemaker_launch.py \
--user kylehatch \
--base-job-name bridge_data_v2 \
--wandb_proj_name susie_low_level \
--wandb-api-key 65915e3ae3752bc3ddc4b7eef1b066067b9d1cb1 \
--config experiments/configs/susie/calvin/configs/gcbc_train_config.py:liberosplit4_gcdiffusion_noactnorm-sagemaker-auggoaldiff \
--calvin_dataset_config experiments/configs/susie/calvin/configs/gcbc_data_config.py:libero \
--algo gcdiffusion \
--description auggoaldiff \
--seeds 0 1 2 3 \
--log_to_wandb 1 \
--debug 0 \
--save_to_s3 1 \
--input-source s3 \
--instance_type ml.p4d.24xlarge 





"""