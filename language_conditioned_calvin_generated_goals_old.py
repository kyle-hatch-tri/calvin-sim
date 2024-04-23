"""
    This script processes the language annotated portions of the CALVIN dataset, writing it into TFRecord format.

    The dataset constructed with this script is meant to be used to train a language conditioned policy.

    Written by Pranav Atreya (pranavatreya@berkeley.edu).
"""

import numpy as np
import tensorflow as tf 
from tqdm import tqdm, trange
import os
from multiprocessing import Pool
from smart_open import smart_open

from calvin_agent.evaluation import jax_diffusion_model






def make_seven_characters(id):
    id = str(id)
    while len(id) < 7:
        id = "0" + id
    return id

def tensor_feature(value):
    return tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(value).numpy()])
    )

def string_to_feature(str_value):
    return tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[str_value.encode("UTF-8")])
    )

def process_trajectory(function_data):
    global raw_dataset_path, tfrecord_dataset_path
    use_s3, idx_range, letter, ctr, split, lang_ann = function_data
    unique_pid = split + "_" + letter + "_" + str(ctr)

    start_id, end_id = idx_range[0], idx_range[1]

    # print(f"\tstart_id: {start_id}, end_id: {end_id}")

    # We will filter the keys to only include what we need
    # Namely "rel_actions", "robot_obs", and "rgb_static"
    traj_rel_actions, traj_robot_obs, traj_rgb_static = [], [], []

    traj_rgb_generated_goals = []

    diffusion_model = jax_diffusion_model.DiffusionModel(50, num_samples=16)
    

    for ep_id in trange(start_id, end_id+1): # end_id is inclusive
        #print(unique_pid + ": iter " + str(ep_id-start_id) + " of " + str(end_id-start_id))

        ep_id = make_seven_characters(ep_id)

        if use_s3:
            with smart_open(os.path.join(raw_dataset_path, split, "episode_" + ep_id + ".npz"), 'rb') as s3_source:
                timestep_data = np.load(s3_source)
        else:
            timestep_data = np.load(os.path.join(raw_dataset_path, split, "episode_" + ep_id + ".npz"))
        
        
        rel_actions = timestep_data["rel_actions"]
        traj_rel_actions.append(rel_actions)

        robot_obs = timestep_data["robot_obs"]
        traj_robot_obs.append(robot_obs)

        rgb_static = timestep_data["rgb_static"] # not normalized, so we have to do normalization in another script
        traj_rgb_static.append(rgb_static)

        
        goal_images = diffusion_model.generate(lang_ann, rgb_static)
        traj_rgb_generated_goals.append(goal_images)
    
    traj_rel_actions, traj_robot_obs, traj_rgb_static = np.array(traj_rel_actions, dtype=np.float32), np.array(traj_robot_obs, dtype=np.float32), np.array(traj_rgb_static, dtype=np.uint8)
    traj_rgb_generated_goals = np.array(traj_rgb_generated_goals, dtype=np.uint8)

    # # Determine the output path
    write_dir = os.path.join(tfrecord_dataset_path, split, letter)
    # with open(os.path.join(write_dir, "traj" + str(ctr) + "_test.txt"), "w") as f:
    #     f.write(f"start_id: {start_id}, end_id: {end_id}\n")
    
    # Write the TFRecord
    output_tfrecord_path = os.path.join(write_dir, "traj" + str(ctr) + ".tfrecord")
    with tf.io.TFRecordWriter(output_tfrecord_path) as writer:
        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    "actions" : tensor_feature(traj_rel_actions),
                    "proprioceptive_states" : tensor_feature(traj_robot_obs),
                    "image_states" : tensor_feature(traj_rgb_static),
                    "language_annotation" : string_to_feature(lang_ann),
                    "generated_goals": tensor_feature(traj_rgb_generated_goals),
                }
            )
        )
        writer.write(example.SerializeToString())

    return end_id + 1 - start_id



    
def run_process_loop(idx_data):
    use_s3, gpu_idx, start_i, end_i = idx_data

    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_idx)

    # Let's prepare the inputs
    function_inputs = []

    # First let's do the train data
    if use_s3:
        with smart_open(os.path.join(raw_dataset_path, "training", "lang_annotations", "auto_lang_ann.npy"), 'rb') as s3_source:
            auto_lang_ann = np.load(s3_source, allow_pickle=True)
    else:
        auto_lang_ann = np.load(os.path.join(raw_dataset_path, "training", "lang_annotations", "auto_lang_ann.npy"), allow_pickle=True)

    auto_lang_ann = auto_lang_ann.item()
    all_language_annotations = auto_lang_ann["language"]["ann"]
    idx_ranges = auto_lang_ann["info"]["indx"]

    if use_s3:
        with smart_open(os.path.join(raw_dataset_path, "training", "scene_info.npy"), 'rb') as s3_source:
            scene_info = np.load(s3_source, allow_pickle=True)
    else:
        scene_info = np.load(os.path.join(raw_dataset_path, "training", "scene_info.npy"), allow_pickle=True)

    scene_info = scene_info.item()

    A_ctr, B_ctr, C_ctr, D_ctr = 0, 0, 0, 0
    for i, idx_range in enumerate(idx_ranges):
        start_idx = idx_range[0]
        if start_idx <= scene_info["calvin_scene_D"][1]:
            ctr = D_ctr
            D_ctr += 1
            letter = "D"
        elif start_idx <= scene_info["calvin_scene_B"][1]: # This is actually correct. In ascending order we have D, B, C, A
            ctr = B_ctr
            B_ctr += 1
            letter = "B"
        elif start_idx <= scene_info["calvin_scene_C"][1]:
            ctr = C_ctr
            C_ctr += 1
            letter = "C"
        else:
            ctr = A_ctr
            A_ctr += 1
            letter = "A"

        function_inputs.append((use_s3, idx_range, letter, ctr, "training", all_language_annotations[i]))

    # Next let's do the validation data
    if use_s3:
        with smart_open(os.path.join(raw_dataset_path, "validation", "lang_annotations", "auto_lang_ann.npy"), 'rb') as s3_source: 
            auto_lang_ann = np.load(s3_source, allow_pickle=True)
    else:
        auto_lang_ann = np.load(os.path.join(raw_dataset_path, "validation", "lang_annotations", "auto_lang_ann.npy"), allow_pickle=True)

    auto_lang_ann = auto_lang_ann.item()
    all_language_annotations = auto_lang_ann["language"]["ann"]
    idx_ranges = auto_lang_ann["info"]["indx"]

    ctr = 0
    for i, idx_range in enumerate(idx_ranges):
        function_inputs.append((use_s3, idx_range, "D", ctr, "validation", all_language_annotations[i]))
        ctr += 1



    traj_lens = []
    # print("Before process")
    # print("len(function_inputs):", len(function_inputs))
    for i, function_input in tqdm(enumerate(function_inputs)):
        if i >= start_i and i < end_i:
            print(f"[{gpu_idx}] Processing {i}")
            traj_len = process_trajectory(function_input)
            traj_lens.append(traj_len)
        else:
            # print(f"Skipping {i}")
            pass 
    # print("After process")

    # print("np.sum(traj_lens):", np.sum(traj_lens))
    # print("np.mean(traj_lens):", np.mean(traj_lens))
    # print("len(traj_lens):", len(traj_lens))


def aws_s3_sync(source, destination):
    """aws s3 sync in quiet mode and time profile"""
    import time, subprocess

    cmd = ["aws", "s3", "sync", "--quiet", source, destination]
    print(f"Syncing files from {source} to {destination}")
    start_time = time.time()
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    p.wait()
    end_time = time.time()
    print("Time Taken to Sync: ", (end_time - start_time))
    return


def sync_local_to_s3(
    local_path,
    s3_uri,
):
    """ sample function to sync checkpoints from local path to s3 """

    import boto3

    # check if local path exists
    if not os.path.exists(local_path):
        raise RuntimeError(
            f"Provided local path {local_path} does not exist. Please check"
        )

    # check if s3 bucket exists
    s3 = boto3.resource("s3")
    if not s3_uri.startswith("s3://"):
        raise ValueError(f"Provided s3 uri {s3_uri} is not valid.")

    s3_bucket = s3_uri.replace("s3://", "").split("/")[0]
    print(f"S3 Bucket: {s3_bucket}")
    try:
        s3.meta.client.head_bucket(Bucket=s3_bucket)
    except Exception as e:
        raise e
    aws_s3_sync(local_path, s3_uri)
    return 
 

def sync_s3_to_local(
    s3_uri,
    local_path
):
    """ sample function to sync checkpoints from local path to s3 """

    import boto3

    # # check if local path exists
    # if not os.path.exists(local_path):
    #     raise RuntimeError(
    #         f"Provided local path {local_path} does not exist. Please check"
    #     )
    os.makedirs(local_path, exist_ok=True)

    # check if s3 bucket exists
    s3 = boto3.resource("s3")
    if not s3_uri.startswith("s3://"):
        raise ValueError(f"Provided s3 uri {s3_uri} is not valid.")

    s3_bucket = s3_uri.replace("s3://", "").split("/")[0]
    print(f"S3 Bucket: {s3_bucket}")
    try:
        s3.meta.client.head_bucket(Bucket=s3_bucket)
    except Exception as e:
        raise e
    aws_s3_sync(s3_uri, local_path)
    return 


def parse_arguments():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate a trained model on multistep sequences with language goals.")
    # parser.add_argument("--split", type=str, choices=["training", "validation"], help="Path to the dataset root directory.")
    parser.add_argument("--start_i", type=int, help="Path to the dataset root directory.")
    parser.add_argument("--n_idxs", type=int, help="Path to the dataset root directory.")
    parser.add_argument("--n_gpus", type=int, help="Path to the dataset root directory.")
    parser.add_argument("--s3_dir", type=str, default=None, help="Path to the dataset root directory.")
    parser.add_argument("--s3", type=int, default=0, help="Path to the dataset root directory.")
    return parser.parse_args()


args = parse_arguments()

if args.s3:
    # raw_dataset_path = "/opt/ml/input/data/calvin/dataset/task_ABCD_D"
    raw_dataset_path = "s3://susie-data/calvin_data/task_ABCD_D"

    # sync_s3_to_local(args.s3_dir, raw_dataset_path)


    # tfrecord_dataset_path = "/opt/ml/input/data/calvin_data_processed/language_conditioned_with_generated"# TODO: change these for docker 
    tfrecord_dataset_path = "/home/kylehatch/Desktop/hidql/data/calvin_data_processed/language_conditioned_with_generated"# TODO: change these for docker
else:
    ########## Dataset paths ###########
    # raw_dataset_path = "/home/kylehatch/Desktop/hidql/data/calvin_data/task_ABCD_D"
    raw_dataset_path = "/home/kylehatch/Desktop/hidql/calvin/dataset/task_ABCD_D"
    tfrecord_dataset_path = "/home/kylehatch/Desktop/hidql/data/calvin_data_processed/language_conditioned_with_generated"# TODO: change these for docker 

print("raw_dataset_path:", raw_dataset_path)

########## Main logic ###########
if not os.path.exists(tfrecord_dataset_path):
    os.mkdir(tfrecord_dataset_path)
if not os.path.exists(os.path.join(tfrecord_dataset_path, "training")):
    os.mkdir(os.path.join(tfrecord_dataset_path, "training"))
if not os.path.exists(os.path.join(tfrecord_dataset_path, "validation")):
    os.mkdir(os.path.join(tfrecord_dataset_path, "validation"))
if not os.path.exists(os.path.join(tfrecord_dataset_path, "training/A")):
    os.mkdir(os.path.join(tfrecord_dataset_path, "training/A"))
if not os.path.exists(os.path.join(tfrecord_dataset_path, "training/B")):
    os.mkdir(os.path.join(tfrecord_dataset_path, "training/B"))
if not os.path.exists(os.path.join(tfrecord_dataset_path, "training/C")):
    os.mkdir(os.path.join(tfrecord_dataset_path, "training/C"))
if not os.path.exists(os.path.join(tfrecord_dataset_path, "training/D")):
    os.mkdir(os.path.join(tfrecord_dataset_path, "training/D"))
if not os.path.exists(os.path.join(tfrecord_dataset_path, "validation/D")):
    os.mkdir(os.path.join(tfrecord_dataset_path, "validation/D"))


# num_idxs = args.end_i - args.start_i
assert args.n_idxs % args.n_gpus == 0

n_idxs_per_gpu = args.n_idxs // args.n_gpus
start_and_end_idxs = []
for i in range(args.n_gpus):
    start_idx = args.start_i + (i * n_idxs_per_gpu)
    end_idx = start_idx + n_idxs_per_gpu
    start_and_end_idxs.append((args.s3, i, start_idx, end_idx))

with Pool(len(start_and_end_idxs)) as p: # We have one process per input because we are io bound, not cpu bound
    p.map(run_process_loop, start_and_end_idxs)
    
# for idx_info in start_and_end_idxs:
#     run_process_loop(idx_info)

if args.s3:
    sync_local_to_s3(tfrecord_dataset_path, args.s3_dir)
    


"""
export PYTHONPATH="/home/kylehatch/Desktop/hidql/bridge_data_v2/external/susie:$PYTHONPATH"
export DIFFUSION_MODEL_CHECKPOINT=/home/kylehatch/Desktop/hidql/susie-calvin-checkpoints/susie_test/test1_400smthlib_2024.02.21_06.44.06/40000/params_ema
python3 -u experiments/configs/susie/calvin/dataset_conversion_scripts/language_conditioned_calvin_generated_goals.py

export DIFFUSION_MODEL_CHECKPOINT=/home/kylehatch/Desktop/hidql/susie-calvin-checkpoints/susie_test/test1_400smthlib_2024.02.21_06.44.06/40000/params_ema
python3 -u language_conditioned_calvin_generated_goals.py \
--start_i 0 \
--n_gpus 8 \
--n_idxs 1600


export DIFFUSION_MODEL_CHECKPOINT=/home/kylehatch/Desktop/hidql/susie-calvin-checkpoints/susie_test/test1_400smthlib_2024.02.21_06.44.06/40000/params_ema
python3 -u language_conditioned_calvin_generated_goals.py \
--start_i 0 \
--n_gpus 8 \
--n_idxs 1600



export DIFFUSION_MODEL_CHECKPOINT=/home/kylehatch/Desktop/hidql/susie-calvin-checkpoints/susie_test/test1_400smthlib_2024.02.21_06.44.06/40000/params_ema
python3 -u language_conditioned_calvin_generated_goals.py \
--start_i 100 \
--n_gpus 3 \
--n_idxs 6 \
--s3 1 \
--s3_dir s3://susie-data/calvin_data_processed/language_conditioned_with_generated


export DIFFUSION_MODEL_CHECKPOINT=/home/kylehatch/Desktop/hidql/susie-calvin-checkpoints/susie_test/test1_400smthlib_2024.02.21_06.44.06/40000/params_ema
python3 -u language_conditioned_calvin_generated_goals.py \
--start_i 100 \
--n_gpus 1 \
--n_idxs 2 \
--s3 1 \
--s3_dir s3://susie-data/calvin_data_processed/language_conditioned_with_generated


(susie-calvin) kylehatch@TRI-251002:~/Desktop/hidql/calvin-sim$ aws s3 ls s3://susie-data/calvin_data/task_ABCD_D/training/ | wc -l 
1283407
(susie-calvin) kylehatch@TRI-251002:~/Desktop/hidql/calvin$ ls dataset/task_ABCD_D/training/ | wc -l
2307138

"""