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
import time
import shutil
from glob import glob 

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

    debug, s3_smart_load, num_samples, gpu_idx, idx_range, letter, ctr, split, lang_ann = function_data
    unique_pid = split + "_" + letter + "_" + str(ctr)

    start_id, end_id = idx_range[0], idx_range[1]

    if debug:
        end_id = min(end_id, start_id + 3)

    # print(f"\tstart_id: {start_id}, end_id: {end_id}")

    # We will filter the keys to only include what we need
    # Namely "rel_actions", "robot_obs", and "rgb_static"
    traj_rel_actions, traj_robot_obs, traj_rgb_static = [], [], []

    traj_rgb_generated_goals = []
    traj_rgb_encoded_decoded = []
    traj_rgb_noised_encoded_decoded = []

    diffusion_model = jax_diffusion_model.DiffusionModel(50, num_samples=num_samples)
    
    progress_bar = trange(start_id, end_id+1)
    
    for ep_id in progress_bar: # end_id is inclusive
        #print(unique_pid + ": iter " + str(ep_id-start_id) + " of " + str(end_id-start_id))

        ep_id = make_seven_characters(ep_id)

        if s3_smart_load:
            t0 = time.time()
            with smart_open(os.path.join(raw_dataset_path, split, "episode_" + ep_id + ".npz"), 'rb') as s3_source:
                timestep_data = np.load(s3_source)
            t1 = time.time()

            print(f"smart_open load time{t1 - t0:.3f}")


        else:
            timestep_data = np.load(os.path.join(raw_dataset_path, split, "episode_" + ep_id + ".npz"))


        
        
        
        rel_actions = timestep_data["rel_actions"]
        traj_rel_actions.append(rel_actions)

        robot_obs = timestep_data["robot_obs"]
        traj_robot_obs.append(robot_obs)

        rgb_static = timestep_data["rgb_static"] # not normalized, so we have to do normalization in another script
        traj_rgb_static.append(rgb_static)

        
        goal_images, inference_time_generated = diffusion_model.generate(lang_ann, rgb_static, return_inference_time=True)
        traj_rgb_generated_goals.append(goal_images)

        encoded_decoded_images, inference_time_encoded_decoded = diffusion_model.vae_encode_decode(rgb_static, return_inference_time=True)
        traj_rgb_encoded_decoded.append(encoded_decoded_images)


        noised_encoded_decoded_images, inference_time_encoded_decoded_noised = diffusion_model.vae_encode_decode(rgb_static, noise_scale=0.5, return_inference_time=True)
        traj_rgb_noised_encoded_decoded.append(noised_encoded_decoded_images)


        progress_bar.set_description(f"[GPU {gpu_idx}] processing {start_id} to {end_id}, inference_time_generated: {inference_time_generated:.3f}s,  inference_time_encoded_decoded: {inference_time_encoded_decoded:.3f}s, inference_time_encoded_decoded_noised: {inference_time_encoded_decoded_noised:.3f}s")
    
    traj_rel_actions, traj_robot_obs, traj_rgb_static = np.array(traj_rel_actions, dtype=np.float32), np.array(traj_robot_obs, dtype=np.float32), np.array(traj_rgb_static, dtype=np.uint8)
    traj_rgb_generated_goals = np.array(traj_rgb_generated_goals, dtype=np.uint8)
    traj_rgb_encoded_decoded = np.array(traj_rgb_encoded_decoded, dtype=np.uint8)
    traj_rgb_noised_encoded_decoded = np.array(traj_rgb_noised_encoded_decoded, dtype=np.uint8)

    # # traj_rgb_static
    # diffusion_model = jax_diffusion_model.DiffusionModel(50, num_samples=1)
    # traj_rgb_static2 = np.concatenate([traj_rgb_static, traj_rgb_static], axis=0)
    # traj_rgb_generated_goals = diffusion_model.generate(lang_ann, traj_rgb_static
    # traj_rgb_generated_goals2 = diffusion_model.generate(lang_ann, traj_rgb_static2)
    # traj_rgb_generated_goals = np.array(traj_rgb_generated_goals, dtype=np.uint8)

    # # Determine the output path
    write_dir = os.path.join(tfrecord_dataset_path, split, letter)
    # with open(os.path.join(write_dir, "traj" + str(ctr) + "_test.txt"), "w") as f:
    #     f.write(f"start_id: {start_id}, end_id: {end_id}\n")

    os.makedirs(write_dir, exist_ok=True)
    
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
                    "encoded_decoded": tensor_feature(traj_rgb_encoded_decoded),
                    "noised_encoded_decoded": tensor_feature(traj_rgb_noised_encoded_decoded),
                }
            )
        )
        writer.write(example.SerializeToString())

    return end_id + 1 - start_id





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
    remove=False
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

    if remove:
        shutil.rmtree(local_path)
        print(f"Removed \"{local_path}\".")
    
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
    # parser.add_argument("--start_i", type=int, help="Path to the dataset root directory.")
    # parser.add_argument("--n_idxs", type=int, help="Path to the dataset root directory.")
    # parser.add_argument("--n_gpus", type=int, help="Path to the dataset root directory.")

    parser.add_argument("--start_i", type=int, help="Path to the dataset root directory.")
    parser.add_argument("--end_i", type=int, help="Path to the dataset root directory.")
    parser.add_argument("--s3_smart_load", type=int, default=0, help="Path to the dataset root directory.")
    parser.add_argument("--docker", type=int, default=0, help="Path to the dataset root directory.")
    parser.add_argument("--n_samples", type=int, default=4, help="Path to the dataset root directory.")
    parser.add_argument("--debug", type=int, default=0, help="Path to the dataset root directory.")
    parser.add_argument("--s3_dir", type=str, help="Path to the dataset root directory.")
    return parser.parse_args()


args = parse_arguments()

if args.s3_smart_load:
    raw_dataset_path = "s3://susie-data/calvin_data/task_ABCD_D"
else:
    if args.docker:
        raw_dataset_path = "/opt/ml/input/data/calvin_data/task_ABCD_D"
    else:
        raw_dataset_path = "/home/kylehatch/Desktop/hidql/calvin/dataset/task_ABCD_D"

    # count files 
    
    files = glob(os.path.join(raw_dataset_path, "**", "*.npz"), recursive=True)
    print("len(files):", len(files))
    train_files = [file for file in files if "training" in file]
    val_files = [file for file in files if "validation" in file]
    print("len(train_files):", len(train_files))
    print("len(val_files):", len(val_files))

gpu_idx = int(os.getenv("CUDA_VISIBLE_DEVICES"))

if args.docker:
    tfrecord_dataset_path =  f"/opt/ml/code/data/calvin_data_processed/language_conditioned_with_generated_{args.n_samples}_samples/gpu_{gpu_idx}"
else:
    tfrecord_dataset_path = f"/home/kylehatch/Desktop/hidql/data/calvin_data_processed/language_conditioned_with_generated_{args.n_samples}_samples/gpu_{gpu_idx}"


print("raw_dataset_path:", raw_dataset_path)



########## Main logic ###########
if not os.path.exists(tfrecord_dataset_path):
    os.makedirs(tfrecord_dataset_path, exist_ok=True)
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


# # num_idxs = args.end_i - args.start_i
# assert args.n_idxs % args.n_gpus == 0

# n_idxs_per_gpu = args.n_idxs // args.n_gpus
# start_and_end_idxs = []
# for i in range(args.n_gpus):
#     start_idx = args.start_i + (i * n_idxs_per_gpu)
#     end_idx = start_idx + n_idxs_per_gpu
#     start_and_end_idxs.append((args.s3, i, start_idx, end_idx))



start_i, end_i = args.start_i, args.end_i


# Let's prepare the inputs
function_inputs = []

# First let's do the train data
if args.s3_smart_load:
    with smart_open(os.path.join(raw_dataset_path, "training", "lang_annotations", "auto_lang_ann.npy"), 'rb') as s3_source:
        auto_lang_ann = np.load(s3_source, allow_pickle=True)
else:
    auto_lang_ann = np.load(os.path.join(raw_dataset_path, "training", "lang_annotations", "auto_lang_ann.npy"), allow_pickle=True)

auto_lang_ann = auto_lang_ann.item()
all_language_annotations = auto_lang_ann["language"]["ann"]
idx_ranges = auto_lang_ann["info"]["indx"]

if args.s3_smart_load:
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

    function_inputs.append((args.debug, args.s3_smart_load, args.n_samples, gpu_idx, idx_range, letter, ctr, "training", all_language_annotations[i]))

# Next let's do the validation data
if args.s3_smart_load:
    with smart_open(os.path.join(raw_dataset_path, "validation", "lang_annotations", "auto_lang_ann.npy"), 'rb') as s3_source: 
        auto_lang_ann = np.load(s3_source, allow_pickle=True)
else:
    auto_lang_ann = np.load(os.path.join(raw_dataset_path, "validation", "lang_annotations", "auto_lang_ann.npy"), allow_pickle=True)

auto_lang_ann = auto_lang_ann.item()
all_language_annotations = auto_lang_ann["language"]["ann"]
idx_ranges = auto_lang_ann["info"]["indx"]

ctr = 0
for i, idx_range in enumerate(idx_ranges):
    function_inputs.append((args.debug, args.s3_smart_load, args.n_samples, gpu_idx, idx_range, "D", ctr, "validation", all_language_annotations[i]))
    ctr += 1



traj_lens = []
# print("Before process")
print("len(function_inputs):", len(function_inputs))
for i, function_input in tqdm(enumerate(function_inputs)):
    if i >= start_i and i <= end_i:
        print(f"\n[{gpu_idx}] Processing idx range {i}/{len(function_inputs)}: {function_inputs[i][4]}")
        traj_len = process_trajectory(function_input)
        traj_lens.append(traj_len)

        tf_record_files = glob(os.path.join(tfrecord_dataset_path, "**", "*.tfrecord"), recursive=True)
        train_files_tf_record_files = [file for file in tf_record_files if "training" in file]
        val_files_tf_record_files = [file for file in tf_record_files if "validation" in file]
        print("tfrecord_dataset_path:", tfrecord_dataset_path)
        print(f"Processed {len(tf_record_files)} files ({len(train_files_tf_record_files)} training {len(val_files_tf_record_files)} validation)")
        print("tf_record_files:", tf_record_files)

        sync_local_to_s3(tfrecord_dataset_path, args.s3_dir, remove=True)
    else:
        # print(f"Skipping {i}")
        pass 
# print("After process")

# print("np.sum(traj_lens):", np.sum(traj_lens))
# print("np.mean(traj_lens):", np.mean(traj_lens))
# print("len(traj_lens):", len(traj_lens))




"""
export DIFFUSION_MODEL_CHECKPOINT=/home/kylehatch/Desktop/hidql/susie-calvin-checkpoints/susie_test/test1_400smthlib_2024.02.21_06.44.06/40000/params_ema
export CUDA_VISIBLE_DEVICES=0
python3 -u language_conditioned_calvin_generated_goals.py \
--start_i 0 \
--end_i 2 \
--s3_smart_load 0 \
--docker 0 \
--n_samples 2 \
--s3_dir s3://susie-data/calvin_data_processed/language_conditioned_2_samples/ \
--debug 1





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