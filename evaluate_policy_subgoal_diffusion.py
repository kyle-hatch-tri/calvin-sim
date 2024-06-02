import argparse
from collections import Counter, defaultdict
import logging
import os
from pathlib import Path
import sys
import time
import requests
import json
import cv2
from tqdm import tqdm, trange
import tensorflow as tf
import base64
import io
# import jax_diffusion_model
# import pytorch_diffusion_model 
# import diffusion_gc_policy

from calvin_agent.evaluation import jax_diffusion_model
from calvin_agent.evaluation import pytorch_diffusion_model
from calvin_agent.evaluation import diffusion_gc_policy

import datetime
from s3_save import S3SyncCallback
import random 
from copy import deepcopy
import shutil
import pickle
from functools import partial

from jaxrl_m.data.text_processing import text_processors

from calvin_agent.evaluation.gcbc_train_config import get_config

# This is for using the locally installed repo clone when using slurm
from calvin_agent.models.calvin_base_model import CalvinBaseModel

sys.path.insert(0, Path(__file__).absolute().parents[2].as_posix())

from calvin_agent.evaluation.multistep_sequences import get_sequences, tasks
from calvin_agent.evaluation.utils import (
    collect_plan,
    count_success,
    create_tsne,
    get_default_model_and_env,
    get_env_state_for_initial_condition,
    get_log_dir,
    join_vis_lang,
    print_and_save,
)

from calvin_agent.utils.utils import get_all_checkpoints, get_checkpoints_for_epochs, get_last_checkpoint
import hydra
import numpy as np
from omegaconf import OmegaConf
# from pytorch_lightning import seed_everything # UNCOMMENT
from termcolor import colored
import torch
from tqdm.auto import tqdm
from PIL import Image
from glob import glob

from multiprocessing import Pool, Manager
from itertools import combinations

from calvin_env.envs.play_table_env import get_env
from calvin_agent.evaluation.libero_env import LiberoEnv

logger = logging.getLogger(__name__)

DEBUG = int(os.getenv("DEBUG"))

if DEBUG: 
    EP_LEN = 50
    NUM_SEQUENCES = 1

    # EP_LEN = int(os.getenv("EP_LEN"))
    # NUM_SEQUENCES = 1
else:
    EP_LEN = int(os.getenv("EP_LEN"))
    NUM_SEQUENCES = int(os.getenv("NUM_EVAL_SEQUENCES"))

print("os.getenv(\"CUDA_VISIBLE_DEVICES\"):", os.getenv("CUDA_VISIBLE_DEVICES"))
print("EP_LEN:", EP_LEN)
print("NUM_SEQUENCES:", NUM_SEQUENCES)

def make_env(env_name, dataset_path):
    if env_name == "calvin" or env_name == "calvinlcbc":
        val_folder = Path(dataset_path) / "validation"
        env = get_env(val_folder, show_gui=False)

        # insert your own env wrapper
        # env = Wrapper(env)
        return env
    elif "libero" in env_name:
        env = LiberoEnv(env_name, im_size=(200, 200))
        return env
    else:
        raise ValueError(f"Unsupported env_name: \"{env_name}\".")

def save_video(output_video_file, frames, fps=30):
     # Extract frame dimensions
    height, width, _ = frames.shape[1:]

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can also use other codecs such as 'XVID'
    # fps = 30  # Adjust the frame rate as needed

    os.makedirs(os.path.dirname(output_video_file), exist_ok=True)
    video_writer = cv2.VideoWriter(output_video_file, fourcc, fps, (width, height))

    # Write each frame to the video file
    for frame in frames:
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video_writer.write(bgr_frame)

    # Release the video writer object
    video_writer.release()

def get_oracle_goals_calvin(oracle_goals_dir, oracle_goals_type):
    oracle_goals = {}
    language_task_files = glob(os.path.join(oracle_goals_dir, "**", "language_task.txt"), recursive=True)
    for language_task_file in language_task_files:
        with open(language_task_file, "r") as f:
            subtask = f.readline().strip() 
            f.readline()
            line = f.readline().strip()
            if line.split(":")[-1].strip() == "True":
                if subtask not in oracle_goals:
                    oracle_goals_file = os.path.join(os.path.dirname(language_task_file), f"{oracle_goals_type}_goals.npy")
                    print(f"Loading oracle goals from \"{oracle_goals_file}\".")
                    goals = np.load(oracle_goals_file)
                    goals = np.concatenate([goals, goals[-1][None], goals[-1][None], goals[-1][None], goals[-1][None], goals[-1][None]])

                    oracle_goals[subtask] = goals

    return oracle_goals


def get_oracle_goals_libero(oracle_goals_dir, oracle_goals_type):
    assert oracle_goals_type == "dataset_true"

    PROTO_TYPE_SPEC = {
        "actions": tf.float64,
        "proprioceptive_states": tf.float64,
        "image_states": tf.uint8,
        # "generated_goals": tf.uint8,
        "language_annotation": tf.string,
    }

    def _decode_example(example_proto):
        # decode the example proto according to PROTO_TYPE_SPEC
        features = {
            key: tf.io.FixedLenFeature([], tf.string)
            for key in PROTO_TYPE_SPEC.keys()
        }
        parsed_features = tf.io.parse_single_example(example_proto, features)
        parsed_tensors = {}
        for key, dtype in PROTO_TYPE_SPEC.items():
            if dtype == tf.string:
                parsed_tensors[key] = parsed_features[key]
            else:
                parsed_tensors[key] = tf.io.parse_tensor(parsed_features[key], dtype)
        # restructure the dictionary into the downstream format
        return {
            "observations": {
                "image": parsed_tensors["image_states"][:-1],
                # "generated_goals": parsed_tensors["generated_goals"][:-1],
                "proprio": parsed_tensors["proprioceptive_states"][:-1],
            },
            "next_observations": {
                "image": parsed_tensors["image_states"][1:],
                # "generated_goals": parsed_tensors["generated_goals"][1:],
                "proprio": parsed_tensors["proprioceptive_states"][1:],
            },
            **({"language": parsed_tensors["language_annotation"]} if True else {}),
            "actions": parsed_tensors["actions"][:-1],
            "terminals": tf.zeros_like(parsed_tensors["actions"][:-1][:, 0:1], dtype=tf.bool)
        }
    
    def load_tfrecord_file(file_path):
        dataset = tf.data.TFRecordDataset(file_path)
        # parsed_dataset = dataset.map(parse_tfrecord)
        parsed_dataset = dataset.map(_decode_example)
        return parsed_dataset

    oracle_goals = {}

    oracle_goals["initial_dataset_images"] = {}

    # traj0_tf_record_files = glob(os.path.join(oracle_goals_dir, "val", "**", "traj0.tfrecord"), recursive=True)
    traj0_tf_record_files = glob(os.path.join(oracle_goals_dir, "val", "**", "traj*.tfrecord"), recursive=True)
    for tf_record_file in traj0_tf_record_files:
        task_suite_name = tf_record_file.split("/")[-3]
        task_name = tf_record_file.split("/")[-2]
        traj_no = int(tf_record_file.split("/")[-1].split(".")[-2][4:])

        print(f"Loading oracle goals from \"{tf_record_file}\".")

        tf_record_data = load_tfrecord_file(tf_record_file)

        for i, record in enumerate(tf_record_data):
            pass 

        assert i == 0

        K = 20
        
        images = np.array(record["observations"]["image"])
        goal_images_idxs = np.arange(0, images.shape[0], K)[1:]
        goal_images_idxs = np.concatenate([goal_images_idxs, np.array([-1, -1, -1, -1, -1])], axis=0)
        goal_images = images[goal_images_idxs]
        # assert task_name not in oracle_goals
        # oracle_goals[task_name] = goal_images
        # oracle_goals["initial_dataset_images"][task_name] = images[0]

        if task_name not in oracle_goals:
            oracle_goals[task_name] = {}

        assert traj_no not in oracle_goals[task_name]
        oracle_goals[task_name][traj_no] = goal_images

        if task_name not in oracle_goals["initial_dataset_images"]:
            oracle_goals["initial_dataset_images"][task_name] = {}

        assert traj_no not in oracle_goals["initial_dataset_images"][task_name]
        oracle_goals["initial_dataset_images"][task_name][traj_no] = images[0]

    return oracle_goals


def encode_image(
    # image_filepath: str,
    image,
    image_format: str = "PNG",
    max_side_length: int = 512,
    debug: bool = False,
):
    image = Image.fromarray(image)
    # Calculate the new size, keeping aspect ratio.
    ratio = min(max_side_length / image.width, max_side_length / image.height)
    new_size = (int(image.width * ratio), int(image.height * ratio))
    resized_image = image.resize(new_size, Image.LANCZOS)

    # if debug:
        # debug_dir = Path(
        #     "/home/blakewulfe/data/datasets/r2d2/visualization/attempt_2/data/debug"
        # )
        # debug_filepath = debug_dir / "_".join(image_filepath.parts[-6:])
        # resized_image.save(debug_filepath, format=image_format)

    buffered = io.BytesIO()
    resized_image.save(buffered, format=image_format)
    image_bytes = buffered.getvalue()


    image_base64 = base64.b64encode(image_bytes)
    image_base64_str = image_base64.decode("utf-8")
    return image_base64_str



def high_level_vf_filter(vf_agent, language_goal, rgb_obs, goal_images):
    v = vf_agent.value_function_ranking_lcgc(rgb_obs, goal_images, language_goal)
    sorted_idxs = np.argsort(v)[::-1]
    ordered_goal_images = goal_images[sorted_idxs]
    ordered_vs = v[sorted_idxs]

    best_goal_idx = sorted_idxs[0]

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_color = (0, 255, 0) 
    line_type = 2  # Line thickness

    frame = []
    for i in range(ordered_goal_images.shape[0]):
        img = ordered_goal_images[i]
        orig_idx = np.arange(goal_images.shape[0])[i]
        img = cv2.putText(img, f'[{i}] v: {ordered_vs[i]:.3f}', (25, 20), font, font_scale, font_color, line_type)
        frame.append(img)

    assert len(frame) % 4 == 0, f"len(frame): {len(frame)}"
    frame_rows = []
    for row_idx in range(len(frame) // 4):
        start = row_idx * 4
        end = start + 4
        frame_row = np.concatenate([rgb_obs] + frame[start:end], axis=1)
        frame_rows.append(frame_row)

    query_frame = np.concatenate(frame_rows, axis=0)

    mode = "okay"
    return best_goal_idx, {"query_frame":query_frame}, mode


def low_level_vf_filter(vf_agent, language_goal, rgb_obs, goal_images):
    v = vf_agent.value_function_ranking(rgb_obs, goal_images)
    sorted_idxs = np.argsort(v)#[::-1] # have lowest v one first 
    ordered_goal_images = goal_images[sorted_idxs]
    ordered_vs = v[sorted_idxs]

    best_goal_idx = sorted_idxs[0]

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_color = (0, 255, 0) 
    line_type = 2  # Line thickness

    frame = []
    for i in range(ordered_goal_images.shape[0]):
        img = ordered_goal_images[i]
        orig_idx = np.arange(goal_images.shape[0])[i]
        img = cv2.putText(img, f'[{i}] v: {ordered_vs[i]:.3f}', (25, 20), font, font_scale, font_color, line_type)
        frame.append(img)

    assert len(frame) % 4 == 0, f"len(frame): {len(frame)}"
    frame_rows = []
    for row_idx in range(len(frame) // 4):
        start = row_idx * 4
        end = start + 4
        frame_row = np.concatenate([rgb_obs] + frame[start:end], axis=1)
        frame_rows.append(frame_row)

    query_frame = np.concatenate(frame_rows, axis=0)

    mode = "okay"
    return best_goal_idx, {"query_frame":query_frame}, mode

def hooman_in_the_loop_filter(language_goal, rgb_obs, goal_images, dummy=False):
    rgb_obs = cv2.resize(rgb_obs, (512, 512))
    new_goal_images = []
    for goal_img in goal_images:
        goal_img = cv2.resize(goal_img, (512, 512))
        new_goal_images.append(goal_img)
    goal_images = np.stack(new_goal_images, axis=0)

    

    frame = []

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (0, 255, 0)  # White color in BGR
    line_type = 2  # Line thickness

    for i in range(goal_images.shape[0]):
        img = goal_images[i]
        img = cv2.putText(img, f'[{i}]', (25, 40), font, font_scale, font_color, line_type)
        frame.append(img)

    assert len(frame) % 4 == 0, f"len(frame): {len(frame)}"
    frame_rows = []
    for row_idx in range(len(frame) // 4):
        start = row_idx * 4
        end = start + 4
        frame_row = np.concatenate([rgb_obs] + frame[start:end], axis=1)
        frame_rows.append(frame_row)

    query_frame = np.concatenate(frame_rows, axis=0)

    for _ in range(5):
        try:
            if dummy or language_goal.strip() == "grasp and lift the red block":
                best_goal_idx = 0
                mode = "okay"
            else:
                os.makedirs("candidate_goal_images", exist_ok=True)
                # cv2.imwrite(f"candidate_goal_images.png", query_frame[..., ::-1])
                cv2.imwrite(os.path.join("candidate_goal_images", f"{language_goal.strip().replace(' ', '_')}.png"), query_frame[..., ::-1])

                print(f"\nThe language instruction is, \"{language_goal}\".")
                print(f"Which goal image do you want to select?")
                best_goal_idx = input(f"> ").strip()

                if best_goal_idx == "q":
                    best_goal_idx = 0 
                    mode = "quit"
                elif best_goal_idx == "e":
                    best_goal_idx = 0 
                    mode = "end_episode"
                else:
                    best_goal_idx = int(best_goal_idx)
                    print(f"You entered goal image idx: {best_goal_idx}")
                    mode = "okay"

            assert best_goal_idx in list(range(goal_images.shape[0])), f"choice: {best_goal_idx}, list(range(goal_images.shape[0])): {list(range(goal_images.shape[0]))}"

            break 
        except:
            print(f"Invalid input: {best_goal_idx}")

    frame = []
    for i in range(goal_images.shape[0]):
        img = goal_images[i]
        img = cv2.putText(img, f'[{i}]', (25, 40), font, font_scale, font_color, line_type)

        if i == best_goal_idx:
            img = cv2.rectangle(img, (0, 0), (img.shape[1] - 1, img.shape[0] - 1), font_color, 10)

        frame.append(img)

    assert len(frame) % 4 == 0, f"len(frame): {len(frame)}"
    frame_rows = []
    for row_idx in range(len(frame) // 4):
        start = row_idx * 4
        end = start + 4
        frame_row = np.concatenate([rgb_obs] + frame[start:end], axis=1)
        frame_rows.append(frame_row)

    query_frame = np.concatenate(frame_rows, axis=0)

    return best_goal_idx, {"query_frame":query_frame}, mode




def chat_gpt_goal_pairwise_filtering_fn(language_goal, rgb_obs, goal_images):

    rgb_obs = cv2.resize(rgb_obs, (512, 512))

    new_goal_images = []
    for goal_img in goal_images:
        goal_img = cv2.resize(goal_img, (512, 512))
        new_goal_images.append(goal_img)
    goal_images = np.stack(new_goal_images, axis=0)

    win_counts, query_infos, total_query_time = ask_chat_gpt5(rgb_obs, goal_images, language_goal) 

    

    best_goal_idx = 0
    max_win_count = 0
    for goal_idx, count in win_counts.items():
        if count > max_win_count:
            max_win_count = count 
            best_goal_idx = goal_idx

    frame = []

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (0, 255, 0)  # White color in BGR
    line_type = 2  # Line thickness

    for i in range(goal_images.shape[0]):
        img = goal_images[i]

        img = cv2.putText(img, f'[{i}]    win_count: {win_counts[i]}', (25, 40), font, font_scale, font_color, line_type)

        if win_counts[i] == max_win_count:
            img = cv2.rectangle(img, (0, 0), (img.shape[1] - 1, img.shape[0] - 1), font_color, 10)

        if i == best_goal_idx:
            img = cv2.rectangle(img, (10, 10), (img.shape[1] - 11, img.shape[0] - 11), (255, 0, 255), 10)

        frame.append(img)

    assert len(frame) % 4 == 0, f"len(frame): {len(frame)}"
    frame_rows = []
    for row_idx in range(len(frame) // 4):
        start = row_idx * 4
        end = start + 4
        frame_row = np.concatenate([rgb_obs] + frame[start:end], axis=1)
        frame_rows.append(frame_row)

    query_frame = np.concatenate(frame_rows, axis=0)

    images_info = {"query_infos":query_infos, "win_counts":win_counts, "total_query_time":total_query_time, "query_frame":query_frame}


    return best_goal_idx, images_info



def ask_chat_gpt5(rgb_image_obs, goal_images, language_goal):

    base64_rgb_obs = encode_image(rgb_image_obs)

    prompt1 = f"We have a computer simulation of a robot tabletop manipulation environment. \
In this simulated environment, we are using a machine learning algorithm to control the robot so that \
it completes the following task: \"{language_goal}.\" The robot is partway through completing the task. We have an image of the environment taken from a third person camera. We will call this image the image observation. \
We also have a second image. This image is generated by a neural network, and shows the robot what it should do a few seconds into the future in order to come closer to completing the task \"{language_goal}\". This image is called the goal image. The robot will \
compare the image observation and the goal image, and figure out how it needs to move its gripper and what objects it needs to move or manipulate in order to make the environment match what is shown in the goal image. \
Since the goal image shows what the environment should look like a few seconds into the future, it will have some differences from the image observation. These differences will primarily be the location of the robot gripper and position of the robot arm, as well \
as the locations and positions of any objects that the robot is directly manipulating. However, sometimes the neural network will make errors when generating the goal image, which will cause the robot to become confused. These errors are when the goal image has harmful inconsistencies with the image observation. \
The following is a list of the most common types of harmful inconsistencies to look out for: \
1. Hallucinated objects. This is when there are objects that appear in the goal image but that do not appear in the image observation. Note, however, that the position of the robot gripper and arm in the image \
observation may be covering up objects that are behind them. So, if the position of the robot arm or gripper changes between the goal image and the image observation, there may be objects that were previously \
covered by the the robot arm or gripper in the image observation that are now visible in the goal image. These should not be considered hallucinations. Also note that not all hallucinations will be relevant/harmful. Hallucinated objects that are in an area of the image that the robot is not interacting with, or \
hallucinated objects that are not relevant to completing the task \"{language_goal}\" are unlikely to confuse the robot and so should not be considered relevant inconsistencies. \
2. Changes in object shape or color. This is when one or more objects in the goal image don't match their original shapes or colors in the image observation. Note, however, that slight changes in shape or color are okay and should not be considered inconsistencies \
(such as a shift to a slightly different shade of the same color or a slight shape elongations or distortions), but major changes in shape or color (such as an object becoming a completely different shape or becoming a completely different color) are not okay and should be considered inconsistencies. \
Note, however, that in both of these cases, these inconsistencies are only likely to be harmful if they involve objects that are directly related to completing the task \"{language_goal}\" or involve objects that the robot gripper is attempting to manipulate. Inconsistencies in the background of the image \
or in parts of the image that the robot gripper is not interacting with are most likely not harmful and should not be taken into consideration. Similarly, inconsistencies in the image not related to completing the task \"{language_goal}\", or not related to the objects that the robot gripper is attempting to manipulate \
should not be taken into consideration. Now, here is the image observation of the environment taken from the third person camera and here are two the goal images generated by the neural network. The first image is the image observation, and the second two images are the goal images. \
Please take special care not to confuse the ordering of the two goal images. \
Which of the two goal images is least likely to contain relevant/harmful inconsistencies with the image observation that would cause the robot to become confused? Please include \"First\" or \"Second\" in your answer (or \"Same\" if there is no significant difference between level of relevant/harmful inconsistencies). \
Please take special care not to confuse the ordering of the two goal images."


# The first image is the image observation, and the second image contains the two goal images, \
# stacked side-by-side. \
# Please take special care not to confuse the ordering of the two goal images. \
# Which of the two goal images is least likely to contain relevant/harmful inconsistencies with the image observation that would cause the robot to become confused? Is it the goal image on the left side, or the goal image on the right side? Please include \"Left\" or \"Right\" in your answer. \
# Please take special care not to confuse the ordering of the two goal images."

    prompt2 = f"Now please answer in just one word: \"First\", \"Second\", or \"Same\"."

    n_attempts = 10
    for attempt_idx in range(n_attempts):
        try:
            manager = Manager()

            # answers = {}
            win_counts = manager.dict()
            infos = manager.dict()
            function_inputs = []

            idx_pairs = list(combinations(range(goal_images.shape[0]), 2))

            # initialize the dictionaries
            for i in range(goal_images.shape[0]):
                win_counts[i] = 0
            for i, j in idx_pairs:
                infos[i, j] = manager.dict()

            for i, j in idx_pairs:
                goal_image1 = goal_images[i]
                goal_image2 = goal_images[j]

                
                base64_goal_image1 = encode_image(goal_image1)
                base64_goal_image2 = encode_image(goal_image2)

                function_inputs.append((win_counts, infos, i, j, base64_rgb_obs, base64_goal_image1, base64_goal_image2, prompt1, prompt2))


            t0 = time.time()
            with Pool(len(function_inputs)) as pool: # We have one process per input because we are io bound, not cpu bound
                pool.starmap(compare_pair_of_goal_images, function_inputs)
            t1 = time.time()

            total_query_time = t1 - t0

            win_counts = dict(win_counts)
            infos = {key:dict(val) for key, val in infos.items()}

            return win_counts, infos, total_query_time
        except Exception as e:
            print(f"Error querying chat GPT pool on attempt {attempt_idx + 1}: {e}")

            if attempt_idx in [5, 8, n_attempts - 1]:
                time.sleep(60 * 10)
            else:
                time.sleep(60)
    raise ValueError(f"Chat GPT pool thing errored out after {attempt_idx + 1} attempts.")
    


def compare_pair_of_goal_images(win_counts, infos, i, j, base64_rgb_obs, base64_goal_image1, base64_goal_image2, prompt1, prompt2):
    # global win_counts, infos
    # i, j, base64_rgb_obs, base64_goal_image1, base64_goal_image2, prompt1, prompt2 = function_input
    content = [
        {"type": "text","text": prompt1},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_rgb_obs}", "detail": "high"}},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_goal_image1}", "detail": "high"}},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_goal_image2}", "detail": "high"}},
        # {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_goal_image}", "detail": "high"}},

    ]

    messages = [{"role": "user", "content":content,}] 

    t0 = time.time()
    answer1 = query_chat_gpt2(messages, n_attempts=1)   
    t1 = time.time()

    messages.append({"role":"assistant", "content":[{"type": "text","text": answer1},]})
    messages.append({"role":"user", "content":[{"type": "text","text": prompt2},]})

    t2 = time.time()
    answer2 = query_chat_gpt2(messages, n_attempts=1)   
    t3 = time.time()

    answer2 = answer2.strip()
    answer2 = answer2.strip(".\n\"\',")

    if answer2.lower() == "first":
        win_counts[i] += 1 
    elif answer2.lower() == "second":
        win_counts[j] += 1
    elif answer2.lower() == "same": 
        win_counts[i] += 1  # assumption here is that they are more likely to be "same" if they are good, not if they are both bad
        win_counts[j] += 1
    else:
        raise ValueError(f"answer2.lower() not in [first, second, same]. answer2: \"{answer2.lower()}\".")

    answer = f"> {answer1} \n\n> {answer2}"

    

    # answers[(i, j)] = answer
    infos[(i, j)]["answer"] = answer
    infos[(i, j)]["query1 time"] = t1 - t0 
    infos[(i, j)]["query2 time"] = t3 - t2



def query_chat_gpt2(messages, dryrun=False, n_attempts=10):
    # api_key = "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"
    # api_key = "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"
    # api_key = "sk-proj-VtrcFJQBB1W4MEGrdibfT3BlbkFJoSpSwukCtDlxcADLAGLc"
    # api_key = "sk-proj-mr8ZIOp3MVVEVbCNYgFFT3BlbkFJCdSIulmzOATndgxhFDuU" # new one from Blake
    api_key = "sk-proj-57WuzJ0j1I4hEY61wDSXT3BlbkFJPmgnOBqR1jThDq5d8czD"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    payload = {
        # "model": "gpt-4-vision-preview",
        "model": "gpt-4-turbo",
        # "messages": [
        #     {
        #         "role": "user",
        #         "content":content,
        #     }
        # ],
        "messages":messages,
        # "max_tokens": 300,
        "max_tokens": 1200,
    }

    if dryrun:
        return "Did not run due to dryrun=True"
    

    # for i in range(n_attempts):
    #     response = requests.post(
    #         "https://api.openai.com/v1/chat/completions",
    #         headers=headers,
    #         json=payload,
    #     )
    #     answer = None
    #     try:
    #         answer = response.json()["choices"][0]["message"]["content"]
    #     except Exception as e:
    #         print(f"Exception parsing response: {e}")
    #         answer = "Error"

    #     if answer != "Error":
    #         return answer
        
    #     time.sleep(30)
    #     print(f"chat GPT query attempt {i}...")
        
    # raise ValueError(f"chat GPT errored out after {n_attempts} attempts.")
    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers=headers,
        json=payload,
    )
    answer = None
    try:
        answer = response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"Exception parsing response.json(): {response.json()}, error: {e}")
        answer = "Error"

    return answer
   

def query_chat_gpt(language_goal, rgb_obs, goal_images, dryrun=False):
    # api_key = "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"
    api_key = "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"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    prompt1 = f"We have a computer simulation of a robot tabletop manipulation environment. \
In this simulated environment, we are using a machine learning algorithm to control the robot so that \
it completes the following task: \"{language_goal}.\" The machine learning algorithm works as follows: \
at each time-steps, an image of the environment is taken from a third person camera. This image is \
given to a generative neural network that generates a goal image of what the robot should accomplish \
twenty time-steps into the future in order to come closer to completing the task, \"{language_goal}.\" \
A low level controller then tries to control the robot to reach the generated goal image. After twenty time-steps, \
a new image is taken from the camera, and the generative neural network produces a new goal image for the low level \
controller to reach. This is repeated until the task, \"{language_goal}\" is completed. The robot is partway through \
completing the task. Here is the image from the third person camera showing the current state of the simulated environment."

    base64_rgb_obs = encode_image(rgb_obs)

    content = []
    content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_rgb_obs}"}})
    content.append({"type": "text","text": prompt1})
    

    image_numbers_list_str = ""

    for i, goal_image in enumerate(goal_images):
        base64_goal_image = encode_image(goal_image)
        
        if i < goal_images.shape[0] - 1:
            image_numbers_list_str += f"image {i + 1}, "
        else:
            image_numbers_list_str += f"or image {i + 1}"


        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_goal_image}"}})

    prompt2 = f"Here are several candidate goal images generated by the neural network showing possible goals of what the robot \
should do over the next twenty time-steps in order to get closer to completing the task, \"{language_goal}.\" However, some \
of the generated goal images are better than the others. For example, sometimes the neural network may generate images that have \
hallucinated objects that do not actually exist in the environment. Sometimes the neural network may generate images that show it \
completing the task incorrectly, or that show it completing a different task. Or sometimes the neural network may generate goal \
images that are sub-optimal for other reasons. Of these generated goal images, which one is most likely to lead the robot closer \
to completing the task? Is it {image_numbers_list_str}? Please give an \
answer that is simply either: {image_numbers_list_str}. Or, if all of the images appear identical, please simply answer: identical."


    content.append({"type": "text","text": prompt2})

    payload = {
        "model": "gpt-4-vision-preview",
        "messages": [
            {
                "role": "user",
                "content": content,
            }
        ],
        "max_tokens": 300,
    }

    if dryrun:
        return "Did not run due to dryrun=True"

    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers=headers,
        json=payload,
    )
    classification = None
    try:
        classification = response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"Exception parsing response: {e}")
        classification = "Error"
    
    return classification


def chat_gpt_goal_filtering_fn(language_goal, rgb_obs, goal_images):

    possible_images = [f"image {i + 1}" for i in range(goal_images.shape[0])]



    chat_gpt_error = True 
    for i in range(20):
        chat_gpt_answer = query_chat_gpt(language_goal, rgb_obs, goal_images, dryrun=False)
        chat_gpt_answer = chat_gpt_answer.lower()
        chat_gpt_answer = chat_gpt_answer.strip()
        chat_gpt_answer = chat_gpt_answer.strip(".")

        if chat_gpt_answer != "error" and ("identical" in chat_gpt_answer or chat_gpt_answer in possible_images):
            chat_gpt_error = False 
            break 
        
        print(f"Error {i}, chat_gpt_answer: \"{chat_gpt_answer}\".")


    if chat_gpt_error:
        raise ValueError(f"Chat GPT error after {20} tries.")


    
    if "identical" in chat_gpt_answer:
        # chat_gpt_answer == "identical"
        goal_idx = 0 
    else:
        assert chat_gpt_answer in possible_images, f"chat_gpt_answer: \"{chat_gpt_answer}\", possible_images: {possible_images}"
        goal_idx = possible_images.index(chat_gpt_answer)


    # info = {i:f"image_{i}_true" if i == goal_idx else f"image_{i}_false" for i in range(goal_images.shape[0])}
    images_info = np.concatenate(goal_images, axis=1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    images_info = {"unfiltered_goal_images_frames":cv2.putText(images_info, f"{chat_gpt_answer}", (50, 50), font, color=(0, 255, 0), fontScale=1, thickness=2),
                   "unfiltered_goal_images":goal_images,
                   "chat_gpt_answer":chat_gpt_answer}


    

    return goal_idx, images_info


def chat_gpt_dummy_goal_filtering_fn(language_goal, rgb_obs, goal_images):
    goal_idx = 0
    images_info = {"unfiltered_goal_images":goal_images, "rgb_obs_at_filtering":rgb_obs,}
    return goal_idx, images_info

class CustomModel(CalvinBaseModel):
    def __init__(self, agent_config, env_name, agent_type, single_task, oracle_goals_dir, oracle_goals_type, use_temporal_ensembling, diffusion_model_checkpoint_path, gc_policy_checkpoint_path, gc_vf_checkpoint_path, num_denoising_steps, num_samples=1, prompt_w=7.5, context_w=1.5, filtering_method=None, flat_policy=False, diffusion_model_framework="jax", vf_agent_config=None, vf_agent_type=None):
        # Initialize diffusion model

        self.num_samples = num_samples
        self.filtering_method = filtering_method
        self.flat_policy = flat_policy
        self.prompt_w = prompt_w
        self.context_w = context_w

        assert use_temporal_ensembling, f"use_temporal_ensembling: {use_temporal_ensembling}"

        self.oracle_goals_type = oracle_goals_type
    
        if not self.flat_policy:
            if diffusion_model_framework == "jax":
                self.diffusion_model = jax_diffusion_model.DiffusionModel(num_denoising_steps, num_samples=self.num_samples)
            elif diffusion_model_framework == "pytorch":
                self.diffusion_model = pytorch_diffusion_model.PytorchDiffusionModel()
            else:
                raise ValueError(f"Unsupported diffusion model framework: \"{diffusion_model_framework}\".")

        print("agent_type:", agent_type)
        print(f"Initializing policy from \"{os.getenv('GC_POLICY_CHECKPOINT')}\"...")

        if self.flat_policy:
            text_processor = text_processors["muse_embedding"]()
        else:
            text_processor = None

        
        # normalize_actions = "noactnorm" in os.getenv("GC_POLICY_CHECKPOINT") or "nam" in os.getenv("GC_POLICY_CHECKPOINT")
        normalize_actions = False ### TODO better way of handling this. Also, the above line of how it used to be is opposite of what it should be, but probably doesn't matter, since I don't think the value of normalize_actions matters for eval
        self.gc_policy = diffusion_gc_policy.GCPolicy(agent_config, env_name, agent_type, os.getenv("GC_POLICY_CHECKPOINT"), normalize_actions=normalize_actions, use_temporal_ensembling=use_temporal_ensembling, text_processor=text_processor)
        
        if self.filtering_method == "high_level_vf":
            assert vf_agent_config is not None
            assert vf_agent_type is not None
            self.vf_agent = diffusion_gc_policy.GCPolicy(vf_agent_config, env_name, vf_agent_type, os.getenv("HIGH_LEVEL_VF_CHECKPOINT"), normalize_actions=False, use_temporal_ensembling=False, text_processor=text_processors["muse_embedding"]())
        elif self.filtering_method == "low_level_vf":
            assert vf_agent_config is not None
            assert vf_agent_type is not None
            self.vf_agent = diffusion_gc_policy.GCPolicy(vf_agent_config, env_name, vf_agent_type, os.getenv("LOW_LEVEL_VF_CHECKPOINT"), normalize_actions=False, use_temporal_ensembling=False, text_processor=None)

        timestamp = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
        
        self.log_dir = "results"
        if DEBUG:
            self.log_dir = os.path.join(self.log_dir, "el_trasho")

        if single_task:
            self.log_dir = os.path.join(self.log_dir, "single_task")

        self.oracle_goals = None
        if oracle_goals_dir is not None:
            # conf_dir = Path(__file__).absolute().parents[0] / "calvin_models" / "conf"
            # val_annotations = OmegaConf.load(conf_dir / "annotations/new_playtable_validation.yaml")

            if "calvin" in env_name:
                # self.oracle_goals = {}
                # language_task_files = glob(os.path.join(oracle_goals_dir, "**", "language_task.txt"), recursive=True)
                # for language_task_file in language_task_files:
                #     with open(language_task_file, "r") as f:
                #         subtask = f.readline().strip() 
                #         f.readline()
                #         line = f.readline().strip()
                #         if line.split(":")[-1].strip() == "True":
                #             if subtask not in self.oracle_goals:
                #                 oracle_goals_file = os.path.join(os.path.dirname(language_task_file), f"{self.oracle_goals_type}_goals.npy")
                #                 print(f"Loading oracle goals from \"{oracle_goals_file}\".")
                #                 goals = np.load(oracle_goals_file)
                #                 goals = np.concatenate([goals, goals[-1][None], goals[-1][None], goals[-1][None], goals[-1][None], goals[-1][None]])

                #                 self.oracle_goals[subtask] = goals
                self.oracle_goals = get_oracle_goals_calvin(oracle_goals_dir, self.oracle_goals_type)
            elif "libero" in env_name:
                self.oracle_goals = get_oracle_goals_libero(oracle_goals_dir, self.oracle_goals_type)

        if self.num_samples > 1:
            assert self.filtering_method is not None
        else:
            assert self.num_samples == 1, f"self.num_samples: {self.num_samples}"


        self.log_dir = os.path.join(self.log_dir, env_name, *diffusion_model_checkpoint_path.strip("/").split("/")[-3:-1], *gc_policy_checkpoint_path.strip("/").split("/")[-6:], timestamp)

        print(f"Logging to \"{self.log_dir}\"...")
        os.makedirs(self.log_dir, exist_ok=True)
        self.episode_counter = None
        self.language_task = None
        self.sub_task = None
        self.obs_image_seq = None
        self.goal_image_seq = None
        # self.vranked_goal_images_seq = None
        self.vranking_save_freq = 1
        self.action_seq = None
        self.combined_images = None

        # Other necessary variables for running rollouts
        self.goal_image = None
        self.subgoal_counter = 0
        # self.subgoal_max = 20
        self.subgoal_max = int(os.getenv("SUBGOAL_MAX"))
        self.pbar = None

        print("self.subgoal_max:", self.subgoal_max)

        shutil.copy2("eval_susie.sh", os.path.join(self.log_dir, "eval_susie.sh"))

    def save_info(self, success, initial_images=None):
        episode_log_dir = os.path.join(self.log_dir, "ep" + str(self.episode_counter))
        if not os.path.exists(episode_log_dir):
            os.makedirs(episode_log_dir)

        # Log the language task
        with open(os.path.join(episode_log_dir, "language_task.txt"), "w") as f:
            f.write(self.subtask + "\n")
            f.write(self.language_task + "\n")
            f.write(f"success: {success}\n")
        
        # # Log the observation video
        # size = (200, 200)
        # out = cv2.VideoWriter(os.path.join(episode_log_dir, "trajectory.mp4"), cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
        # for i in range(len(self.obs_image_seq)):
        #     rgb_img = cv2.cvtColor(self.obs_image_seq[i], cv2.COLOR_RGB2BGR)
        #     out.write(rgb_img)
        # out.release()

        if not self.flat_policy:
        #     # Log the goals video
        #     size = (200, 200)
        #     out = cv2.VideoWriter(os.path.join(episode_log_dir, "goals.mp4"), cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
        #     for i in range(len(self.goal_image_seq)):
        #         rgb_img = cv2.cvtColor(self.goal_image_seq[i], cv2.COLOR_RGB2BGR)
        #         out.write(rgb_img)
        #     out.release()

            # Log the combined image
            size = (400, 200)
            out = cv2.VideoWriter(os.path.join(episode_log_dir, "combined.mp4"), cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
            for i in range(len(self.combined_images)):
                rgb_img = cv2.cvtColor(self.combined_images[i], cv2.COLOR_RGB2BGR)
                out.write(rgb_img)
            out.release()
        
        # # Log the actions
        # np.save(os.path.join(episode_log_dir, "actions.npy"), np.array(self.action_seq))

        # np.save(os.path.join(episode_log_dir, "retrospective_true_goals.npy"), np.array(self.retrospective_true_goals)) 

        # if not self.flat_policy:
        #     np.save(os.path.join(episode_log_dir, "generated_goals.npy"), np.array(self.generated_goals))


        # if not self.flat_policy:
        #     size = (400, 200)
        #     out = cv2.VideoWriter(os.path.join(episode_log_dir, "generated_vs_retrospective_true_goals.mp4"), cv2.VideoWriter_fourcc(*'DIVX'), 2, size)
        #     for i in range(len(self.generated_goals)):
        #         img = np.concatenate([self.generated_goals[i], self.retrospective_true_goals[i]], axis=1)
        #         rgb_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        #         out.write(rgb_img)
        #     out.release()

        if initial_images is not None:
            cv2.imwrite(os.path.join(episode_log_dir, "initial_images.png"), initial_images[..., ::-1])

        if "unfiltered_goal_images_frames" in self.goal_images_info:
            size = (self.goal_images_info["unfiltered_goal_images_frames"][0].shape[1], 200)
            out = cv2.VideoWriter(os.path.join(episode_log_dir, "unfiltered_goal_images.mp4"), cv2.VideoWriter_fourcc(*'DIVX'), 2, size)
            for i in range(len(self.goal_images_info["unfiltered_goal_images_frames"])):
                img = self.goal_images_info["unfiltered_goal_images_frames"][i]
                rgb_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                out.write(rgb_img)
            out.release()

        if "unfiltered_goal_images" in self.goal_images_info:
            for t in range(len(self.goal_images_info["unfiltered_goal_images"])):
                goal_images = self.goal_images_info["unfiltered_goal_images"][t]
                save_dir = os.path.join(episode_log_dir, "unfiltered_goal_images", f"timestep_{t * self.subgoal_max:03}")
                os.makedirs(save_dir, exist_ok=True)
                for i, goal_image in enumerate(goal_images):
                    cv2.imwrite(os.path.join(save_dir, f"goal_image_{i:03}.png"), goal_image[..., ::-1])


                rgb_obs = self.goal_images_info["rgb_obs_at_filtering"][t]
                cv2.imwrite(os.path.join(save_dir, f"rgb_obs.png"), rgb_obs[..., ::-1])

        if "chat_gpt_answer" in self.goal_images_info:
            with open(os.path.join(episode_log_dir, "chat_gpt_answers.txt"), "w") as f:
                for i in range(len(self.goal_images_info["chat_gpt_answer"])):
                    chat_gpt_answer = self.goal_images_info["chat_gpt_answer"][i]
                    f.write(f"[{i}] {chat_gpt_answer}\n")

        if "query_infos" in self.goal_images_info:
            save_dir = os.path.join(episode_log_dir, "chat_gpt_answers")
            os.makedirs(save_dir, exist_ok=True)
            for i in range(len(self.goal_images_info["query_infos"])):
                query_info = self.goal_images_info["query_infos"][i]
                win_counts = self.goal_images_info["win_counts"][i]
                total_query_time = self.goal_images_info["total_query_time"][i]

                with open(os.path.join(save_dir, f"chat_gpt_answers_{i * self.subgoal_max:03}.txt"), "w") as f:
                    f.write(f"WIN COUNTS\n")
                    for i, count in win_counts.items():
                        f.write(f"\tgoal image {i}: {count}\n")

                    f.write(f"\\nTotal query time: {total_query_time:.3f}s.\n\n")

                    f.write("\n\n\n")
                    

                    for (i, j), info in query_info.items():
                        f.write("=" * 30 + f" goal images ({i}, {j}) " + "=" * 30 + "\n")
                        f.write(f"timestep: {i * self.subgoal_max:03}\n")
                        # f.write(f"language goal: \"{language_goal}\".\n")
                        f.write(f"Query time: {info['query1 time']:.3f}s, {info['query2 time']:.3f}s.\n")
                        answer = info["answer"]
                        f.write(f"{answer}\n\n\n")

        if "query_frame" in self.goal_images_info:
            save_video(os.path.join(episode_log_dir, "query_frames.mp4"), np.stack(self.goal_images_info["query_frame"], axis=0), fps=10)

        # if "query_frame" in self.goal_images_info:


                # query_frame = self.goal_images_info["query_frame"][i]

        # {"query_infos":query_infos, "total_query_time":total_query_time, "query_frame":query_frame}



    def reset(self):
        if self.episode_counter is None: # this is the first time reset has been called
            self.episode_counter = 0
            self.obs_image_seq = []
            self.goal_image_seq = []
            # self.vranked_goal_images_seq = []
            self.action_seq = []
            self.combined_images = []

            self.generated_goals = []
            self.retrospective_true_goals = []
            self.hardcoded_goal_idx_counter = 0

            self.goal_images_info = defaultdict(list)
        else:
            # Update/reset all the variables
            self.episode_counter += 1
            self.obs_image_seq = []
            self.goal_image_seq = []
            # self.vranked_goal_images_seq = []
            self.action_seq = []
            self.goal_image = None
            self.combined_images = []
            self.subgoal_counter = 0

            self.generated_goals = []
            self.retrospective_true_goals = []
            self.hardcoded_goal_idx_counter = 0

            self.goal_images_info = defaultdict(list)

            # Reset the GC policy
            self.gc_policy.reset()

        # tqdm progress bar
        if self.pbar is not None:
            self.pbar.close()
        self.pbar = tqdm(total=EP_LEN)

    def goal_filtering_fn(self, goal, rgb_obs, goal_images):
        if self.filtering_method == "chat_gpt":
            return chat_gpt_goal_filtering_fn(goal, rgb_obs, goal_images)
        elif self.filtering_method == "chat_gpt_pairwise":
            return chat_gpt_goal_pairwise_filtering_fn(goal, rgb_obs, goal_images)
        elif self.filtering_method == "chat_gpt_dummy":
            return chat_gpt_dummy_goal_filtering_fn(goal, rgb_obs, goal_images)
        elif self.filtering_method == "human":
            return hooman_in_the_loop_filter(goal, rgb_obs, goal_images)
        elif self.filtering_method == "human_dummy":
            return hooman_in_the_loop_filter(goal, rgb_obs, goal_images, dummy=True)
        elif self.filtering_method == "high_level_vf":
            return high_level_vf_filter(self.vf_agent, goal, rgb_obs, goal_images)
        elif self.filtering_method == "low_level_vf":
            return low_level_vf_filter(self.vf_agent, goal, rgb_obs, goal_images)
        else:
            raise NotImplementedError(f"Goal images filtering method \"{self.filtering_method}\" is not implemented.")


    def step(self, obs, goal, subtask, ep_idx=0):
        """
        Args:
            obs: environment observations
            goal: embedded language goal
        Returns:
            action: predicted action
        """
        rgb_obs = obs["rgb_obs"]["rgb_static"]
        self.language_task = goal
        self.subtask = subtask

        mode = "okay"

        if not self.flat_policy:
 
            # If we need to, generate a new goal image
            if self.goal_image is None or self.subgoal_counter >= self.subgoal_max:
                
                # self.goal_image = self.diffusion_model.generate(self.language_task, rgb_obs)

                if self.oracle_goals is None:
                    t0 = time.time()
                    goal_images = self.diffusion_model.generate(self.language_task, rgb_obs, prompt_w=self.prompt_w, context_w=self.context_w)
                    t1 = time.time()
                    # print(f"t1 - t0: {t1 - t0:.3f}") 
                else:
                    if self.hardcoded_goal_idx_counter >= self.oracle_goals[subtask][ep_idx].shape[0]:
                        return -1
                
                    # goal_images = self.hard_coded_goal_images[self.hardcoded_goal_idx_counter][None]
                    goal_images = self.oracle_goals[subtask][ep_idx][self.hardcoded_goal_idx_counter][None]

                    print(f"self.hardcoded_goal_idx_counter:", self.hardcoded_goal_idx_counter)

                    self.hardcoded_goal_idx_counter += 1


                self.subgoal_counter = 0
                

                if self.num_samples > 1:
                    goal_idx, goal_images_info, mode = self.goal_filtering_fn(goal, rgb_obs, goal_images)

                    
                    

                    
                    self.goal_image = goal_images[goal_idx]

                    for key, val in goal_images_info.items():
                        self.goal_images_info[key].append(val)

                    # v = self.gc_vf.value_function_ranking(rgb_obs, goal_images)
                    # v_idx = v.argmax()
                    # self.goal_image = goal_images[v_idx]

                    # self.vranked_goal_images_seq.append({v[i]:goal_images[i] for i in range(v.shape[0])})
                else:
                    assert goal_images.shape[0] == 1, f"goal_images.shape: {goal_images.shape}"
                    self.goal_image = goal_images[0]


                self.generated_goals.append(self.goal_image.copy())

            self.retrospective_true_goals.append(rgb_obs.copy())

        # Log the image observation and the goal image
        self.obs_image_seq.append(rgb_obs)

        if not self.flat_policy:
            self.goal_image_seq.append(self.goal_image)
            self.combined_images.append(np.concatenate([rgb_obs, self.goal_image], axis=1))
            assert self.combined_images[-1].shape == (200, 400, 3)


        if self.flat_policy:
            action_cmd = self.gc_policy.predict_action_lc(rgb_obs, self.language_task)
        else:
            # Query the behavior cloning model
            action_cmd = self.gc_policy.predict_action(rgb_obs, self.goal_image)

        # Log the predicted action
        self.action_seq.append(action_cmd)

        # Update variables
        self.subgoal_counter += 1

        # Update progress bar
        self.pbar.update(1)

        return action_cmd, mode


def evaluate_policy(model, env, epoch=0, eval_log_dir=None, debug=False, create_plan_tsne=False):
    """
    Run this function to evaluate a model on the CALVIN challenge.

    Args:
        model: Must implement methods of CalvinBaseModel.
        env: (Wrapped) calvin env.
        epoch:
        eval_log_dir: Path where to log evaluation results. If None, logs to /tmp/evaluation/
        debug: If True, show camera view and debug info.
        create_plan_tsne: Collect data for TSNE plots of latent plans (does not work for your custom model)

    Returns:
        Dictionary with results
    """
    # conf_dir = Path(__file__).absolute().parents[2] / "conf"
    conf_dir = Path(__file__).absolute().parents[0] / "calvin_models" / "conf"
    task_cfg = OmegaConf.load(conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml")
    task_oracle = hydra.utils.instantiate(task_cfg)
    val_annotations = OmegaConf.load(conf_dir / "annotations/new_playtable_validation.yaml")

    eval_log_dir = get_log_dir(eval_log_dir)

    eval_sequences = get_sequences(NUM_SEQUENCES)


    pickle_object(os.path.join(eval_log_dir, "saved_state", "eval_sequences.pkl"), eval_sequences)



    results = []
    plans = defaultdict(list) 

    if not debug:
        eval_sequences = tqdm(eval_sequences, position=0, leave=True)

    eval_idx = 0
    for initial_state, eval_sequence in eval_sequences:
        print("initial_state:", initial_state)
        print("eval_sequence:", eval_sequence)
        result, _ = evaluate_sequence(env, model, task_oracle, initial_state, eval_sequence, val_annotations, plans, debug)
        results.append(result)
        if not debug:
            eval_sequences.set_description(
                " ".join([f"{i + 1}/5 : {v * 100:.1f}% |" for i, v in enumerate(count_success(results))]) + "|"
            )
        
        pickle_object(os.path.join(eval_log_dir, "saved_state", f"results_{eval_idx}.pkl"), results)
        pickle_object(os.path.join(eval_log_dir, "saved_state", f"plans_{eval_idx}.pkl"), plans)
        eval_idx += 1

    if create_plan_tsne:
        create_tsne(plans, eval_log_dir, epoch)
    print_and_save(results, eval_sequences, eval_log_dir, epoch)

    return results


def pickle_object(filepath, obj):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, 'wb') as file:
        pickle.dump(obj, file)

def evaluate_policy_singletask(model, env, epoch=0, eval_log_dir=None, debug=False, create_plan_tsne=False):
    """
    Run this function to evaluate a model on the CALVIN challenge.

    Args:
        model: Must implement methods of CalvinBaseModel.
        env: (Wrapped) calvin env.
        epoch:
        eval_log_dir: Path where to log evaluation results. If None, logs to /tmp/evaluation/
        debug: If True, show camera view and debug info.
        create_plan_tsne: Collect data for TSNE plots of latent plans (does not work for your custom model)

    Returns:
        Dictionary with results
    """
    # conf_dir = Path(__file__).absolute().parents[2] / "conf"
    conf_dir = Path(__file__).absolute().parents[0] / "calvin_models" / "conf"
    task_cfg = OmegaConf.load(conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml")
    task_oracle = hydra.utils.instantiate(task_cfg)
    val_annotations = OmegaConf.load(conf_dir / "annotations/new_playtable_validation.yaml")

    eval_log_dir = get_log_dir(eval_log_dir)

    base_initial_state = {'led': 0, 'lightbulb': 0, 'slider': 'left', 'drawer': 'open', 'red_block': 'slider_right', 'blue_block': 'table', 'pink_block': 'slider_left', 'grasped': 0}

    
    # Figure out issues w the tasks that aren't resetting properly 
    #     - fopr block tasks, the code doesn't support setting the state so that the block is in the drawer (calvin_models/calvin_agent/evaluation/utils.py)
    #     - also doesn't support having it grasped for place_in_slider, place_in_drawer, stack_block (calvin_models/calvin_agent/evaluation/utils.py)
    #     - for unstack, seems like it's correctly implemented, it just fails at it. The success fn is a little confusing. Failing the first condition. Maybe this means that the block isn't supposed to flip over or something? (calvin_env/calvin_env/envs/tasks.py)
    skip_tasks = ["lift_red_block_drawer", "lift_blue_block_drawer", "lift_pink_block_drawer", "place_in_slider", "place_in_drawer", "stack_block"] # use for collect
    # skip_tasks = ["lift_red_block_drawer", "lift_blue_block_drawer", "lift_pink_block_drawer", "place_in_slider", "place_in_drawer", "stack_block", "unstack_block", "push_pink_block_right"] # use for replay/eval
    # skip_tasks = ["lift_red_block_drawer", "lift_blue_block_drawer", "lift_pink_block_drawer", "place_in_slider", "place_in_drawer", "stack_block", "unstack_block", "turn_off_led"] # use for replay/eval

    # chat_gpt_tasks = ["push_pink_block_right", "push_blue_block_right", "push_red_block_right", "push_into_drawer"]
    chat_gpt_tasks = ["push_red_block_right", "push_blue_block_right", "push_pink_block_right", "push_pink_block_left", "turn_off_led", "push_into_drawer"]
    # chat_gpt_tasks = ["push_into_drawer"]

    # hooman_tasks = ["lift_red_block_table"] 

    if model.filtering_method in ["human", "human_dummy", "high_level_vf"]:
        initial_state = deepcopy(base_initial_state)
        initial_state["red_block"] = "table"
        initial_state["blue_block"] = "table"
        initial_state["pink_block"] = "slider_right"
        
        # Make sure the initial state has the correct starting conditions for that task 
        condition = tasks["lift_red_block_table"][0]["condition"]
        for key, val in condition.items():
            if isinstance(val, list):
                initial_state[key] = val[0]
            else:
                initial_state[key] = val

        eval_sequences = [(initial_state, ("lift_red_block_table", "place_in_slider")) for _ in range(NUM_SEQUENCES)]
    else:

        eval_sequences = []
        for task_name in tasks.keys():
            for _ in range(NUM_SEQUENCES):
                
                if model.filtering_method == "chat_gpt" or model.filtering_method == "chat_gpt_dummy" or model.filtering_method == "chat_gpt_pairwise":
                    if task_name not in chat_gpt_tasks:
                        continue 
                # if model.filtering_method == "human" or model.filtering_method == "human_dummy":
                #     if task_name not in hooman_tasks:
                #         continue 
                else:
                    if task_name in skip_tasks:
                        continue


                # initial_state = base_initial_state.copy()
                initial_state = deepcopy(base_initial_state)
                
                # Make sure the initial state has the correct starting conditions for that task 
                condition = tasks[task_name][0]["condition"]
                for key, val in condition.items():
                    if isinstance(val, list):
                        initial_state[key] = val[0]
                    else:
                        initial_state[key] = val

                # Manually make sure for the lift_blue_block_slider task that the pink block is not on top of the blue block in the slider 
                if task_name == "lift_blue_block_slider":
                    initial_state["pink_block"] = "table"
                if task_name == "lift_red_block_slider":
                    initial_state["pink_block"] = "table"

                eval_sequences.append((initial_state, (task_name,)))

    
    if DEBUG:
        eval_sequences = eval_sequences[:NUM_SEQUENCES]
    
    if model.oracle_goals is not None:
        for eval_sequence in eval_sequences:
            task_name = eval_sequence[1][0]
            assert task_name in model.oracle_goals, f"\"{task_name}\" not in model.oracle_goals. model.oracle_goals.keys(): {model.oracle_goals.keys()}"


    pickle_object(os.path.join(eval_log_dir, "saved_state", "eval_sequences.pkl"), eval_sequences)

    results = []
    plans = defaultdict(list) 

    

    if not debug:
        eval_sequences = tqdm(eval_sequences, position=0, leave=True)

    eval_idx = 0
    for initial_state, eval_sequence in eval_sequences:
    # for eval_idx in range(len(eval_sequences)):
        # initial_state, eval_sequence = eval_sequences[eval_idx]
        print("initial_state:", initial_state)
        print("eval_sequence:", eval_sequence)
        result, _ = evaluate_sequence(env, model, task_oracle, initial_state, eval_sequence, val_annotations, plans, debug)
        results.append(result)
        if not debug:
            eval_sequences.set_description(
                " ".join([f"{i + 1}/5 : {v * 100:.1f}% |" for i, v in enumerate(count_success(results))]) + "|"
            )

        pickle_object(os.path.join(eval_log_dir, "saved_state", f"results_{eval_idx}.pkl"), results)
        pickle_object(os.path.join(eval_log_dir, "saved_state", f"plans_{eval_idx}.pkl"), plans)
        print(f"eval sequence {eval_idx + 1}/{len(eval_sequences)}")
        eval_idx += 1

    if create_plan_tsne:
        create_tsne(plans, eval_log_dir, epoch)
    print_and_save(results, eval_sequences, eval_log_dir, epoch)

    return results



def evaluate_policy_libero(model, env, eval_log_dir=None, fix_init_state=False):
    eval_log_dir = get_log_dir(eval_log_dir)
    # results = defaultdict(list)
    results = defaultdict(lambda: defaultdict(list))
    # success_sequence_lengths = defaultdict(list) 
    # num_subtask_successeses = defaultdict(list) 
    plans = defaultdict(list) 
    eval_tasks = env.val_tasks


    if fix_init_state:
        skip_tasks = ["LIVING_ROOM_SCENE4_pick_up_the_black_bowl_on_the_left_and_put_it_in_the_tray"]
        eval_tasks = [task for task in eval_tasks if task not in skip_tasks]


    if DEBUG:
        eval_tasks = eval_tasks[:2]


    if model.filtering_method in ["human", "human_dummy", "high_level_vf"]:
        # eval_tasks = [
        #     "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy",
        #     # "KITCHEN_SCENE2_put_the_black_bowl_at_the_front_on_the_plate",
        #     # "KITCHEN_SCENE5_put_the_black_bowl_on_top_of_the_cabinet",
        #     "KITCHEN_SCENE9_put_the_frying_pan_on_top_of_the_cabinet",
        #     # "LIVING_ROOM_SCENE2_pick_up_the_tomato_sauce_and_put_it_in_the_basket",
        #     # "LIVING_ROOM_SCENE4_pick_up_the_black_bowl_on_the_left_and_put_it_in_the_tray",

        #     # "pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate",
        #     # "pick_up_the_bbq_sauce_and_place_it_in_the_basket",
        #     "KITCHEN_SCENE1_open_the_top_drawer_of_the_cabinet",
        #     "KITCHEN_SCENE6_close_the_microwave",
        #     # "LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate",
        #     "STUDY_SCENE2_pick_up_the_book_and_place_it_in_the_front_compartment_of_the_caddy",

        # ]
        pass 
    elif model.filtering_method in ["low_level_vf"]:
        eval_tasks = [
            "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy",
            "KITCHEN_SCENE9_put_the_frying_pan_on_top_of_the_cabinet",
            "KITCHEN_SCENE1_open_the_top_drawer_of_the_cabinet",
            "KITCHEN_SCENE6_close_the_microwave",
            "STUDY_SCENE2_pick_up_the_book_and_place_it_in_the_front_compartment_of_the_caddy",
        ]

    for task_idx, eval_task in enumerate(eval_tasks):
        progress_bar = trange(NUM_SEQUENCES, position=0, leave=True)
        for ep_idx in progress_bar:
            success, success_sequence_length, num_subtask_successes, end_early = rollout_libero(env, model, eval_task, plans, fix_init_state=fix_init_state, ep_idx=ep_idx)
            # results[eval_task].append((success, success_sequence_length, num_subtask_successes))
            # results[eval_task].append((success, success_sequence_length, num_subtask_successes))

            results[eval_task]["success"].append(success)
            results[eval_task]["success_sequence_length"].append(success_sequence_length)
            results[eval_task]["num_subtask_successes"].append(num_subtask_successes)

            # success_sequence_lengths[eval_task].append(success_sequence_length)
            # num_subtask_successeses[eval_task].append(num_subtask_successes)
            progress_bar.set_description(f"[task {task_idx + 1}/{len(eval_tasks)} ep {ep_idx + 1}/{NUM_SEQUENCES}] ")
            # progress_bar.set_description(" ".join([f"{i + 1}/5 : {v * 100:.1f}% |" for i, v in enumerate(count_success(results))]) + "|")

            if end_early:
                break 


    print_and_save_libero(results, eval_tasks, eval_log_dir)
    return results



def print_and_save_libero(results, eval_tasks, eval_log_dir):

    for task_name, result in results.items():
        successes = result["success"]
        success_sequence_lengths = result["success_sequence_length"]
        num_subtask_successeses = result["num_subtask_successes"]

        print(f"{task_name}: success {np.mean(successes) * 100:.1f}%, success_sequence_length: {np.mean(success_sequence_lengths):.1f}, num_subtask_successes: {np.mean(num_subtask_successeses):.1f}")

    json_data = {key:{k:list(np.array(v, dtype=float)) for k,v  in val.items()} for key, val in results.items()}
    with open(eval_log_dir / "results.json", "w") as file:
        json.dump(json_data, file)




def rollout_libero(env, model, eval_task, plans, fix_init_state=False, ep_idx=0):
    # init_demo_id = 0 if fix_init_state else None
    init_demo_id = ep_idx if fix_init_state else None
    obs = env.reset(eval_task, init_demo_id=init_demo_id)
    lang_annotation = env.language_instruction
    model.reset()

    if fix_init_state and model.oracle_goals is not None:
        env_initial_image = obs["rgb_obs"]["rgb_static"]
        dataset_initial_image = model.oracle_goals["initial_dataset_images"][eval_task][ep_idx]
        initial_images = np.concatenate([env_initial_image, dataset_initial_image], axis=1)
    else:
        initial_images = None

    rewards = []
    # success_sequence_lengths = []
    # num_subtask_successes = []
    for step in range(EP_LEN):
        # make sure eval_task is the right thing to pass in for subtask, or if I need to do env.current_task. Or at least assert eval_task == env.current_task
        end_early = False
        action, mode = model.step(obs, lang_annotation, eval_task, ep_idx=ep_idx)

        if isinstance(action, int) and action == -1:
            print("Ran out of hard coded goal images")
            break 

        if mode == "end_episode":
            print("mode == \"end_episode\", ending episode early")
            break 


        if mode == "quit":
            print("mode == \"quit\", exiting out of eval loop")
            end_early = True
            break 

        if step == 0:
            # for tsne plot, only if available
            collect_plan(model, plans, lang_annotation)

        obs, reward, done, info = env.step(action)
        # success_sequence_lengths.append(info["success_sequence_length"])
        # success_sequence_lengths.append(info["num_subtask_successes"])

        rewards.append(reward)
        if done:
            break 

    success = np.sum(rewards) > 0

    if success:
        print(colored("success", "green"))
    else:
        print(colored("fail", "red"))

    model.save_info(success, initial_images=initial_images)
    return int(success), info["success_sequence_length"], info["num_subtask_successes"], end_early




def evaluate_sequence(env, model, task_checker, initial_state, eval_sequence, val_annotations, plans, debug):
    """
    Evaluates a sequence of language instructions.
    """
    robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
    env.reset(robot_obs=robot_obs, scene_obs=scene_obs)

    success_counter = 0
    if debug:
        time.sleep(1)
        print()
        print()
        print(f"Evaluating sequence: {' -> '.join(eval_sequence)}")
        print("Subtask: ", end="")
    for subtask in eval_sequence:
        success, end_early = rollout(env, model, task_checker, subtask, val_annotations, plans, debug)
        if success:
            success_counter += 1
        else:
            return success_counter, end_early
    return success_counter, end_early 


def rollout(env, model, task_oracle, subtask, val_annotations, plans, debug):
    """
    Run the actual rollout on one subtask (which is one natural language instruction).
    """
    if debug:
        print(f"{subtask} ", end="")
        time.sleep(0.5)

    obs = env.get_obs()

    # get lang annotation for subtask
    lang_annotation = val_annotations[subtask][0]
    model.reset()
    start_info = env.get_info()

    print("lang_annotation:", lang_annotation)

    for step in range(EP_LEN):
        action, _ = model.step(obs, lang_annotation, subtask)

        if isinstance(action, int) and action == -1:
            print("Ran out of hard coded goal images")
            break 

        obs, _, _, current_info = env.step(action)
        rgb_obs = obs["rgb_obs"]["rgb_static"]
        if debug:
            img = env.render(mode="rgb_array")
            join_vis_lang(img, lang_annotation)
            # time.sleep(0.1)
        if step == 0:
            # for tsne plot, only if available
            collect_plan(model, plans, subtask)

        # check if current step solves a task
        current_task_info = task_oracle.get_task_info_for_set(start_info, current_info, {subtask})
        if len(current_task_info) > 0:
            if debug:
                print(colored("success", "green"), end=" ")
            print(colored("success", "green"), end=" ")
            print("step:", step)
            print("current_task_info:", current_task_info)
            model.retrospective_true_goals.append(rgb_obs)
            model.retrospective_true_goals = model.retrospective_true_goals[1:]
            model.save_info(True)
            return True, False
    if debug:
        print(colored("fail", "red"), end=" ")
    model.retrospective_true_goals = model.retrospective_true_goals[1:]
    model.retrospective_true_goals.append(rgb_obs)
    model.save_info(False)
    print(colored("fail", "red"), end=" ")
    return False, False

def parse_arguments():
    parser = argparse.ArgumentParser(description="Evaluate a trained model on multistep sequences with language goals.")
    parser.add_argument("--dataset_path", type=str, help="Path to the dataset root directory.")

    # arguments for loading default model
    parser.add_argument(
        "--train_folder", type=str, help="If calvin_agent was used to train, specify path to the log dir."
    )
    parser.add_argument(
        "--checkpoints",
        type=str,
        default=None,
        help="Comma separated list of epochs for which checkpoints will be loaded",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path of the checkpoint",
    )
    parser.add_argument(
        "--last_k_checkpoints",
        type=int,
        help="Specify the number of checkpoints you want to evaluate (starting from last). Only used for calvin_agent.",
    )

    # # arguments for loading custom model or custom language embeddings
    # # parser.add_argument(
    # #     "--custom_model", action="store_true", help="Use this option to evaluate a custom model architecture."
    # # )
    # parser.add_argument(
    #     "--custom_model", type=int, default=1, help="Use this option to evaluate a custom model architecture."
    # )

    parser.add_argument(
        "--diffusion_model_framework",
        type=str,
        default="jax",
        choices=["jax", "pytorch"],
        help="Comma separated list of epochs for which checkpoints will be loaded",
    )

    parser.add_argument(
        "--diffusion_model_checkpoint_path", type=str, help="Use this option to evaluate a custom model architecture."
    )

    parser.add_argument(
        "--gc_policy_checkpoint_path", type=str, help="Use this option to evaluate a custom model architecture."
    )


    parser.add_argument(
        "--gc_vf_checkpoint_path", type=str, help="Use this option to evaluate a custom model architecture."
    )

    parser.add_argument(
        "--s3_save_uri", type=str, help="Use this option to evaluate a custom model architecture."
    )

    parser.add_argument(
        "--save_to_s3", type=int, default=1, help="Use this option to evaluate a custom model architecture."
    )


    parser.add_argument(
        "--num_denoising_steps", type=int, default=200, help="Use this option to evaluate a custom model architecture."
    )

    # parser.add_argument(
    #     "--agent_type", type=str, help="Use this option to evaluate a custom model architecture."
    # )

    parser.add_argument(
        "--use_temporal_ensembling", type=int, default=1, help="Use this option to evaluate a custom model architecture."
    )

    parser.add_argument(
        "--num_samples", type=int, default=1, help="Use this option to evaluate a custom model architecture."
    )


    parser.add_argument(
        "--filtering_method", type=str, default=None, help="Use this option to evaluate a custom model architecture."
    )

    parser.add_argument(
        "--flat_policy", type=int, default=0, help="Use this option to evaluate a custom model architecture."
    )


    parser.add_argument(
        "--prompt_w", type=float, default=7.5, help="Use this option to evaluate a custom model architecture."
    )

    parser.add_argument(
        "--context_w", type=float, default=1.5, help="Use this option to evaluate a custom model architecture."
    )



    # parser.add_argument("--debug", action="store_true", help="Print debug info and visualize environment.")
    parser.add_argument("--debug", type=int, default=0)


    parser.add_argument("--single_task", type=int, default=0)

    # parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--eval_log_dir", default=None, type=str, help="Where to log the evaluation results.")

    parser.add_argument("--oracle_goals_dir", default=None, type=str, help="Where to log the evaluation results.")

    parser.add_argument("--agent_config_string", default="calvin", type=str, help="Where to log the evaluation results.")

    parser.add_argument("--vf_agent_config_string", default=None, type=str, help="Where to log the evaluation results.")

    parser.add_argument("--oracle_goals_type", default=None, choices=[None, "generated", "retrospective_true", "dataset_true"], type=str, help="Where to log the evaluation results.")

     

    parser.add_argument("--device", default=0, type=int, help="CUDA device")
    return parser.parse_args()



def main():
    # seed_everything(0, workers=True)  # type:ignore ###UNCOMMENT
    # random.seed(args.seed)
    # np.random.seed(args.seed)
    # torch.manual_seed(args.seed)

    args = parse_arguments()

    # evaluate a custom model
    agent_config, env_name, agent_type, _ = get_config(args.agent_config_string)

    if args.filtering_method in ["high_level_vf", "low_level_vf"]:
        vf_agent_config, _, vf_agent_type, _ = get_config(args.vf_agent_config_string)
    else:
        vf_agent_config, vf_agent_type = None, None


    model = CustomModel(agent_config, env_name, agent_type, args.single_task, args.oracle_goals_dir, args.oracle_goals_type, args.use_temporal_ensembling, args.diffusion_model_checkpoint_path, args.gc_policy_checkpoint_path, args.gc_vf_checkpoint_path, args.num_denoising_steps, num_samples=args.num_samples, prompt_w=args.prompt_w, context_w=args.context_w, filtering_method=args.filtering_method, flat_policy=args.flat_policy, diffusion_model_framework=args.diffusion_model_framework, vf_agent_config=vf_agent_config, vf_agent_type=vf_agent_type)
    env = make_env(env_name, args.dataset_path)
    

    if env_name == "calvin" or env_name == "calvinlcbc":
        assert not args.debug

        if args.single_task:
            evaluate_policy_singletask(model, env, debug=args.debug, eval_log_dir=model.log_dir)
        else:
            evaluate_policy(model, env, debug=args.debug, eval_log_dir=model.log_dir)
    elif "libero" in env_name:
        evaluate_policy_libero(model, env, eval_log_dir=model.log_dir, fix_init_state=args.single_task)
        env.close_env()
    else:
        raise ValueError(f"Unsupported env_name: \"{env_name}\".")

    if args.save_to_s3:
        s3_callback = S3SyncCallback(model.log_dir, os.path.join(args.s3_save_uri, "/".join(model.log_dir.split("results")[-1].strip("/").split("/"))) + "/")
        s3_callback.on_train_epoch_end()
    

if __name__ == "__main__":
    main()
