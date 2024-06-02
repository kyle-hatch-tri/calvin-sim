import os
from collections import defaultdict
from tqdm import tqdm, trange 
from glob import glob 
import cv2
import numpy as np
import shutil


from jaxrl_m.data.text_processing import text_processors

from calvin_agent.evaluation.gcbc_train_config import get_config
from calvin_agent.evaluation import diffusion_gc_policy


# os.environ["XDG_SESSION_TYPE"] = "xcb"

# # RESULTS_DIR = "random_util_scripts/datasets/progress_towards_goal/chatgptdummy16samples2024.04.25_14.04.28"
# OUTPUT_DIR = "random_util_scripts/highlevel_vf_output/progress_towards_goal"

# RESULTS_DIR = "/home/kylehatch/Desktop/hidql/saved_calvin_eval_results/results/liberosplit2/test1_400smthlibs2_2024.02.23_14.20.24/40000/liberosplit2/gcdiffusion/auggoaldiff/seed_0/20240228_024329/checkpoint_150000/chatgptdummy16slibs22024.04.26_12.27.01"
# # EPISODES_LIST = [141, 145, 150] # STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy
# # EPISODES_LIST = [320, 321, 324] # KITCHEN_SCENE9_put_the_frying_pan_on_top_of_the_cabinet
# EPISODES_LIST = [364, 365, 368, 373, 375, 378] # LIVING_ROOM_SCENE4_pick_up_the_black_bowl_on_the_left_and_put_it_in_the_tray

OUTPUT_DIR = "random_util_scripts/highlevel_vf_output/low_level_or_frequency"

# RESULTS_DIR = "/home/kylehatch/Desktop/hidql/saved_calvin_eval_results/results/calvin/public_model/checkpoint_only/calvin/gcdiffusion/auggoaldiff/seed_0/20240227_194024/checkpoint_150000/chatgptdummy16samples2024.04.25_14.04.28"
# # EPISODES_LIST = [230, 11, 23, 124, 182, 206, 219, 349, 351, 408]
# EPISODES_LIST = [353, 408, 351, 349]


RESULTS_DIR = "/home/kylehatch/Desktop/hidql/saved_calvin_eval_results/results/liberosplit2/test1_400smthlibs2_2024.02.23_14.20.24/40000/liberosplit2/gcdiffusion/auggoaldiff/seed_0/20240228_024329/checkpoint_150000/chatgptdummy16slibs22024.04.26_12.27.01"
# EPISODES_LIST = [3, 4, 143, 148, 152, 153,  233, 261, 276, 326, 459]
EPISODES_LIST = [201, 215, 218]

def run_value_function_ranking(args):
    os.chdir("..")

    agent_config_string = args.agent_config_string
     
    agent_config, env_name, agent_type, _ = get_config(agent_config_string)

    agent_config.dataset_kwargs.use_generated_goals = False 
    agent_config.dataset_kwargs.use_encode_decode_goals = False 
    agent_config.dataset_kwargs.goal_relabeling_strategy = "delta_goals"
    agent_config.dataset_kwargs.goal_relabeling_kwargs = {"goal_delta": (16, 24)}

    text_processor = text_processors["muse_embedding"]()
    vf_agent = diffusion_gc_policy.GCPolicy(agent_config, env_name, agent_type, os.getenv("GC_POLICY_CHECKPOINT"), normalize_actions=False, use_temporal_ensembling=False, text_processor=text_processor)

    output_dir = os.path.join(OUTPUT_DIR, RESULTS_DIR.split("/")[-1])
    os.makedirs(output_dir, exist_ok=True)

    # Use glob to find file paths matching the pattern
    file_paths = glob(os.path.join(RESULTS_DIR, "ep*/language_task.txt"))


    # Define a custom sorting function based on the folder name
    def get_ep_number(file_path):
        folder_name = os.path.dirname(file_path)
        return int(folder_name.split("/")[-1][2:])
    
    filtered_file_paths = [path for path in file_paths if get_ep_number(path) in EPISODES_LIST]

    # Sort the filtered file paths using the custom sorting function
    sorted_file_paths = sorted(filtered_file_paths, key=get_ep_number)

    episode_dirs = [os.path.dirname(path) for path in sorted_file_paths]

    # print(f"There are {len(list(episodes_by_task.keys()))} total tasks to look through.")
    print(f"There are {len(episode_dirs)} total episodes to label.")

    for ep_idx, episode_dir in enumerate(episode_dirs):
        with open(os.path.join(episode_dir, "language_task.txt"), "r") as f:
            task = f.readline().strip()
            language_goal = f.readline().strip()
        
        timestep_goal_images_dirs = sorted(glob(os.path.join(episode_dir, "unfiltered_goal_images", "timestep_*"))) 
        query_frames = []
        for timestep_idx, timestep_dir in enumerate(tqdm(timestep_goal_images_dirs)):

            ep_name = episode_dir.split("/")[-1]
            timestep_name = timestep_dir.split("/")[-1]
            save_dir = os.path.join(output_dir, ep_name, timestep_name)

            os.makedirs(save_dir, exist_ok=True)

            # with open(os.path.join(timestep_dir, "correct_goal_idxs.txt"), "r") as f:
            #     correct_goal_idxs = f.readline().strip(" \n,").split(",")
            #     correct_goal_idxs = [int(x) for x in correct_goal_idxs if x != ""]
            
            
            rgb_obs_file = os.path.join(timestep_dir, "rgb_obs.png")
            goal_images_files = sorted(glob(os.path.join(timestep_dir, "goal_image_*.png")))

            goal_images = np.stack([cv2.imread(goal_images_file)[..., ::-1] for goal_images_file in goal_images_files], axis=0)
            

            rgb_obs = cv2.imread(rgb_obs_file)[..., ::-1]

            if args.vf_type == "lcgc":
                v = vf_agent.value_function_ranking_lcgc(rgb_obs, goal_images, language_goal)
            elif args.vf_type == "gc":
                v = vf_agent.value_function_ranking(rgb_obs, goal_images)
            else:
                raise ValueError(f"Unsupported value function agent type: \"{args.vf_type}\".")
            
            sorted_idxs = np.argsort(v)[::-1]
            ordered_goal_images = goal_images[sorted_idxs]
            ordered_vs = v[sorted_idxs]

            

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            font_color = (0, 255, 0) 
            line_type = 2  # Line thickness

            frame = []
            for i in range(ordered_goal_images.shape[0]):
                img = ordered_goal_images[i]

                if sorted_idxs[i] == 0:
                    img = cv2.rectangle(img, (0, 0), (img.shape[1] - 1, img.shape[0] - 1), (255, 0, 0), 6) 

                img = cv2.putText(img, f'[{i}] v: {ordered_vs[i]:.3f}', (25, 20), font, font_scale, font_color, line_type)
                # if orig_idx in correct_goal_idxs:
                #     img = cv2.putText(img, f'[{i}] v: {ordered_vs[i]:.3f}', (25, 20), font, font_scale, font_color, line_type)
                # else:
                #     img = cv2.putText(img, f'[{i}] v: {ordered_vs[i]:.3f}', (25, 20), font, font_scale, (255, 0, 0), line_type)
                frame.append(img)
            

            assert len(frame) % 4 == 0, f"len(frame): {len(frame)}"
            frame_rows = []
            for row_idx in range(len(frame) // 4):
                start = row_idx * 4
                end = start + 4
                frame_row = np.concatenate([rgb_obs] + frame[start:end], axis=1)
                frame_rows.append(frame_row)

            query_frame = np.concatenate(frame_rows, axis=0)
            query_frame = cv2.putText(query_frame, f't={timestep_name.split("_")[-1]}', (25, 40), font, font_scale, font_color, line_type)
            query_frames.append(query_frame)

            # cv2.imwrite(os.path.join(save_dir, "query_frame.png"), query_frame[..., ::-1])

        query_frames = np.stack(query_frames, axis=0)
        save_video(os.path.join(os.path.join(output_dir, ep_name), "query_frames.mp4"), query_frames, fps=10) # channel reversal happens in here



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

def parse_arguments():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate a trained model on multistep sequences with language goals.")
    parser.add_argument("--agent_config_string", type=str, default=None, help="Path to the dataset root directory.")
    parser.add_argument("--vf_type", type=str, default=None, help="Path to the dataset root directory.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    run_value_function_ranking(args)


"""
export CUDA_VISIBLE_DEVICES=0 
export PYTHONPATH="/home/kylehatch/Desktop/hidql/calvin-sim/external/susie:$PYTHONPATH"
export PYTHONPATH="/home/kylehatch/Desktop/hidql/calvin-sim/external/jaxrl_m:$PYTHONPATH"
export PYTHONPATH="/home/kylehatch/Desktop/hidql/calvin-sim/external/urdfpy:$PYTHONPATH"
export PYTHONPATH="/home/kylehatch/Desktop/hidql/calvin-sim/external/networkx:$PYTHONPATH"

export GC_POLICY_CHECKPOINT=/home/kylehatch/Desktop/hidql/susie-calvin-checkpoints/susie_low_level/calvinlcbc/lcgcprogressvf/auggoaldiff/seed_0/20240510_005751/checkpoint_100000
python3 -u eval_highlevel_vf_on_dataset.py \
--agent_config_string calvinlcbc_lcgcprogressvf_noactnorm-auggoaldiff \
--vf_type lcgc


export GC_POLICY_CHECKPOINT=/home/kylehatch/Desktop/hidql/susie-calvin-checkpoints/susie_low_level/calvin/gciql/auggoaldiffhparams5/seed_0/20240505_020833/checkpoint_150000
python3 -u eval_highlevel_vf_on_dataset.py \
--agent_config_string calvin_gciql_noactnorm-hparams5-auggoaldiff \
--vf_type gc



export GC_POLICY_CHECKPOINT=/home/kylehatch/Desktop/hidql/susie-calvin-checkpoints/susie_low_level/liberosplit2/gciql/auggoaldiffhparams5/seed_0/20240513_190251/checkpoint_100000
python3 -u eval_highlevel_vf_on_dataset.py \
--agent_config_string liberosplit2_gciql_noactnorm-hparams5-auggoaldiff \
--vf_type gc




"""