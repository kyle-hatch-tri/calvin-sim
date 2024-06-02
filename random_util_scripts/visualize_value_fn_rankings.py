import os 
import numpy as np 
from glob import glob 
from tqdm import tqdm, trange 
from jaxrl_m.data.text_processing import text_processors

from calvin_agent.evaluation.gcbc_train_config import get_config
from calvin_agent.evaluation import diffusion_gc_policy
import cv2



def visualize_outputs(args):
    os.chdir("..")
    # episode_dirs = sorted(glob(os.path.join(args.logdir, "ep*")))

    # for episode_dir in episode_dirs:
        
    #     goal_image_files = glob(os.path.join(episode_dir, "unfiltered_goal_images", ""))

    # agent_config_string = "calvingcbclcbc_gcdiscriminator_noactnorm-auggoaldiff-sagemaker-generatedencdecgoal-frac0.25"
    agent_config_string = args.agent_config_string
     
    agent_config, env_name, agent_type, _ = get_config(agent_config_string)

    agent_config.dataset_kwargs.use_generated_goals = False 
    agent_config.dataset_kwargs.use_encode_decode_goals = False 
    agent_config.dataset_kwargs.goal_relabeling_strategy = "delta_goals"
    agent_config.dataset_kwargs.goal_relabeling_kwargs = {"goal_delta": (16, 24)}

    

    vf_agent = diffusion_gc_policy.GCPolicy(agent_config, env_name, agent_type, os.getenv("GC_POLICY_CHECKPOINT"), normalize_actions=False, use_temporal_ensembling=False, text_processor=None)

    
    timestep_dirs = sorted(glob(os.path.join(args.ep_dir, "unfiltered_goal_images", "timestep_*")))

    frames = []
    for timestep_dir in timestep_dirs:

        goal_images = []
        goal_image_files = sorted(glob(os.path.join(timestep_dir, "goal_image_*.png")))
        for i, goal_image_file in enumerate(goal_image_files):
            
            goal_image = cv2.imread(goal_image_file)
            goal_image = goal_image[..., ::-1]
            goal_images.append(goal_image)

        goal_images = np.stack(goal_images, axis=0)

        
        rgb_image_obs = cv2.imread(os.path.join(timestep_dir, "rgb_obs.png"))
        rgb_image_obs = rgb_image_obs[..., ::-1]
        

        with open(os.path.join(os.path.dirname(os.path.dirname(timestep_dir)), "language_task.txt"), "r") as f:
            _ = f.readline()
            language_goal = f.readline().strip()

        
        v = vf_agent.value_function_ranking(rgb_image_obs, goal_images)

        sorted_idxs = np.argsort(v)[::-1]

        ordered_goal_images = goal_images[sorted_idxs]
        ordered_vs = v[sorted_idxs]


        frame = []

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_color = (0, 255, 0)  # White color in BGR
        line_type = 2  # Line thickness
        for i in range(ordered_vs.shape[0]):
            img = ordered_goal_images[i]
            val = ordered_vs[i]
            # img = np.stack([img], axis=0)[0] # Have to do this to get around the cv2 error for some reason 
            img = cv2.putText(img, f'{val:.3f}', (10, 15), font, font_scale, font_color, line_type)
            frame.append(img)

        # frame = np.concatenate(frame, axis=1)

        assert len(frame) % 4 == 0, f"len(frame): {len(frame)}"
        frame_rows = []
        for row_idx in range(len(frame) // 4):
            start = row_idx * 4
            end = start + 4
            frame_row = np.concatenate([rgb_image_obs] + frame[start:end], axis=1)
            frame_rows.append(frame_row)

        query_frame = np.concatenate(frame_rows, axis=0)


        frames.append(query_frame)

    # cv2.imwrite("frame.png", frame[..., ::-1])
        
    save_video("./random_util_scripts/goal_images_vranked/traj.mp4", np.array(frames))


def save_video(output_video_file, frames):
     # Extract frame dimensions
    height, width, _ = frames.shape[1:]

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can also use other codecs such as 'XVID'
    fps = 30  # Adjust the frame rate as needed

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
    # parser.add_argument("--logdir", type=str, help="Path to the dataset root directory.")
    parser.add_argument("--ep_dir", type=str, help="Path to the dataset root directory.")
    parser.add_argument("--agent_config_string", type=str, help="Path to the dataset root directory.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    visualize_outputs(args)

"""
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="/home/kylehatch/Desktop/hidql/calvin-sim/external/susie:$PYTHONPATH"
export PYTHONPATH="/home/kylehatch/Desktop/hidql/calvin-sim/external/jaxrl_m:$PYTHONPATH"
export PYTHONPATH="/home/kylehatch/Desktop/hidql/calvin-sim/external/urdfpy:$PYTHONPATH"
export PYTHONPATH="/home/kylehatch/Desktop/hidql/calvin-sim/external/networkx:$PYTHONPATH"

export GC_POLICY_CHECKPOINT=/home/kylehatch/Desktop/hidql/susie-calvin-checkpoints/susie_low_level/calvin/contrastivevf/auggoaldiff/seed_0/20240423_222107/checkpoint_100000
python3 -u visualize_value_fn_rankings.py \
--agent_config_string calvin_contrastivevf_noactnorm-auggoaldiff-b1024 \
--ep_dir /home/kylehatch/Desktop/hidql/saved_calvin_eval_results/results/calvin/public_model/checkpoint_only/calvin/gcdiffusion/auggoaldiff/seed_0/20240227_194024/checkpoint_150000/chatgptdummy16samples2024.04.25_14.04.28/ep0

/home/kylehatch/Desktop/hidql/saved_calvin_eval_results/results/calvin/public_model/checkpoint_only/calvin/gcdiffusion/auggoaldiff/seed_0/20240227_194024/checkpoint_150000/chatgptdummy16samples2024.04.25_14.04.28/ep91

See how it does on real vs generated goal images 
See how it does on goal images from another task (both real and not real)
See how it does on images w and w/o hallucinations (esp. ones where the block is already in the gripper)




export GC_POLICY_CHECKPOINT=/home/kylehatch/Desktop/hidql/susie-calvin-checkpoints/susie_low_level/calvingcbclcbc/gcdiscriminator/auggoaldiff/seed_0/20240412_030212/checkpoint_50000
python3 -u visualize_value_fn_rankings.py \
--ep_dir /home/kylehatch/Desktop/hidql/results/single_task/calvin/public_model/checkpoint_only/calvin/gcdiffusion/auggoaldiff/seed_0/20240227_194024/checkpoint_150000/chat_gpt_dummy_filter_4s_2024.04.10_15.47.24/ep31



"""