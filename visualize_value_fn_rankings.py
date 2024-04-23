import os 
import numpy as np 
from glob import glob 
from tqdm import tqdm, trange 
from jaxrl_m.data.text_processing import text_processors

from calvin_agent.evaluation.gcbc_train_config import get_config
from calvin_agent.evaluation import diffusion_gc_policy
import cv2


def visualize_outputs(args):
    # episode_dirs = sorted(glob(os.path.join(args.logdir, "ep*")))

    # for episode_dir in episode_dirs:
        
    #     goal_image_files = glob(os.path.join(episode_dir, "unfiltered_goal_images", ""))

    agent_config_string = "calvingcbclcbc_gcdiscriminator_noactnorm-auggoaldiff-sagemaker-generatedencdecgoal-frac0.25"
     
    agent_config, env_name, agent_type, _ = get_config(agent_config_string)

    agent_config.dataset_kwargs.use_generated_goals = False 
    agent_config.dataset_kwargs.use_encode_decode_goals = False 
    agent_config.dataset_kwargs.goal_relabeling_strategy = "delta_goals"
    agent_config.dataset_kwargs.goal_relabeling_kwargs = {"goal_delta": (16, 24)}

    

    vf_agent = diffusion_gc_policy.GCPolicy(agent_config, env_name, agent_type, os.getenv("GC_POLICY_CHECKPOINT"), normalize_actions=False, use_temporal_ensembling=False, text_processor=None)

    
    frames = []
    timestep_dirs = sorted(glob(os.path.join(args.ep_dir, "unfiltered_goal_images", "timestep_*")))
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


        frame = [rgb_image_obs]

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_color = (0, 255, 0)  # White color in BGR
        line_type = 2  # Line thickness
        for i in range(ordered_vs.shape[0]):
            img = ordered_goal_images[i]
            val = ordered_vs[i]
            img = cv2.putText(img, f'{val:.3f}', (10, 15), font, font_scale, font_color, line_type)
            frame.append(img)

        frame = np.concatenate(frame, axis=1)
        frames.append(frame)

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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    visualize_outputs(args)

"""
export PYTHONPATH="/home/kylehatch/Desktop/hidql/calvin-sim/external/susie:$PYTHONPATH"
export PYTHONPATH="/home/kylehatch/Desktop/hidql/calvin-sim/external/jaxrl_m:$PYTHONPATH"
export PYTHONPATH="/home/kylehatch/Desktop/hidql/calvin-sim/external/urdfpy:$PYTHONPATH"
export PYTHONPATH="/home/kylehatch/Desktop/hidql/calvin-sim/external/networkx:$PYTHONPATH"

export GC_POLICY_CHECKPOINT=/home/kylehatch/Desktop/hidql/susie-calvin-checkpoints/susie_low_level/calvingcbclcbc/gcdiscriminator/auggoaldiff/seed_0/20240412_030212/checkpoint_50000
python3 -u visualize_value_fn_rankings.py \
--ep_dir /home/kylehatch/Desktop/hidql/results/single_task/calvin/public_model/checkpoint_only/calvin/gcdiffusion/auggoaldiff/seed_0/20240227_194024/checkpoint_150000/chat_gpt_dummy_filter_4s_2024.04.10_15.47.24/ep31



"""