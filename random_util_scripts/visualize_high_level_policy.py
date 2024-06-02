import os 
import numpy as np 
from glob import glob 
from tqdm import tqdm, trange 
from jaxrl_m.data.text_processing import text_processors

from calvin_agent.evaluation.gcbc_train_config import get_config
from calvin_agent.evaluation import diffusion_gc_policy
import cv2

from calvin_agent.evaluation import jax_diffusion_model


def visualize_outputs(args):
    os.chdir("..")


    diffusion_model = jax_diffusion_model.DiffusionModel(50, num_samples=16)

    
    timestep_dirs = sorted(glob(os.path.join(args.ep_dir, "unfiltered_goal_images", "timestep_*")))

    if args.slice is not None:
        timestep_dirs = timestep_dirs[:args.slice]

    frames = []
    for timestep_dir in tqdm(timestep_dirs):

        # goal_images = []
        # goal_image_files = sorted(glob(os.path.join(timestep_dir, "goal_image_*.png")))
        # for i, goal_image_file in enumerate(goal_image_files):
            
        #     goal_image = cv2.imread(goal_image_file)
        #     goal_image = goal_image[..., ::-1]
        #     goal_images.append(goal_image)

        # goal_images = np.stack(goal_images, axis=0)

        
        rgb_image_obs = cv2.imread(os.path.join(timestep_dir, "rgb_obs.png"))
        rgb_image_obs = rgb_image_obs[..., ::-1]
        

        with open(os.path.join(os.path.dirname(os.path.dirname(timestep_dir)), "language_task.txt"), "r") as f:
            _ = f.readline()
            language_goal = f.readline().strip()


        goal_images, inference_time = diffusion_model.generate(language_goal, rgb_image_obs, return_inference_time=True, prompt_w=args.prompt_w, context_w=args.context_w)

        frame = []

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_color = (0, 255, 0)  # White color in BGR
        line_type = 2  # Line thickness
        for i in range(goal_images.shape[0]):
            img = goal_images[i]
            # img = np.stack([img], axis=0)[0] # Have to do this to get around the cv2 error for some reason 
            # img = cv2.putText(img, f'{val:.3f}', (10, 15), font, font_scale, font_color, line_type)
            frame.append(img)


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
        
    ep_name = args.ep_dir.split("/")[-1]
        
    save_video(f"./random_util_scripts/high_level_policy_vis/{ep_name}_{args.desc}_p{args.prompt_w}_c{args.context_w}.mp4", np.array(frames))


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
    parser.add_argument("--desc", type=str, help="Path to the dataset root directory.")
    parser.add_argument("--slice", type=int, default=None, help="Path to the dataset root directory.")
    parser.add_argument("--prompt_w", type=float, default=7.5, help="Path to the dataset root directory.")
    parser.add_argument("--context_w", type=float, default=1.5, help="Path to the dataset root directory.")
    # parser.add_argument("--agent_config_string", type=str, help="Path to the dataset root directory.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    visualize_outputs(args)

"""
export CUDA_VISIBLE_DEVICES=2
export PYTHONPATH="/home/kylehatch/Desktop/hidql/calvin-sim/external/susie:$PYTHONPATH"
export PYTHONPATH="/home/kylehatch/Desktop/hidql/calvin-sim/external/jaxrl_m:$PYTHONPATH"
export PYTHONPATH="/home/kylehatch/Desktop/hidql/calvin-sim/external/urdfpy:$PYTHONPATH"
export PYTHONPATH="/home/kylehatch/Desktop/hidql/calvin-sim/external/networkx:$PYTHONPATH"

# export DIFFUSION_MODEL_CHECKPOINT=/home/kylehatch/Desktop/hidql/susie-calvin-checkpoints/susie_test/test1_400smthlibs2_2024.02.23_14.20.24/40000/params_ema
export DIFFUSION_MODEL_CHECKPOINT=/home/kylehatch/Desktop/hidql/susie-calvin-checkpoints/susie_test/test1_sagemaker400lbs2currdrop0_2_2024.04.29_01.00.04/40000/params_ema
# export DIFFUSION_MODEL_CHECKPOINT=/home/kylehatch/Desktop/hidql/susie-calvin-checkpoints/susie_test/test1_sagemaker400lbs2promptdrop0_2_2024.04.29_00.59.42/40000/params_ema
# export DIFFUSION_MODEL_CHECKPOINT=/home/kylehatch/Desktop/hidql/susie-calvin-checkpoints/susie_test/public_model/checkpoint_only/params_ema
python3 -u visualize_high_level_policy.py \
--ep_dir /home/kylehatch/Desktop/hidql/saved_calvin_eval_results/results/calvin/public_model/checkpoint_only/calvin/gcdiffusion/auggoaldiff/seed_0/20240227_194024/checkpoint_150000/chatgptdummy16samples2024.04.25_14.04.28/ep203 \
--desc currdrop0_2 \
--prompt_w 1 \
--context_w 1 \
--slice 5



/home/kylehatch/Desktop/hidql/saved_calvin_eval_results/results/calvin/public_model/checkpoint_only/calvin/gcdiffusion/auggoaldiff/seed_0/20240227_194024/checkpoint_150000/chatgptdummy16samples2024.04.25_14.04.28/ep91

"""