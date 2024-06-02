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
    

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_color = (0, 255, 0)  # White color in BGR
    line_type = 2  # Line thickness
    
    agent_config_string = args.agent_config_string
     
    agent_config, env_name, agent_type, _ = get_config(agent_config_string)

    agent_config.dataset_kwargs.use_generated_goals = False 
    agent_config.dataset_kwargs.use_encode_decode_goals = False 
    agent_config.dataset_kwargs.goal_relabeling_strategy = "delta_goals"
    agent_config.dataset_kwargs.goal_relabeling_kwargs = {"goal_delta": (16, 24)}

    

    vf_agent = diffusion_gc_policy.GCPolicy(agent_config, env_name, agent_type, os.getenv("GC_POLICY_CHECKPOINT"), normalize_actions=False, use_temporal_ensembling=False, text_processor=None)

    
    timestep_dirs = sorted(glob(os.path.join(args.ep_dir, "unfiltered_goal_images", "timestep_*")))

    if args.wrong_ep_dir is not None:
        wrong_timestep_dirs = sorted(glob(os.path.join(args.wrong_ep_dir, "unfiltered_goal_images", "timestep_*")))

        assert len(wrong_timestep_dirs) >= len(timestep_dirs), f"len(timestep_dirs): {len(timestep_dirs)}, len(wrong_timestep_dirs): {len(wrong_timestep_dirs)}"


    rgb_obses = []
    frames = []
    # for timestep_idx, timestep_dir in enumerate(timestep_dirs):
    for timestep_idx in range(len(timestep_dirs) - 1):

        timestep_dir = timestep_dirs[timestep_idx]

        goal_image_types = []
        goal_images = []

        if not args.real_only:
            goal_image_files = sorted(glob(os.path.join(timestep_dir, "goal_image_*.png")))
            for i, goal_image_file in enumerate(goal_image_files):
                
                goal_image = cv2.imread(goal_image_file)
                goal_image = goal_image[..., ::-1]
                # goal_image = np.stack([goal_image], axis=0)[0] # Have to do this to get around the cv2 error for some reason 
                # goal_image = cv2.rectangle(goal_image, (0, 0), (goal_image.shape[1] - 1, goal_image.shape[0] - 1), (0, 255, 0), 10)
                goal_images.append(goal_image)
                goal_image_types.append("generated")

            if args.wrong_ep_dir is not None:
                wrong_timestep_dir = wrong_timestep_dirs[timestep_idx]
                wrong_goal_image_files = sorted(glob(os.path.join(wrong_timestep_dir, "goal_image_*.png")))
                for i, wrong_goal_image_file in enumerate(wrong_goal_image_files):
                    
                    wrong_goal_image = cv2.imread(wrong_goal_image_file)
                    wrong_goal_image = wrong_goal_image[..., ::-1]
                    # wrong_goal_image = np.stack([wrong_goal_image], axis=0)[0] # Have to do this to get around the cv2 error for some reason 
                    # wrong_goal_image = cv2.rectangle(wrong_goal_image, (0, 0), (wrong_goal_image.shape[1] - 1, wrong_goal_image.shape[0] - 1), (255, 0, 0), 10)
                    goal_images.append(wrong_goal_image)
                    goal_image_types.append("wrong_generated")

        real_goal_image = cv2.imread(os.path.join(timestep_dirs[timestep_idx + 1], "rgb_obs.png"))
        real_goal_image = real_goal_image[..., ::-1]
        real_goal_image = np.stack([real_goal_image], axis=0)[0] # Have to do this to get around the cv2 error for some reason 
        # real_goal_image = cv2.rectangle(real_goal_image, (0, 0), (real_goal_image.shape[1] - 1, real_goal_image.shape[0] - 1), (0, 255, 0), 10)
        # real_goal_image = cv2.rectangle(real_goal_image, (10, 10), (real_goal_image.shape[1] - 11, real_goal_image.shape[0] - 11), (0, 0, 255), 3)
        goal_images.append(real_goal_image)
        goal_image_types.append("real")

        if args.wrong_ep_dir is not None:
            wrong_real_goal_image = cv2.imread(os.path.join(wrong_timestep_dirs[timestep_idx + 1], "rgb_obs.png"))
            wrong_real_goal_image = wrong_real_goal_image[..., ::-1]
            wrong_real_goal_image = np.stack([wrong_real_goal_image], axis=0)[0] # Have to do this to get around the cv2 error for some reason 
            # wrong_real_goal_image = cv2.rectangle(wrong_real_goal_image, (0, 0), (wrong_real_goal_image.shape[1] - 1, wrong_real_goal_image.shape[0] - 1), (255, 0, 0), 10)
            # wrong_real_goal_image = cv2.rectangle(wrong_real_goal_image, (10, 10), (wrong_real_goal_image.shape[1] - 11, wrong_real_goal_image.shape[0] - 11), (0, 0, 255), 3)
            goal_images.append(wrong_real_goal_image)
            goal_image_types.append("wrong_real")

        goal_images = np.stack(goal_images, axis=0)
        goal_image_types = np.array(goal_image_types)


        rgb_image_obs = cv2.imread(os.path.join(timestep_dir, "rgb_obs.png"))
        rgb_image_obs = rgb_image_obs[..., ::-1]

        rgb_obses.append(rgb_image_obs)
        

        with open(os.path.join(os.path.dirname(os.path.dirname(timestep_dir)), "language_task.txt"), "r") as f:
            _ = f.readline()
            language_goal = f.readline().strip()

        
        v = vf_agent.value_function_ranking(rgb_image_obs, goal_images)

        sorted_idxs = np.argsort(v)[::-1]

        ordered_goal_images = goal_images[sorted_idxs]
        ordered_vs = v[sorted_idxs]
        ordered_goal_image_types = goal_image_types[sorted_idxs]

        frame = []
        for i in range(ordered_vs.shape[0]):
            img = ordered_goal_images[i]
            val = ordered_vs[i]

            goal_type = ordered_goal_image_types[i]
            if goal_type == "real":
                img = cv2.rectangle(img, (0, 0), (img.shape[1] - 1, img.shape[0] - 1), (0, 255, 0), 10) 
                img = cv2.rectangle(img, (8, 8), (img.shape[1] - 9, img.shape[0] - 9), (0, 0, 255), 3) 
            elif goal_type == "generated":
                img = cv2.rectangle(img, (0, 0), (img.shape[1] - 1, img.shape[0] - 1), (0, 255, 0), 10) 
            elif goal_type == "wrong_generated":
                img = cv2.rectangle(img, (0, 0), (img.shape[1] - 1, img.shape[0] - 1), (255, 0, 0), 10)  
            elif goal_type == "wrong_real":
                img = cv2.rectangle(img, (0, 0), (img.shape[1] - 1, img.shape[0] - 1), (255, 0, 0), 10)  
                img = cv2.rectangle(img, (8, 8), (img.shape[1] - 9, img.shape[0] - 9), (0, 0, 255), 3)  
            else:
                raise ValueError(f"Unknown goal type: {goal_type}.")

            img = cv2.putText(img, f'{val:.3f}', (10, 15), font, font_scale, font_color, line_type)
            frame.append(img)

        # frame = np.concatenate(frame, axis=1)

        # assert len(frame) % 4 == 0, f"len(frame): {len(frame)}"
        # frame_rows = []
        # for row_idx in range(len(frame) // 4):
        #     start = row_idx * 4
        #     end = start + 4
        #     frame_row = np.concatenate([rgb_image_obs] + frame[start:end], axis=1)
        #     frame_rows.append(frame_row)
            

        if args.wrong_ep_dir is None:
            row_size = 5
        else:
            row_size = 10

        if args.real_only:
            row_size = 1

        # assert len(frame) % row_size == 0, f"len(frame): {len(frame)}"
        if len(frame) % row_size != 0:
            remainder = (((len(frame) // row_size) + 1) * row_size) - len(frame)
            for _ in range(remainder):
                frame.append(np.zeros_like(rgb_image_obs))

        frame_rows = []
        for row_idx in range(len(frame) // row_size):
            start = row_idx * row_size
            end = start + row_size
            frame_row = np.concatenate([rgb_image_obs] + frame[start:end], axis=1)
            frame_rows.append(frame_row)

        query_frame = np.concatenate(frame_rows, axis=0)

        frames.append(query_frame)

    save_video("./random_util_scripts/goal_images_vranked/traj.mp4", np.array(frames))
    

    rgb_image_obs = cv2.imread(os.path.join(timestep_dirs[-1], "rgb_obs.png"))
    rgb_image_obs = rgb_image_obs[..., ::-1]

    rgb_obses.append(rgb_image_obs)

    rgb_obses = np.stack(rgb_obses, axis=0)
    last_img_goals = np.stack(rgb_obses[-1] for _ in range(rgb_obses.shape[0]))

    v = np.array(vf_agent.agent.value_function({"image" : rgb_obses}, {"image" : last_img_goals}).tolist())

    frames = []
    for i in range(v.shape[0]):
        img = np.concatenate([rgb_obses[i], last_img_goals[i]], axis=0)
        img = cv2.putText(img, f'{v[i]:.3f}', (10, 15), font, font_scale, font_color, line_type)
        frames.append(img)
    frames = np.concatenate(frames, axis=1)
    cv2.imwrite("./random_util_scripts/goal_images_vranked/v_progression_last.png", frames[..., ::-1])

    curr_img_goals = rgb_obses.copy()

    v = np.array(vf_agent.agent.value_function({"image" : rgb_obses}, {"image" : curr_img_goals}).tolist())

    frames = []
    for i in range(v.shape[0]):
        img = np.concatenate([rgb_obses[i], curr_img_goals[i]], axis=0)
        img = cv2.putText(img, f'{v[i]:.3f}', (10, 15), font, font_scale, font_color, line_type)
        frames.append(img)
    frames = np.concatenate(frames, axis=1)
    cv2.imwrite("./random_util_scripts/goal_images_vranked/v_progression_same.png", frames[..., ::-1])


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
    parser.add_argument("--wrong_ep_dir", type=str, help="Path to the dataset root directory.")
    parser.add_argument("--real_only", action="store_true", help="Path to the dataset root directory.")
    parser.add_argument("--agent_config_string", type=str, default=None, help="Path to the dataset root directory.")
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
python3 -u visualize_value_fn_rankings_contrastive_vf.py \
--agent_config_string calvin_contrastivevf_noactnorm-auggoaldiff-b1024 \
--ep_dir /home/kylehatch/Desktop/hidql/saved_calvin_eval_results/results/calvin/public_model/checkpoint_only/calvin/gcdiffusion/auggoaldiff/seed_0/20240227_194024/checkpoint_150000/chatgptdummy16samples2024.04.25_14.04.28/ep0 \
--wrong_ep_dir /home/kylehatch/Desktop/hidql/saved_calvin_eval_results/results/calvin/public_model/checkpoint_only/calvin/gcdiffusion/auggoaldiff/seed_0/20240227_194024/checkpoint_150000/chatgptdummy16samples2024.04.25_14.04.28/ep91

python3 -u visualize_value_fn_rankings_contrastive_vf.py \
--agent_config_string calvin_contrastivevf_noactnorm-auggoaldiff-b1024 \
--ep_dir /home/kylehatch/Desktop/hidql/saved_calvin_eval_results/results/calvin/public_model/checkpoint_only/calvin/gcdiffusion/auggoaldiff/seed_0/20240227_194024/checkpoint_150000/chatgptdummy16samples2024.04.25_14.04.28/ep119


python3 -u visualize_value_fn_rankings_contrastive_vf.py \
--agent_config_string calvin_contrastivevf_noactnorm-auggoaldiff-b1024 \
--ep_dir /home/kylehatch/Desktop/hidql/saved_calvin_eval_results/results/calvin/public_model/checkpoint_only/calvin/gcdiffusion/auggoaldiff/seed_0/20240227_194024/checkpoint_150000/chatgptdummy16samples2024.04.25_14.04.28/ep353


python3 -u visualize_value_fn_rankings_contrastive_vf.py \
--agent_config_string calvin_contrastivevf_noactnorm-auggoaldiff-b1024 \
--ep_dir /home/kylehatch/Desktop/hidql/saved_calvin_eval_results/results/calvin/public_model/checkpoint_only/calvin/gcdiffusion/auggoaldiff/seed_0/20240227_194024/checkpoint_150000/chatgptdummy16samples2024.04.25_14.04.28/ep359 \
--wrong_ep_dir /home/kylehatch/Desktop/hidql/saved_calvin_eval_results/results/calvin/public_model/checkpoint_only/calvin/gcdiffusion/auggoaldiff/seed_0/20240227_194024/checkpoint_150000/chatgptdummy16samples2024.04.25_14.04.28/ep353 \
--real_only


python3 -u visualize_value_fn_rankings_contrastive_vf.py \
--agent_config_string calvin_contrastivevf_noactnorm-auggoaldiff-b1024 \
--ep_dir /home/kylehatch/Desktop/hidql/saved_calvin_eval_results/results/calvin/public_model/checkpoint_only/calvin/gcdiffusion/auggoaldiff/seed_0/20240227_194024/checkpoint_150000/chatgptdummy16samples2024.04.25_14.04.28/ep349 \
--wrong_ep_dir /home/kylehatch/Desktop/hidql/saved_calvin_eval_results/results/calvin/public_model/checkpoint_only/calvin/gcdiffusion/auggoaldiff/seed_0/20240227_194024/checkpoint_150000/chatgptdummy16samples2024.04.25_14.04.28/ep351 \
--real_only



python3 -u visualize_value_fn_rankings_contrastive_vf.py \
--agent_config_string calvin_contrastivevf_noactnorm-auggoaldiff-b1024 \
--ep_dir /home/kylehatch/Desktop/hidql/saved_calvin_eval_results/results/calvin/public_model/checkpoint_only/calvin/gcdiffusion/auggoaldiff/seed_0/20240227_194024/checkpoint_150000/chatgptdummy16samples2024.04.25_14.04.28/ep399 \
--wrong_ep_dir /home/kylehatch/Desktop/hidql/saved_calvin_eval_results/results/liberosplit2/test1_400smthlibs2_2024.02.23_14.20.24/40000/liberosplit2/gcdiffusion/auggoaldiff/seed_0/20240228_024329/checkpoint_150000/chatgptdummy16slibs22024.04.26_12.27.01/ep0 \
--real_only







Take a batch loaded with the dataset, see if the outputs for true vs false are reasonable there. Visualize the batch and such. 
- visualize it in the bridge repo, load the checkpoint with that script, and then just print things out in the training script and save things. 
    - use a batch size of like 4 
    - use both get debug metrics, and the value_fn function I wrote. Make sure they match up. 
    - do both train and val set 

- Then, do this from a traj from the dataset, and then also switch out the end goal images w one from a different trajectory 

- if it's still doing nonesense, maybe compare to a contrastive RL checkpoint w actions? 

"""