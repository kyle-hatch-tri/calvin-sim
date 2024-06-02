import tensorflow as tf
import numpy as np
import cv2
import os

from jaxrl_m.data.text_processing import text_processors

from calvin_agent.evaluation.gcbc_train_config import get_config
from calvin_agent.evaluation import diffusion_gc_policy

from matplotlib import pyplot as plt 
from io import BytesIO
from PIL import Image



def visualize_vf(args):
    os.chdir("..")

    # Choose font and scale
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_color = (0, 255, 0)  # White color in BGR
    line_type = 2  # Line thickness
    
     # file_path = "/home/kylehatch/Desktop/hidql/data/calvin_data_processed/language_conditioned_16_samples/training/A/traj0.tfrecord"
    # file_path = "/home/kylehatch/Desktop/hidql/data/calvin_data_processed/language_conditioned_4_samples_encodedecode_noisedencodedecode/validation/D/traj100.tfrecord"
    
    assert args.end_idx > args.start_idx, f"args.start_idx: {args.start_idx}, args.end_idx: {args.end_idx}"

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

    
    
    loaded_data = load_tfrecord_file(args.file_path)
    
    # Example: Iterate through the loaded data
    for _, record in enumerate(loaded_data):
        # Access your features here, e.g., record['feature1'], record['feature2']
        
        image = record["observations"]["image"]
        image = np.array(image)
        rgb_image_obs = image[args.start_idx:args.end_idx + 1:args.step_rate]

        goal_image = image[args.end_idx]
        goal_images = np.stack([goal_image for _ in range(rgb_image_obs.shape[0])], axis=0)

        v = vf_agent.agent.value_function({"image" : rgb_image_obs}, {"image" : goal_images})
        frames = np.concatenate([rgb_image_obs, goal_images], axis=2)

        
        new_frames = []
        for i, frame in enumerate(frames):
            

            # Add text to the image
            frame = cv2.putText(frame, f'{v[i]:.2f}', (10, 15), font, font_scale, font_color, line_type)

            # Plot the array
            plt.figure(figsize=(4, 2))  # Set figure size
            plt.plot(np.arange(image.shape[0])[args.start_idx:args.end_idx + 1:args.step_rate], v)  # Plot the data
            plt.title(f'tfrecord_vf{args.start_idx}_{args.end_idx + 1}_step_{args.step_rate}')  # Add a title
            plt.xlabel('Index')  # Label for x-axis
            plt.ylabel('Vf')  # Label for y-axis
            plt.ylim(np.minimum(v.min() - 2, -1), v.max() + 2)
            # plt.ylim(np.minimum(-25, -1), -5)

            plt.axvline(x=i * args.step_rate, color='black', linewidth=2)  # Black vertical line at x = 5
            

            # Save the plot to a BytesIO object in memory (as a PNG file)
            buf = BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)  # Move the reading cursor to the start of the buffer

            # Convert the buffer into a PIL image
            plot_image = Image.open(buf)
            plot_image = np.array(plot_image)

            frame = np.concatenate([frame, plot_image[..., :-1]], axis=0)
            new_frames.append(frame)

        break 


    if args.wrong_file_path is not None:
        wrong_loaded_data = load_tfrecord_file(args.wrong_file_path)
        for _, wrong_record in enumerate(wrong_loaded_data):
            wrong_image = wrong_record["observations"]["image"]
            wrong_image = np.array(wrong_image)

            # wrong_goal_image = wrong_image[args.end_idx]
            wrong_goal_image = wrong_image[-1]
            wrong_goal_images = np.stack([wrong_goal_image for _ in range(rgb_image_obs.shape[0])], axis=0)

            wrong_v = vf_agent.agent.value_function({"image" : rgb_image_obs}, {"image" : wrong_goal_images})
            wrong_frames = np.concatenate([rgb_image_obs, wrong_goal_images], axis=2)

            

            
            wrong_new_frames = []
            for i, wrong_frame in enumerate(wrong_frames):
                

                # Add text to the image
                wrong_frame = cv2.putText(wrong_frame, f'{wrong_v[i]:.2f}', (10, 15), font, font_scale, font_color, line_type)

                # Plot the array
                plt.figure(figsize=(4, 2))  # Set figure size
                plt.plot(np.arange(image.shape[0])[args.start_idx:args.end_idx + 1:args.step_rate], wrong_v)  # Plot the data
                plt.title(f'wrong_tfrecord_vf{args.start_idx}_{args.end_idx + 1}_step_{args.step_rate}')  # Add a title
                plt.xlabel('Index')  # Label for x-axis
                plt.ylabel('Vf')  # Label for y-axis
                plt.ylim(np.minimum(v.min() - 2, -1), v.max() + 2)
                # plt.ylim(np.minimum(-25, -1), -5)

                plt.axvline(x=i * args.step_rate, color='black', linewidth=2)  # Black vertical line at x = 5
                

                # Save the plot to a BytesIO object in memory (as a PNG file)
                buf = BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)  # Move the reading cursor to the start of the buffer

                # Convert the buffer into a PIL image
                wrong_plot_image = Image.open(buf)
                wrong_plot_image = np.array(wrong_plot_image)

                wrong_frame = np.concatenate([wrong_frame, wrong_plot_image[..., :-1]], axis=0)
                wrong_new_frames.append(wrong_frame)


            new_frames = np.concatenate([new_frames, wrong_new_frames], axis=2)
            break 
    # frames = np.flip(frames, axis=-1)
    save_video(f"./random_util_scripts/goal_images_vranked/tfrecord_vf_{args.desc}_{args.start_idx}_{args.end_idx + 1}_step_{args.step_rate}.mp4", np.array(new_frames))


PROTO_TYPE_SPEC = {
        "actions": tf.float32,
        "proprioceptive_states": tf.float32,
        "image_states": tf.uint8,
        # "generated_goals": tf.uint8,
        # "encoded_decoded": tf.uint8,
        # "noised_encoded_decoded": tf.uint8,
        # "language_annotation": tf.string, 
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
                # "encoded_decoded": parsed_tensors["encoded_decoded"][:-1],
                # "noised_encoded_decoded": parsed_tensors["noised_encoded_decoded"][:-1],
                "proprio": parsed_tensors["proprioceptive_states"][:-1],
            },
            "next_observations": {
                "image": parsed_tensors["image_states"][1:],
                # "generated_goals": parsed_tensors["generated_goals"][1:],
                # "encoded_decoded": parsed_tensors["encoded_decoded"][1:],
                # "noised_encoded_decoded": parsed_tensors["noised_encoded_decoded"][1:],
                "proprio": parsed_tensors["proprioceptive_states"][1:],
            },
            # **({"language": parsed_tensors["language_annotation"]} if True else {}),
            "actions": parsed_tensors["actions"][:-1],
            "terminals": tf.zeros_like(parsed_tensors["actions"][:-1][:, 0:1], dtype=tf.bool)
        }


def load_tfrecord_file(file_path):
    dataset = tf.data.TFRecordDataset(file_path)
    # parsed_dataset = dataset.map(parse_tfrecord)
    parsed_dataset = dataset.map(_decode_example)
    return parsed_dataset


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
    parser.add_argument("--file_path", type=str, help="Path to the dataset root directory.")
    parser.add_argument("--wrong_file_path", type=str, default=None, help="Path to the dataset root directory.")
    parser.add_argument("--desc", type=str, default="default", help="Path to the dataset root directory.")
    # parser.add_argument("--real_only", action="store_true", help="Path to the dataset root directory.")
    parser.add_argument("--agent_config_string", type=str, default=None, help="Path to the dataset root directory.")
    parser.add_argument("--start_idx", type=int, help="Path to the dataset root directory.")
    parser.add_argument("--end_idx", type=int, help="Path to the dataset root directory.")
    parser.add_argument("--step_rate", type=int, default=1, help="Path to the dataset root directory.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    visualize_vf(args)


"""
export CUDA_VISIBLE_DEVICES=0 
export PYTHONPATH="/home/kylehatch/Desktop/hidql/calvin-sim/external/susie:$PYTHONPATH"
export PYTHONPATH="/home/kylehatch/Desktop/hidql/calvin-sim/external/jaxrl_m:$PYTHONPATH"
export PYTHONPATH="/home/kylehatch/Desktop/hidql/calvin-sim/external/urdfpy:$PYTHONPATH"
export PYTHONPATH="/home/kylehatch/Desktop/hidql/calvin-sim/external/networkx:$PYTHONPATH"
# export GC_POLICY_CHECKPOINT=/home/kylehatch/Desktop/hidql/susie-calvin-checkpoints/susie_low_level/calvin/contrastivevf/auggoaldiff/seed_0/20240423_222107/checkpoint_100000
export GC_POLICY_CHECKPOINT=/home/kylehatch/Desktop/hidql/susie-calvin-checkpoints/susie_low_level/calvin/gciql/auggoaldiffhparams5/seed_0/20240505_020833/checkpoint_150000


python3 -u visualize_value_fn_rankings_contrastive_vf_tfrecord.py \
--agent_config_string calvin_contrastivevf_noactnorm-auggoaldiff-b1024 \
--file_path /home/kylehatch/Desktop/hidql/data/calvin_data_processed/goal_conditioned/training/A/traj0/0.tfrecord \
--start_idx 0 \
--end_idx 20 \
--step_rate 25


python3 -u visualize_value_fn_rankings_contrastive_vf_tfrecord.py \
--agent_config_string calvin_contrastivevf_noactnorm-auggoaldiff-b1024 \
--file_path /home/kylehatch/Desktop/hidql/data/calvin_data_processed/goal_conditioned/training/A/traj0/0.tfrecord \
--wrong_file_path /home/kylehatch/Desktop/hidql/data/calvin_data_processed/goal_conditioned/training/B/traj0/0.tfrecord \
--start_idx 0 \
--end_idx 500 \
--step_rate 50


python3 -u visualize_value_fn_rankings_contrastive_vf_tfrecord.py \
--agent_config_string calvin_gciql_noactnorm-hparams5-auggoaldiff \
--file_path /home/kylehatch/Desktop/hidql/data/calvin_data_processed/goal_conditioned/training/A/traj0/0.tfrecord \
--wrong_file_path /home/kylehatch/Desktop/hidql/data/calvin_data_processed/goal_conditioned/training/B/traj0/0.tfrecord \
--start_idx 0 \
--end_idx 50 \
--step_rate 1 \
--desc gciql

python3 -u visualize_value_fn_rankings_contrastive_vf_tfrecord.py \
--agent_config_string calvin_gciql_noactnorm-hparams5-auggoaldiff \
--file_path /home/kylehatch/Desktop/hidql/data/calvin_data_processed/goal_conditioned/training/A/traj0/0.tfrecord \
--wrong_file_path /home/kylehatch/Desktop/hidql/data/calvin_data_processed/goal_conditioned/training/B/traj0/0.tfrecord \
--start_idx 0 \
--end_idx 500 \
--step_rate 50 \
--desc gciql



python3 -u visualize_value_fn_rankings_contrastive_vf_tfrecord.py \
--agent_config_string calvin_gciql_noactnorm-hparams5-auggoaldiff \
--file_path /home/kylehatch/Desktop/hidql/data/calvin_data_processed/goal_conditioned/training/A/traj0/0.tfrecord \
--wrong_file_path /home/kylehatch/Desktop/hidql/data/calvin_data_processed/goal_conditioned/training/A/traj0/0.tfrecord \
--start_idx 0 \
--end_idx 500 \
--step_rate 50 \
--desc gciql_2


python3 -u visualize_value_fn_rankings_contrastive_vf_tfrecord.py \
--agent_config_string calvin_contrastivevf_noactnorm-auggoaldiff-b1024 \
--file_path /home/kylehatch/Desktop/hidql/data/calvin_data_processed/goal_conditioned/training/A/traj0/0.tfrecord \
--wrong_file_path /home/kylehatch/Desktop/hidql/data/calvin_data_processed/goal_conditioned/training/A/traj0/0.tfrecord \
--start_idx 0 \
--end_idx 500 \
--step_rate 50 \
--desc crlvf2

"""