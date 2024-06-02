import tensorflow as tf
import numpy as np
import cv2
import os


PROTO_TYPE_SPEC = {
        "observations/images0": tf.uint8,
        "observations/state": tf.float32,
        "next_observations/images0": tf.uint8,
        "next_observations/state": tf.float32,
        "actions": tf.float32,
        "terminals": tf.bool,
        "truncates": tf.bool,
        "language": tf.string, 
    }

def _decode_example(example_proto, load_language=True):
    # decode the example proto according to PROTO_TYPE_SPEC
    features = {
        key: tf.io.FixedLenFeature([], tf.string)
        for key in PROTO_TYPE_SPEC.keys()
    }
    parsed_features = tf.io.parse_single_example(example_proto, features)
    parsed_tensors = {
        key: tf.io.parse_tensor(parsed_features[key], dtype)
        for key, dtype in PROTO_TYPE_SPEC.items()
    }
    # restructure the dictionary into the downstream format
    return {
        "observations": {
            "image": parsed_tensors["observations/images0"],
            "proprio": parsed_tensors["observations/state"],
        },
        "next_observations": {
            "image": parsed_tensors["next_observations/images0"],
            "proprio": parsed_tensors["next_observations/state"],
        },
        **({"language": parsed_tensors["language"][0]} if load_language else {}), ###
        "actions": parsed_tensors["actions"],
        # "terminals": parsed_tensors["terminals"],
        # "truncates": parsed_tensors["truncates"],
        "terminals": tf.expand_dims(parsed_tensors["terminals"], axis=-1), ###
        "truncates": tf.expand_dims(parsed_tensors["truncates"], axis=-1),
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

if __name__ == "__main__":
    # file_path = "/home/kylehatch/Desktop/hidql/data/calvin_data_processed/language_conditioned_16_samples/training/A/traj0.tfrecord"
    file_path = "/home/kylehatch/Desktop/hidql/data/bridgev2_processed/bridge_data_v1/berkeley/laundry_machine/put_clothes_in_laundry_machine/train/out.tfrecord"
    loaded_data = load_tfrecord_file(file_path)
    
    # Example: Iterate through the loaded data
    for i, record in enumerate(loaded_data):
        # Access your features here, e.g., record['feature1'], record['feature2']
        
        image = record["observations"]["image"]
        proprio = record["observations"]["proprio"]
        next_image = record["next_observations"]["image"]
        next_proprio = record["next_observations"]["proprio"]
        actions = record["actions"]
        terminals = record["terminals"]
        truncates = record["truncates"]

        print("image.shape:", image.shape)
        print("proprio.shape:", proprio.shape)
        print("next_image.shape:", next_image.shape)
        print("next_proprio.shape:", next_proprio.shape)
        print("actions.shape:", actions.shape)
        print("terminals.shape:", terminals.shape)
        print("truncates.shape:", truncates.shape)

        import ipdb; ipdb.set_trace()

        image = np.array(image)

        image = np.concatenate([image for i in range(generated_goals.shape[1])], axis=2)
        generated_goals = np.concatenate([generated_goals[:, i] for i in range(generated_goals.shape[1])], axis=2)
        encode_decode = np.concatenate([encode_decode[:, i] for i in range(encode_decode.shape[1])], axis=2)
        noised_encode_decode = np.concatenate([noised_encode_decode[:, i] for i in range(noised_encode_decode.shape[1])], axis=2)


        frames = np.concatenate([image, generated_goals, encode_decode, noised_encode_decode], axis=1)
        # frames = np.flip(frames, axis=-1)
        save_video(f"generated_goals/4samples_noised_encode_decode/record_{i}.mp4", frames)
        print(f"record {i}. instruction: \"{record['language']}\".")