
import tensorflow as tf
import numpy as np
import cv2
import os


PROTO_TYPE_SPEC = {
        "actions": tf.float32,
        "proprioceptive_states": tf.float32,
        "image_states": tf.uint8,
        "generated_goals": tf.uint8,
        "encoded_decoded": tf.uint8,
        "noised_encoded_decoded": tf.uint8,
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
                "generated_goals": parsed_tensors["generated_goals"][:-1],
                "encoded_decoded": parsed_tensors["encoded_decoded"][:-1],
                "noised_encoded_decoded": parsed_tensors["noised_encoded_decoded"][:-1],
                "proprio": parsed_tensors["proprioceptive_states"][:-1],
            },
            "next_observations": {
                "image": parsed_tensors["image_states"][1:],
                "generated_goals": parsed_tensors["generated_goals"][1:],
                "encoded_decoded": parsed_tensors["encoded_decoded"][1:],
                "noised_encoded_decoded": parsed_tensors["noised_encoded_decoded"][1:],
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
    file_path = "/home/kylehatch/Desktop/hidql/data/calvin_data_processed/language_conditioned_4_samples_encodedecode_noisedencodedecode/validation/D/traj100.tfrecord"
    # "/home/kylehatch/Desktop/hidql/data/calvin_data_processed/language_conditioned_16_samples/validation/D/traj234.tfrecord"
    loaded_data = load_tfrecord_file(file_path)
    
    # Example: Iterate through the loaded data
    for i, record in enumerate(loaded_data):
        # Access your features here, e.g., record['feature1'], record['feature2']
        
        image = record["observations"]["image"]
        generated_goals = record["observations"]["generated_goals"]
        encode_decode = record["observations"]["encoded_decoded"]
        noised_encode_decode = record["observations"]["noised_encoded_decoded"]

        image = np.array(image)
        generated_goals = np.array(generated_goals)
        encode_decode = np.array(encode_decode)
        noised_encode_decode = np.array(noised_encode_decode)
        

        image = np.concatenate([image for i in range(generated_goals.shape[1])], axis=2)
        generated_goals = np.concatenate([generated_goals[:, i] for i in range(generated_goals.shape[1])], axis=2)
        encode_decode = np.concatenate([encode_decode[:, i] for i in range(encode_decode.shape[1])], axis=2)
        noised_encode_decode = np.concatenate([noised_encode_decode[:, i] for i in range(noised_encode_decode.shape[1])], axis=2)

        import ipdb; ipdb.set_trace()


        frames = np.concatenate([image, generated_goals, encode_decode, noised_encode_decode], axis=1)
        # frames = np.flip(frames, axis=-1)
        save_video(f"generated_goals/4samples_noised_encode_decode/record_{i}.mp4", frames)
        print(f"record {i}. instruction: \"{record['language']}\".")