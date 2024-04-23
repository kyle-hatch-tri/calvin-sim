import os
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv
from libero.libero.utils import get_libero_path
import libero.libero.utils.utils as libero_utils

import h5py
import cv2
import numpy as np 
import os
from glob import glob
from tqdm import tqdm, trange


def save_numpy_array_as_video(numpy_array, output_path, fps=30):
    # Ensure the array has the correct shape and dtype
    if numpy_array.shape[3] != 3 or numpy_array.dtype != np.uint8:
        raise ValueError("Invalid array shape or dtype. Expected shape (N, H, W, 3) and dtype np.uint8.")

    # Get video dimensions from the array shape
    height, width = numpy_array.shape[1], numpy_array.shape[2]

    # Create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # You can choose other codecs based on your requirements
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    try:
        # Loop through the frames and write them to the video
        for frame in numpy_array:
            video_writer.write(frame)

    except Exception as e:
        print(f"Error writing video frames: {e}")

    finally:
        # Release the VideoWriter object
        video_writer.release()




hdf5_file_path = "/home/kylehatch/Desktop/hidql/data/libero_data/libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo.hdf5"
task_suite_name = hdf5_file_path.split("/")[-2]

benchmark_dict = benchmark.get_benchmark_dict()
task_suite = benchmark_dict[task_suite_name]()
task_names = [task.name for task in task_suite.tasks]

task_name = hdf5_file_path.split("/")[-1].split(".")[0]
if "_demo" in task_name:
    task_name = task_name.split("_demo")[0]

assert task_name in task_names, f"\"{task_name}\" not in task_names: {task_names}"


task_id = task_names.index(task_name)
task = task_suite.get_task(task_id)
assert task.name == task_name, f"task.name: {task.name} != task_name: {task_name}"

init_states = task_suite.get_task_init_states(task_id)

task_description = task.language
task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
print(f"[info] retrieving task {task_id} from suite {task_suite_name}, the " + f"language instruction is {task_description}, and the bddl file is {task_bddl_file}")


env_args = {
    "bddl_file_name": task_bddl_file,
    "camera_heights": 128,
    "camera_widths": 128,
    "control_freq":20,
}  

env = OffScreenRenderEnv(**env_args)
env.seed(0)

with h5py.File(hdf5_file_path, 'r') as hdf5_file:  
    demo_names = hdf5_file["data"].keys()
    assert len(demo_names) == init_states.shape[0], f"len(demo_names): {len(demo_names)}, init_states.shape: {init_states.shape}"
    
    demo_id = 0
    demo = f"demo_{demo_id}"

    agentview_rgb = hdf5_file["data"][demo]["obs"]["agentview_rgb"][:]
    actions = hdf5_file["data"][demo]["actions"][:]
    states = hdf5_file["data"][demo]["states"]

    env.set_init_state(init_states[demo_id])
    env.reset()

    # model_xml = hdf5_file["data/{}".format(demo)].attrs["model_file"]
    # reset_success = False
    # while not reset_success:
    #     try:
    #         env.reset()
    #         reset_success = True
    #     except:
    #         continue

    # model_xml = libero_utils.postprocess_model_xml(model_xml, {})

    # # load the flattened mujoco states
    # states_ = hdf5_file["data/{}/states".format(demo)][()]

    # init_idx = 0
    # model_xml = model_xml.replace("/home/yifengz/workspace/libero-dev/chiliocosm", "/home/kylehatch/Desktop/hidql/calvin-sim/external/LIBERO/libero/libero")
    # env.reset_from_xml_string(model_xml)  ### MISSING
    # env.sim.reset()
    # env.sim.set_state_from_flattened(states_[init_idx])
    # env.sim.forward()

    # Without this, some of the objects start off in the air and then fall to the table
    dummy_action = [0.] * 7
    for step in range(10):
        env.step(dummy_action)
    
    agentview_rgb_rerendered = []

    for t, action in enumerate(tqdm(actions, disable=False)):
        # obs, reward, done, info = env.step(action)
        obs = env.regenerate_obs_from_state(states[t])
        agentview_rgb_rerendered.append(obs["agentview_image"])
        state_playback = env.sim.get_state().flatten()


        if t < actions.shape[0] - 1:
            # err = np.linalg.norm(states[t + 1] - state_playback)
            err = np.linalg.norm(states[t] - state_playback)
            if err > 0.01:
                print(f"[{t}] [warning] playback diverged by {err:.2f} for {demo} at step {t}")

    agentview_rgb_rerendered = np.stack(agentview_rgb_rerendered, axis=0)

    video_array = np.concatenate([agentview_rgb, agentview_rgb_rerendered], axis=2)
    video_array = np.flip(video_array, axis=1)
    video_array = video_array[..., ::-1]

    output_video_path = os.path.join("./data_visualizations", hdf5_file_path.split("/")[-2], hdf5_file_path.split("/")[-1].split(".")[0], f"{demo}.avi")
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
    save_numpy_array_as_video(video_array, output_video_path, fps=30)
    print(f"Saved video to \"{output_video_path}\".")
env.close()