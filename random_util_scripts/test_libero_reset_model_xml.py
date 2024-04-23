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
import json 


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

for i in range(10):
    demo_id = 0
    demo = f"demo_{demo_id}"

    with h5py.File(hdf5_file_path, 'r') as f:
        # env.set_init_state(init_states[demo_id])
        # env.reset()


        # env_name = f["data"].attrs["env"]
        # env_args = f["data"].attrs["env_info"] ### MISSING
        # env_kwargs = json.loads(f["data"].attrs["env_info"]) ### MISSING

        # problem_info = json.loads(f["data"].attrs["problem_info"])
        # problem_info["domain_name"]
        # problem_name = problem_info["problem_name"]
        # language_instruction = problem_info["language_instruction"]


        
        reset_success = False
        while not reset_success:
            try:
                env.reset()
                reset_success = True
            except:
                continue

        model_xml = f["data/{}".format(demo)].attrs["model_file"]
        model_xml = libero_utils.postprocess_model_xml(model_xml, {})

        # load the flattened mujoco states
        states = f["data/{}/states".format(demo)][()]

        init_idx = 0
        model_xml = model_xml.replace("/home/yifengz/workspace/libero-dev/chiliocosm", "/home/kylehatch/Desktop/hidql/calvin-sim/external/LIBERO/libero/libero")
        env.reset_from_xml_string(model_xml)  ### MISSING
        env.sim.reset()
        env.sim.set_state_from_flattened(states[init_idx])

        env.sim.forward()

        dummy_action = [0.] * 7
        for step in range(10):
            obs, _, _, _= env.step(dummy_action)

        img = obs["agentview_image"]
        img = np.flip(img, axis=0)
        img = cv2.resize(img, (200, 200))

        image_file = f"test_reset_images/img_{i}.png"
        os.makedirs(os.path.dirname(image_file), exist_ok=True)
        cv2.imwrite(image_file, img[..., ::-1]) 

