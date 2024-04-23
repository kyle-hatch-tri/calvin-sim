import os 
import numpy as np 
import cv2
from tqdm import tqdm, trange

from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv
from libero.libero.utils import get_libero_path
from libero.libero.benchmark.libero_suite_task_map import libero_task_map



# task_suite_name = "libero_spatial"
# task_name = "pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate"

task_suite_name = "libero_10"
task_name = "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it"

init_demo_id = 0

hdf5_file_path = "/home/kylehatch/Desktop/hidql/data/libero_data/libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo.hdf5"

benchmark_dict = benchmark.get_benchmark_dict()
task_suite = benchmark_dict[task_suite_name]()
task_names = [task.name for task in task_suite.tasks]

task_id = task_names.index(task_name)
task = task_suite.get_task(task_id)
assert task.name == task_name, f"task.name: {task.name} != task_name: {task_name}"

init_states = task_suite.get_task_init_states(task_id)

task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)

env_args = {
    "bddl_file_name": task_bddl_file,
    "camera_heights": 128,
    "camera_widths": 128,
    "control_freq":20,
}  

env = OffScreenRenderEnv(**env_args)
env.seed(0)



for i in trange(10):


    x = env.set_init_state(init_states[init_demo_id])
    # odict_keys(['robot0_joint_pos', 'robot0_joint_pos_cos', 'robot0_joint_pos_sin', 'robot0_joint_vel', 'robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos', 'robot0_gripper_qvel', 'agentview_image', 'robot0_eye_in_hand_image', 'akita_black_bowl_1_pos', 'akita_black_bowl_1_quat', 'akita_black_bowl_1_to_robot0_eef_pos', 'akita_black_bowl_1_to_robot0_eef_quat', 'akita_black_bowl_2_pos', 'akita_black_bowl_2_quat', 'akita_black_bowl_2_to_robot0_eef_pos', 'akita_black_bowl_2_to_robot0_eef_quat', 'cookies_1_pos', 'cookies_1_quat', 'cookies_1_to_robot0_eef_pos', 'cookies_1_to_robot0_eef_quat', 'glazed_rim_porcelain_ramekin_1_pos', 'glazed_rim_porcelain_ramekin_1_quat', 'glazed_rim_porcelain_ramekin_1_to_robot0_eef_pos', 'glazed_rim_porcelain_ramekin_1_to_robot0_eef_quat', 'plate_1_pos', 'plate_1_quat', 'plate_1_to_robot0_eef_pos', 'plate_1_to_robot0_eef_quat', 'robot0_proprio-state', 'object-state'])
    
    
    # odict_keys(['akita_black_bowl_1_pos', 'akita_black_bowl_1_quat', 'akita_black_bowl_1_to_robot0_eef_pos', 'akita_black_bowl_1_to_robot0_eef_quat', 
    # 'akita_black_bowl_2_pos', 'akita_black_bowl_2_quat', 'akita_black_bowl_2_to_robot0_eef_pos', 'akita_black_bowl_2_to_robot0_eef_quat', 
    # 'cookies_1_pos', 'cookies_1_quat', 'cookies_1_to_robot0_eef_pos', 'cookies_1_to_robot0_eef_quat', 
    # 'glazed_rim_porcelain_ramekin_1_pos', 'glazed_rim_porcelain_ramekin_1_quat', 'glazed_rim_porcelain_ramekin_1_to_robot0_eef_pos', 'glazed_rim_porcelain_ramekin_1_to_robot0_eef_quat', 
    # 'plate_1_pos', 'plate_1_quat', 'plate_1_to_robot0_eef_pos', 'plate_1_to_robot0_eef_quat', 
    
    # 'robot0_proprio-state', 
    # 'object-state'])

    # akita_black_bowl_1
    # akita_black_bowl_2
    # cookies_1_pos
    # glazed_rim_porcelain_ramekin_1
    # plate_1


    # dict_keys(['plate_1', 'cookies_1', 'glazed_rim_porcelain_ramekin_1', 'wooden_cabinet_1', 'flat_stove_1', 'akita_black_bowl_2', 'akita_black_bowl_1'])


    # env.reset()
    reset_success = False
    while not reset_success:
        try:
            env.reset()
            reset_success = True
        except:
            continue

    # x = env.set_init_state(init_states[init_demo_id])

    dummy_action = [0.] * 7
    for step in range(10):
        obs, _, _, _= env.step(dummy_action)

    img = obs["agentview_image"]
    img = np.flip(img, axis=0)
    img = cv2.resize(img, (200, 200))

    image_file = f"test_reset_images/img_{i}.png"
    os.makedirs(os.path.dirname(image_file), exist_ok=True)
    cv2.imwrite(image_file, img[..., ::-1]) 



"""
Where I left off 

0) Look at the script that Bo (libero author) emailed me and see if it does anything useful or related to what I'm trying to do 
1) Check quick and dirty to see if the pickle reset works on one env
2) Then check to see if the model file exsits in the saved raw episodes somwhere. If so, spend some time seeing if I can 
deterministic reset using that. If so, then do that. Use both dataset oracle goals and replayed (retrospective real and generated) oracle goals.
If not, then just use the pickle method for eplayed (retrospective real and generated) oracle goals.

# Best way, probably 
Could just save out (pickle) the object_placements object in bddl_base_domain.py and then load it again 
- Would work for replaying generated goals or retrospective true goals, but not for using goals from the dataset 


Could alter the object property samplers and whatever to have a deterministic option 
- Might be hard? 
- still wouldn't help replaying from the dataset goals 


Could try to save out xml files and reset from there in the way done in create_dataset.py. Would also then need to see if env.sim.set_state_from_flattened(states[init_idx])
works using the saved initial states (seems like it might be the same thing?)
- Look at create_dataset.py 
- still wouldn't help replaying from the dataset goals?

- CHECK TO SEE IF the model_file is saved in the saved episodes somewhere. The raw libero data files. Looks like it is? 
"""