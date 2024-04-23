import sys
from os import path
import argparse
import os
import h5py
import numpy as np
import imageio
import json
from tqdm import tqdm
from libero.scripts import init_path
import chiliocosm.utils.utils as chiliocosm_utils
from chiliocosm.envs import *
from libero_extension.envs import *


def rollout_demo(problem_name, env_kwargs, demos, demo_index, f, args, init_idx, cap_idx):
    env = TASK_MAPPING[problem_name](
        **env_kwargs,
    )
    ep = demos[demo_index]
    print(f"Playing back {ep}... (press ESC to quit)")
    states = f["data/{}/states".format(ep)][()]
    initial_state = dict(states=states[0])
    initial_state["model"] = f["data/{}".format(ep)].attrs["model_file"]
    # extract obs, rewards, dones
    actions = f["data/{}/actions".format(ep)][()]
    model_xml = f["data/{}".format(ep)].attrs["model_file"]
    reset_success = False
    while not reset_success:
        try:
            env.reset()
            reset_success = True
        except:
            continue
    model_xml = chiliocosm_utils.postprocess_model_xml(model_xml, {})
    libero_dev_path = os.path.abspath("libero")
    model_xml = model_xml.replace("/Users/yifengz/workspace/libero-dev", libero_dev_path)
    # load the flattened mujoco states
    states = f["data/{}/states".format(ep)][()]
    actions = np.array(f["data/{}/actions".format(ep)][()])
    num_actions = actions.shape[0]
    env.sim.reset()
    env.sim.set_state_from_flattened(states[init_idx])
    env.sim.forward()
    model_xml = env.sim.model.get_xml()
    ee_states = []
    gripper_states = []
    joint_states = []
    robot_states = []
    agentview_images = []
    eye_in_hand_images = []
    rewards = []
    dones = []
    valid_index = []
    obj_states = {}
    for j, action in tqdm(enumerate(actions), total=len(actions)):
        obs, reward, done, info = env.step(action)
        if j < cap_idx:
            continue
        # reset state rather than rollout actions
        if j == len(actions) - 1:
            obs, reward, done, info = env.step(action)
        else:
            env.reset_from_xml_string(model_xml)
            env.sim.reset()
            env.sim.set_state_from_flattened(states[j+1])
            env.sim.forward()
            env._update_observables(force=True)
            obs = env._get_observations()
        if j < num_actions - 1:
            # ensure that the actions deterministically lead to the same recorded states
            state_playback = env.sim.get_state().flatten()
            err = np.linalg.norm(states[j + 1] - state_playback)
            if err > 0.01:
                print(f"[warning] playback diverged by {err:.2f} for ep {ep} at step {j}")
        valid_index.append(j)
        robot_states.append(env.get_robot_state_vector(obs))
        agentview_images.append(obs["agentview_image"])
        eye_in_hand_images.append(obs["robot0_eye_in_hand_image"])
        # visualize the img obs
        if demo_index == 0:
            problem_info = json.loads(f["data"].attrs["problem_info"])
            problem_instruction = problem_info["language_instruction"].replace(" ", "_")
            vis_dir = os.path.join(args.visualization_folder, problem_instruction, ep)
            os.makedirs(vis_dir, exist_ok=True)
            vis_images = np.concatenate([obs["agentview_image"][::-1], obs["robot0_eye_in_hand_image"][::-1]], axis=1)
            # vis_images = np.concatenate([obs["agentview_image"][::-1]], axis=1)
            imageio.imsave(os.path.join(vis_dir, f"{str(j).zfill(5)}.jpg"), vis_images)
    # end of one trajectory
    states = states[valid_index]
    actions = actions[valid_index]
    dones = np.zeros(len(actions)).astype(np.uint8)
    dones[-1] = 1
    rewards = np.zeros(len(actions)).astype(np.uint8)
    rewards[-1] = 1
    print(len(actions) ,len(agentview_images))
    assert(len(actions) == len(agentview_images))
    env.close()
    # concatenate obj states:
    if len(obj_states.keys()) > 0:
        for key in obj_states.keys():
            obj_states[key] = np.stack(obj_states[key], axis=0)
    return {
        "gripper_states": np.stack(gripper_states, axis=0),
        "joint_states": np.stack(joint_states, axis=0),
        "ee_states": np.stack(ee_states, axis=0),
        "ee_pos": np.stack(ee_states, axis=0)[:, :3],
        "ee_quat": np.stack(ee_states, axis=0)[:, 3:],
        "agentview_rgb": np.stack(agentview_images, axis=0),
        "eye_in_hand_rgb": np.stack(eye_in_hand_images, axis=0),
        "actions": actions,
        "states": states,
        "robot_states": np.stack(robot_states, axis=0),
        "obj_states": obj_states,
        "rewards": rewards,
        "dones": dones,
        "model_xml": model_xml,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--demo-file',
        default="datasets/libero/libero_90/KITCHEN_SCENE1_open_the_bottom_drawer_of_the_cabinet_demo.hdf5"
    )
    parser.add_argument(
        '--visualization_folder',
        type=str,
        default="vis_libero_tabletop_manipulation",
    )
    args = parser.parse_args()
    hdf5_path = args.demo_file
    f = h5py.File(hdf5_path, "r")
    if "env_name" in f["data"].attrs:
        env_name = f["data"].attrs["env_name"]  # generated demos
        cap_index = 0
    if "env_args" in f["data"].attrs:
        env_args = f["data"].attrs["env_args"]
        env_kwargs = json.loads(env_args)['env_kwargs']
    problem_info = json.loads(f["data"].attrs["problem_info"])
    problem_name = problem_info["problem_name"]
    # list of all demonstrations episodes
    demos = list(f["data"].keys())
    bddl_file_name = f["data"].attrs["bddl_file_name"]
    if not os.path.exists(bddl_file_name):
        task_group = hdf5_path.split("/")[-2]
        hdf5_file_name = hdf5_path.split("/")[-1]
        bddl_file_name = os.path.join("libero/chiliocosm/bddl_files", task_group, hdf5_file_name.replace("_demo.hdf5", ".bddl"))
    chiliocosm_utils.update_env_kwargs(env_kwargs,
        bddl_file_name=bddl_file_name,
        has_renderer=False,
        has_offscreen_renderer=True,
        ignore_done=True,
        use_camera_obs=True,
        camera_depths=False,
        camera_names=["robot0_eye_in_hand",
                    "agentview",
                    ],
        reward_shaping=True,
        control_freq=20,
        camera_heights=128,
        camera_widths=128,
        camera_segmentations=None,
    )
    problem_name = problem_name
    env_args = {"type": 1,
                "env_name": env_name,
                "problem_name": problem_name,
                "bddl_file": bddl_file_name,
                "env_kwargs": env_kwargs}
    init_idx = 0
    demo_obs = rollout_demo(problem_name=problem_name,
                                env_kwargs=env_kwargs,
                                demos=demos,
                                demo_index=0,   # modify here is you'd like to replay a different demo
                                f=f,
                                args=args,
                                init_idx=init_idx,
                                cap_idx=cap_index,)
    return demo_obs

if __name__ == "__main__":
    main()
