import argparse
from collections import Counter, defaultdict
import logging
import os
from pathlib import Path
import sys
import time
import requests
import json
import cv2
from tqdm import tqdm
# import jax_diffusion_model
# import pytorch_diffusion_model
# import diffusion_gc_policy

from calvin_agent.evaluation import jax_diffusion_model
from calvin_agent.evaluation import pytorch_diffusion_model
from calvin_agent.evaluation import diffusion_gc_policy

import datetime
from s3_save import S3SyncCallback
import random 


# This is for using the locally installed repo clone when using slurm
from calvin_agent.models.calvin_base_model import CalvinBaseModel

sys.path.insert(0, Path(__file__).absolute().parents[2].as_posix())

from calvin_agent.evaluation.multistep_sequences import get_sequences
from calvin_agent.evaluation.utils import (
    collect_plan,
    count_success,
    create_tsne,
    get_default_model_and_env,
    get_env_state_for_initial_condition,
    get_log_dir,
    join_vis_lang,
    print_and_save,
)
from calvin_agent.utils.utils import get_all_checkpoints, get_checkpoints_for_epochs, get_last_checkpoint
import hydra
import numpy as np
from omegaconf import OmegaConf
# from pytorch_lightning import seed_everything # UNCOMMENT
from termcolor import colored
import torch
from tqdm.auto import tqdm
from PIL import Image

from calvin_env.envs.play_table_env import get_env

logger = logging.getLogger(__name__)

if  int(os.getenv("DEBUG")):
    EP_LEN = 360
    NUM_SEQUENCES = 1
else:
    EP_LEN = 360
    NUM_SEQUENCES = int(os.getenv("NUM_EVAL_SEQUENCES"))

print("os.getenv(\"CUDA_VISIBLE_DEVICES\"):", os.getenv("CUDA_VISIBLE_DEVICES"))
print("EP_LEN:", EP_LEN)
print("NUM_SEQUENCES:", NUM_SEQUENCES)

def make_env(dataset_path):
    val_folder = Path(dataset_path) / "validation"
    env = get_env(val_folder, show_gui=False)

    # insert your own env wrapper
    # env = Wrapper(env)
    return env


def get_agent_type(checkpoint_path):

    if "diffusion" in checkpoint_path or "public" in checkpoint_path:
        agent_type = "gc_ddpm_bc"
    elif "gcbc" in checkpoint_path:
        agent_type = "gc_bc"
    elif "gciql2" in checkpoint_path:
        agent_type = "gc_iql2"
    elif "gciql3" in checkpoint_path:
        agent_type = "gc_iql3"
    elif "gciql4" in checkpoint_path:
        agent_type = "gc_iql4"
    elif "gciql5" in checkpoint_path:
        agent_type = "gc_iql5"
    elif "gciql" in checkpoint_path:
        agent_type = "gc_iql"
    else:
        raise ValueError(f"Cannot determine agent type from \"{checkpoint_path}\".")
    
    # if "nam" in checkpoint_path or "noactnorm" in checkpoint_path:
    #     agent_type += "noactnorm"
    
    return agent_type


class CustomModel(CalvinBaseModel):
    def __init__(self, use_temporal_ensembling, diffusion_model_checkpoint_path, gc_policy_checkpoint_path, gc_vf_checkpoint_path, num_denoising_steps, num_samples=1, hard_coded_goal_images_path=None, hard_coded_goal_idxs=None, diffusion_model_framework="jax"):
        # Initialize diffusion model

        self.num_samples = num_samples

        if diffusion_model_framework == "jax":
            self.diffusion_model = jax_diffusion_model.DiffusionModel(num_denoising_steps, num_samples=self.num_samples)
        elif diffusion_model_framework == "pytorch":
            self.diffusion_model = pytorch_diffusion_model.PytorchDiffusionModel()
        else:
            raise ValueError(f"Unsupported diffusion model framework: \"{diffusion_model_framework}\".")

        # Initialize GCBC
        
        agent_type = get_agent_type(os.getenv("GC_POLICY_CHECKPOINT"))
        print(f"Initializing policy from \"{os.getenv('GC_POLICY_CHECKPOINT')}\"...")
        normalize_actions = "noactnorm" in os.getenv("GC_POLICY_CHECKPOINT") or "nam" in os.getenv("GC_POLICY_CHECKPOINT")
        self.gc_policy = diffusion_gc_policy.GCPolicy(agent_type, os.getenv("GC_POLICY_CHECKPOINT"), normalize_actions=normalize_actions, use_temporal_ensembling=use_temporal_ensembling)

        # if os.getenv("GC_VF_CHECKPOINT") is not None:
        if self.num_samples > 1:
            assert os.getenv("GC_VF_CHECKPOINT") is not None
            print(f"Initializing value function from \"{os.getenv('GC_VF_CHECKPOINT')}\"...")
            vf_agent_type = get_agent_type(os.getenv("GC_VF_CHECKPOINT"))
            self.gc_vf = diffusion_gc_policy.GCPolicy(vf_agent_type, os.getenv("GC_VF_CHECKPOINT"), use_temporal_ensembling=use_temporal_ensembling)

        # self.gc_policy = diffusion_gc_policy.GCPolicyNoBuffer()

        

        # For each eval episode we need to log the following:
        #   (1) language task
        #   (2) sequence of image observations as a video
        #   (3) sequence of diffusion model generations also as a video, timed with (2)
        #   (4) sequence of actions as numpy array
        # self.log_dir = "experiments"
        # self.log_dir = os.path.join("experiments", *os.getenv("DIFFUSION_MODEL_CHECKPOINT").split("/")[-2:])
        

        # if "jax_model" in os.getenv("DIFFUSION_MODEL_CHECKPOINT"):
        #     self.log_dir = os.path.join("experiments", *os.getenv("DIFFUSION_MODEL_CHECKPOINT").split("/")[-4:-1])
        # else:
        #     self.log_dir = os.path.join("experiments", *os.getenv("DIFFUSION_MODEL_CHECKPOINT").split("/")[-3:])

        # self.log_dir = os.path.join(self.log_dir,  *os.getenv("GC_POLICY_CHECKPOINT").split("/")[-2:])
        # if "jax_model" in diffusion_model_checkpoint_path:
        #     self.log_dir = os.path.join("experiments", *diffusion_model_checkpoint_path.split("/")[-4:-1])
        # else:
        #     self.log_dir = os.path.join("experiments", *diffusion_model_checkpoint_path.split("/")[-3:])

        # self.log_dir = os.path.join(self.log_dir,  *gc_policy_checkpoint_path.split("/")[-2:])

        if "jax_model" in diffusion_model_checkpoint_path:
            self.log_dir = os.path.join("experiments", *diffusion_model_checkpoint_path.replace("-", "/").split("/")[-4:-1])
        else:
            self.log_dir = os.path.join("experiments", *diffusion_model_checkpoint_path.replace("-", "/").split("/")[-3:])

        ensemb_str = "tmpensb" if use_temporal_ensembling else "notmpensb"
        timestamp = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")

        if self.num_samples > 1:
            if "seed" in gc_policy_checkpoint_path:
                self.log_dir = os.path.join(self.log_dir,  *gc_policy_checkpoint_path.replace("-", "/").split("/")[-5:], *gc_vf_checkpoint_path.replace("-", "/").split("/")[-2:], f"{num_denoising_steps}_denoising_steps", f"{num_samples}_samples", ensemb_str, timestamp)
            else:
                self.log_dir = os.path.join(self.log_dir,  *gc_policy_checkpoint_path.replace("-", "/").split("/")[-2:], *gc_vf_checkpoint_path.replace("-", "/").split("/")[-2:], f"{num_denoising_steps}_denoising_steps", f"{num_samples}_samples", ensemb_str, timestamp)
        else:
            if "seed" in gc_policy_checkpoint_path:
                self.log_dir = os.path.join(self.log_dir,  *gc_policy_checkpoint_path.replace("-", "/").split("/")[-5:], "no_vf", "checkpoint_none", f"{num_denoising_steps}_denoising_steps", f"{num_samples}_samples", ensemb_str, timestamp)
            else:
                self.log_dir = os.path.join(self.log_dir,  *gc_policy_checkpoint_path.replace("-", "/").split("/")[-2:], "no_vf", "checkpoint_none", f"{num_denoising_steps}_denoising_steps", f"{num_samples}_samples", ensemb_str, timestamp)
        
        # Make sure name is right, and add seed 

        print(f"Logging to \"{self.log_dir}\"...")
        os.makedirs(self.log_dir, exist_ok=True)
        self.episode_counter = None
        self.language_task = None
        self.obs_image_seq = None
        self.goal_image_seq = None
        self.vranked_goal_images_seq = None
        self.vranking_save_freq = 1
        self.action_seq = None
        self.combined_images = None

        # Other necessary variables for running rollouts
        self.goal_image = None
        self.subgoal_counter = 0
        self.subgoal_max = 20
        # self.subgoal_max = 30
        self.pbar = None


        if hard_coded_goal_images_path is None:
            self.hard_coded_goal_images = None 
        else:
            goal_images = np.load(hard_coded_goal_images_path)

            assert hard_coded_goal_idxs is not None
            idxs = [int(idx) for idx in hard_coded_goal_idxs.split(",")]
            print("idxs:", idxs)
            self.hard_coded_goal_images = goal_images[idxs]
            
            
            np.save(os.path.join(self.log_dir, "hardcoded_goal_images.npy"), self.hard_coded_goal_images)

            os.makedirs(os.path.join(self.log_dir, "hardcoded_goal_images"), exist_ok=True)
            for i, img in enumerate(self.hard_coded_goal_images):
                cv2.imwrite(os.path.join(self.log_dir, "hardcoded_goal_images", f"goal_image_{idxs[i]}.png"), img[..., ::-1])


    def save_info(self, success):
        episode_log_dir = os.path.join(self.log_dir, "ep" + str(self.episode_counter))
        if not os.path.exists(episode_log_dir):
            os.makedirs(episode_log_dir)

        # Log the language task
        with open(os.path.join(episode_log_dir, "language_task.txt"), "w") as f:
            f.write(self.language_task + "\n")
            f.write(f"success: {success}")
        
        # Log the observation video
        size = (200, 200)
        out = cv2.VideoWriter(os.path.join(episode_log_dir, "trajectory.mp4"), cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
        for i in range(len(self.obs_image_seq)):
            rgb_img = cv2.cvtColor(self.obs_image_seq[i], cv2.COLOR_RGB2BGR)
            out.write(rgb_img)
        out.release()

        # Log the goals video
        size = (200, 200)
        out = cv2.VideoWriter(os.path.join(episode_log_dir, "goals.mp4"), cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
        for i in range(len(self.goal_image_seq)):
            rgb_img = cv2.cvtColor(self.goal_image_seq[i], cv2.COLOR_RGB2BGR)
            out.write(rgb_img)
        out.release()

        # Log the combined image
        size = (400, 200)
        out = cv2.VideoWriter(os.path.join(episode_log_dir, "combined.mp4"), cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
        for i in range(len(self.combined_images)):
            rgb_img = cv2.cvtColor(self.combined_images[i], cv2.COLOR_RGB2BGR)
            out.write(rgb_img)
        out.release()

        # Log the value function rankings 
        if self.num_samples > 1 and self.episode_counter % self.vranking_save_freq == 0:
            for i in range(len(self.vranked_goal_images_seq)):
                timestep_dir = os.path.join(episode_log_dir, f"vranked_goal_images", f"timestep_{i * self.subgoal_max:03d}")
                os.makedirs(timestep_dir, exist_ok=True)
                for vpred, goal_image in self.vranked_goal_images_seq[i].items():
                    rgb_img = cv2.cvtColor(goal_image, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(os.path.join(timestep_dir, f"{vpred}.png"), rgb_img)

        
        # Log the actions
        np.save(os.path.join(episode_log_dir, "actions.npy"), np.array(self.action_seq))

        np.save(os.path.join(episode_log_dir, "real_goal_images.npy"), np.array(self.real_goal_images[1:]))

        np.save(os.path.join(episode_log_dir, "goal_images.npy"), np.array(self.goal_images_list))

        for i, img in enumerate(self.real_goal_images[1:]):
            cv2.imwrite(os.path.join(episode_log_dir, f"real_goal_image_{i}.png"), img[..., ::-1])
            cv2.imwrite(os.path.join(episode_log_dir, f"goal_image_side{i}.png"), np.concatenate([img[..., ::-1], self.goal_images_list[i][..., ::-1]], axis=1))

        
        np.save(os.path.join(episode_log_dir, "real_images.npy"), np.array(self.obs_image_seq))
        os.makedirs(os.path.join(episode_log_dir, "real_images"), exist_ok=True)
        for i, img in enumerate(self.obs_image_seq):
            cv2.imwrite(os.path.join(episode_log_dir, "real_images", f"real_image_{i}.png"), img[..., ::-1])


    def reset(self):
        if self.episode_counter is None: # this is the first time reset has been called
            self.episode_counter = 0
            self.obs_image_seq = []
            self.goal_image_seq = []
            self.vranked_goal_images_seq = []
            self.action_seq = []
            self.combined_images = []
            self.real_goal_images = []
            self.goal_images_list = []
            self.hardcoded_goal_idx_counter = 0
        else:
            # episode_log_dir = os.path.join(self.log_dir, "ep" + str(self.episode_counter))
            # if not os.path.exists(episode_log_dir):
            #     os.makedirs(episode_log_dir)

            # # Log the language task
            # with open(os.path.join(episode_log_dir, "language_task.txt"), "a") as f:
            #     f.write(self.language_task)
            
            # # Log the observation video
            # size = (200, 200)
            # out = cv2.VideoWriter(os.path.join(episode_log_dir, "trajectory.mp4"), cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
            # for i in range(len(self.obs_image_seq)):
            #     rgb_img = cv2.cvtColor(self.obs_image_seq[i], cv2.COLOR_RGB2BGR)
            #     out.write(rgb_img)
            # out.release()

            # # Log the goals video
            # size = (200, 200)
            # out = cv2.VideoWriter(os.path.join(episode_log_dir, "goals.mp4"), cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
            # for i in range(len(self.goal_image_seq)):
            #     rgb_img = cv2.cvtColor(self.goal_image_seq[i], cv2.COLOR_RGB2BGR)
            #     out.write(rgb_img)
            # out.release()

            # # Log the combined image
            # size = (400, 200)
            # out = cv2.VideoWriter(os.path.join(episode_log_dir, "combined.mp4"), cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
            # for i in range(len(self.combined_images)):
            #     rgb_img = cv2.cvtColor(self.combined_images[i], cv2.COLOR_RGB2BGR)
            #     out.write(rgb_img)
            # out.release()

            # # Log the actions
            # np.save(os.path.join(episode_log_dir, "actions.npy"), np.array(self.action_seq))

            # Update/reset all the variables
            self.episode_counter += 1
            self.obs_image_seq = []
            self.goal_image_seq = []
            self.vranked_goal_images_seq = []
            self.action_seq = []
            self.goal_image = None
            self.combined_images = []
            self.real_goal_images = []
            self.goal_images_list = []
            self.subgoal_counter = 0
            self.hardcoded_goal_idx_counter = 0

            # Reset the GC policy
            self.gc_policy.reset()

        # tqdm progress bar
        if self.pbar is not None:
            self.pbar.close()
        self.pbar = tqdm(total=EP_LEN)

    def step(self, obs, goal):
        """
        Args:
            obs: environment observations
            goal: embedded language goal
        Returns:
            action: predicted action
        """
        rgb_obs = obs["rgb_obs"]["rgb_static"]
        self.language_task = goal

        # If we need to, generate a new goal image
        if self.goal_image is None or self.subgoal_counter >= self.subgoal_max:
            if self.hard_coded_goal_images is None:
                t0 = time.time()
                goal_images = self.diffusion_model.generate(self.language_task, rgb_obs)
                t1 = time.time()
                print(f"t1 - t0: {t1 - t0:.3f}")
            else:
                if self.hardcoded_goal_idx_counter >= self.hard_coded_goal_images.shape[0]:
                    return -1

                goal_images = self.hard_coded_goal_images[self.hardcoded_goal_idx_counter][None]

                self.hardcoded_goal_idx_counter += 1


            self.subgoal_counter = 0


            if self.num_samples > 1:
                v = self.gc_vf.value_function_ranking(rgb_obs, goal_images)
                v_idx = v.argmax()
                self.goal_image = goal_images[v_idx]

                self.vranked_goal_images_seq.append({v[i]:goal_images[i] for i in range(v.shape[0])})
            else:
                assert goal_images.shape[0] == 1, f"goal_images.shape: {goal_images.shape}"
                self.goal_image = goal_images[0]

            self.real_goal_images.append(rgb_obs.copy())
            self.goal_images_list.append(self.goal_image.copy())

        # Log the image observation and the goal image
        self.obs_image_seq.append(rgb_obs)
        self.goal_image_seq.append(self.goal_image)
        self.combined_images.append(np.concatenate([rgb_obs, self.goal_image], axis=1))
        assert self.combined_images[-1].shape == (200, 400, 3)

        

        # Query the behavior cloning model
        action_cmd = self.gc_policy.predict_action(rgb_obs, self.goal_image)

        # Log the predicted action
        self.action_seq.append(action_cmd)

        # Update variables
        self.subgoal_counter += 1

        # Update progress bar
        self.pbar.update(1)

        return action_cmd


def evaluate_policy(model, env, epoch=0, eval_log_dir=None, debug=False, create_plan_tsne=False):
    """
    Run this function to evaluate a model on the CALVIN challenge.

    Args:
        model: Must implement methods of CalvinBaseModel.
        env: (Wrapped) calvin env.
        epoch:
        eval_log_dir: Path where to log evaluation results. If None, logs to /tmp/evaluation/
        debug: If True, show camera view and debug info.
        create_plan_tsne: Collect data for TSNE plots of latent plans (does not work for your custom model)

    Returns:
        Dictionary with results
    """
    # conf_dir = Path(__file__).absolute().parents[2] / "conf"
    conf_dir = Path(__file__).absolute().parents[0] / "calvin_models" / "conf"
    task_cfg = OmegaConf.load(conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml")
    task_oracle = hydra.utils.instantiate(task_cfg)
    val_annotations = OmegaConf.load(conf_dir / "annotations/new_playtable_validation.yaml")

    eval_log_dir = get_log_dir(eval_log_dir)


    # initial_state: {'led': 0, 'lightbulb': 0, 'slider': 'left', 'drawer': 'closed', 'red_block': 'table', 'blue_block': 'slider_right', 'pink_block': 'table', 'grasped': 0}
    # eval_sequence: ('lift_blue_block_slider', 'place_in_slider', 'turn_on_lightbulb', 'open_drawer', 'push_pink_block_left')

    # initial_state: {'led': 0, 'lightbulb': 1, 'slider': 'right', 'drawer': 'closed', 'red_block': 'table', 'blue_block': 'slider_left', 'pink_block': 'slider_right', 'grasped': 0}
    # eval_sequence: ('lift_blue_block_slider', 'place_in_slider', 'open_drawer', 'rotate_red_block_right', 'turn_on_led')

    # initial_state: {'led': 0, 'lightbulb': 0, 'slider': 'right', 'drawer': 'open', 'red_block': 'slider_left', 'blue_block': 'table', 'pink_block': 'slider_right', 'grasped': 0}
    # eval_sequence: ('lift_red_block_slider', 'place_in_drawer', 'move_slider_left', 'turn_on_led', 'close_drawer')

    # initial_state: {'led': 0, 'lightbulb': 0, 'slider': 'right', 'drawer': 'closed', 'red_block': 'slider_right', 'blue_block': 'slider_left', 'pink_block': 'table', 'grasped': 0}
    # eval_sequence: ('lift_blue_block_slider', 'place_in_slider', 'turn_on_led', 'open_drawer', 'rotate_pink_block_right')

    initial_state = {'led': 0, 'lightbulb': 0, 'slider': 'left', 'drawer': 'open', 'red_block': 'slider_right', 'blue_block': 'table', 'pink_block': 'slider_left', 'grasped': 0}
    # eval_sequence: ('lift_red_block_slider', 'place_in_drawer', 'turn_on_led', 'move_slider_right', 'lift_red_block_drawer')

    # initial_state: {'led': 0, 'lightbulb': 1, 'slider': 'right', 'drawer': 'open', 'red_block': 'table', 'blue_block': 'slider_right', 'pink_block': 'table', 'grasped': 0}
    # eval_sequence: ('lift_pink_block_table', 'place_in_drawer', 'rotate_red_block_left', 'move_slider_left', 'lift_pink_block_drawer')

    # initial_state: {'led': 0, 'lightbulb': 1, 'slider': 'right', 'drawer': 'open', 'red_block': 'table', 'blue_block': 'table', 'pink_block': 'slider_right', 'grasped': 0}
    # eval_sequence: ('lift_blue_block_table', 'place_in_slider', 'turn_on_led', 'move_slider_left', 'close_drawer')

    # initial_state: {'led': 0, 'lightbulb': 0, 'slider': 'left', 'drawer': 'open', 'red_block': 'table', 'blue_block': 'slider_right', 'pink_block': 'table', 'grasped': 0}
    # eval_sequence: ('lift_blue_block_slider', 'place_in_drawer', 'rotate_red_block_left', 'turn_on_lightbulb', 'lift_red_block_table')

    # initial_state: {'led': 0, 'lightbulb': 1, 'slider': 'left', 'drawer': 'open', 'red_block': 'slider_right', 'blue_block': 'table', 'pink_block': 'slider_left', 'grasped': 0}
    # eval_sequence: ('lift_red_block_slider', 'stack_block', 'close_drawer', 'unstack_block', 'push_red_block_left')

    # initial_state: {'led': 0, 'lightbulb': 1, 'slider': 'right', 'drawer': 'closed', 'red_block': 'table', 'blue_block': 'slider_left', 'pink_block': 'table', 'grasped': 0}
    # eval_sequence: ('lift_red_block_table', 'stack_block', 'open_drawer', 'unstack_block', 'push_pink_block_right')

    # initial_state: {'led': 0, 'lightbulb': 1, 'slider': 'right', 'drawer': 'open', 'red_block': 'slider_right', 'blue_block': 'table', 'pink_block': 'table', 'grasped': 0}
    # eval_sequence: ('lift_blue_block_table', 'stack_block', 'close_drawer', 'unstack_block', 'rotate_pink_block_right')

    # initial_state: {'led': 0, 'lightbulb': 1, 'slider': 'right', 'drawer': 'closed', 'red_block': 'slider_left', 'blue_block': 'table', 'pink_block': 'slider_right', 'grasped': 0}
    # eval_sequence: ('lift_red_block_slider', 'stack_block', 'unstack_block', 'push_blue_block_left', 'lift_red_block_table')

    # initial_state: {'led': 0, 'lightbulb': 0, 'slider': 'right', 'drawer': 'open', 'red_block': 'table', 'blue_block': 'table', 'pink_block': 'slider_right', 'grasped': 0}
    # eval_sequence: ('lift_red_block_table', 'place_in_slider', 'move_slider_left', 'turn_on_led', 'rotate_blue_block_right')

    # initial_state: {'led': 0, 'lightbulb': 1, 'slider': 'right', 'drawer': 'open', 'red_block': 'slider_right', 'blue_block': 'table', 'pink_block': 'table', 'grasped': 0}
    # eval_sequence: ('lift_blue_block_table', 'stack_block', 'close_drawer', 'unstack_block', 'rotate_pink_block_right')

    # initial_state: {'led': 0, 'lightbulb': 1, 'slider': 'right', 'drawer': 'closed', 'red_block': 'slider_left', 'blue_block': 'table', 'pink_block': 'slider_right', 'grasped': 0}
    # eval_sequence: ('lift_red_block_slider', 'stack_block', 'unstack_block', 'push_blue_block_left', 'lift_red_block_table')

    # initial_state: {'led': 0, 'lightbulb': 0, 'slider': 'right', 'drawer': 'open', 'red_block': 'table', 'blue_block': 'table', 'pink_block': 'slider_right', 'grasped': 0}
    # eval_sequence: ('lift_red_block_table', 'place_in_slider', 'move_slider_left', 'turn_on_led', 'rotate_blue_block_right')

    # initial_state: {'led': 0, 'lightbulb': 0, 'slider': 'left', 'drawer': 'open', 'red_block': 'table', 'blue_block': 'table', 'pink_block': 'slider_right', 'grasped': 0}
    # eval_sequence: ('lift_blue_block_table', 'place_in_drawer', 'push_red_block_right', 'move_slider_right', 'turn_on_lightbulb')

    # initial_state: {'led': 0, 'lightbulb': 1, 'slider': 'left', 'drawer': 'closed', 'red_block': 'slider_left', 'blue_block': 'slider_right', 'pink_block': 'table', 'grasped': 0}
    # eval_sequence: ('lift_pink_block_table', 'place_in_slider', 'turn_on_led', 'move_slider_right', 'open_drawer')

    # initial_state: {'led': 0, 'lightbulb': 0, 'slider': 'right', 'drawer': 'closed', 'red_block': 'slider_left', 'blue_block': 'table', 'pink_block': 'table', 'grasped': 0}
    # eval_sequence: ('lift_blue_block_table', 'place_in_slider', 'turn_on_lightbulb', 'push_pink_block_right', 'open_drawer')

    # initial_state: {'led': 0, 'lightbulb': 1, 'slider': 'right', 'drawer': 'open', 'red_block': 'table', 'blue_block': 'slider_left', 'pink_block': 'table', 'grasped': 0}
    # eval_sequence: ('lift_pink_block_table', 'place_in_drawer', 'turn_off_lightbulb', 'lift_blue_block_slider', 'stack_block')


    # eval_sequences = get_sequences(NUM_SEQUENCES)
    # eval_sequences = get_sequences(100)
    eval_sequences = [(initial_state, ("lift_red_block_slider",)) for _ in range(NUM_SEQUENCES)]


    eval_sequences

    
    # for init_state, eval_sequence in eval_sequences:
    #     print("\ninitial_state:", init_state)
    #     print("eval_sequence:", eval_sequence)

    results = []
    plans = defaultdict(list) 

    if not debug:
        eval_sequences = tqdm(eval_sequences, position=0, leave=True)

    for initial_state, eval_sequence in eval_sequences:
        print("initial_state:", initial_state)
        print("eval_sequence:", eval_sequence)
        result = evaluate_sequence(env, model, task_oracle, initial_state, eval_sequence, val_annotations, plans, debug)
        results.append(result)
        if not debug:
            eval_sequences.set_description(
                " ".join([f"{i + 1}/5 : {v * 100:.1f}% |" for i, v in enumerate(count_success(results))]) + "|"
            )

    if create_plan_tsne:
        create_tsne(plans, eval_log_dir, epoch)
    print_and_save(results, eval_sequences, eval_log_dir, epoch)

    return results


def evaluate_sequence(env, model, task_checker, initial_state, eval_sequence, val_annotations, plans, debug):
    """
    Evaluates a sequence of language instructions.
    """
    robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
    env.reset(robot_obs=robot_obs, scene_obs=scene_obs)

    success_counter = 0
    if debug:
        time.sleep(1)
        print()
        print()
        print(f"Evaluating sequence: {' -> '.join(eval_sequence)}")
        print("Subtask: ", end="")
    for subtask in eval_sequence:
        success = rollout(env, model, task_checker, subtask, val_annotations, plans, debug)
        if success:
            success_counter += 1
        else:
            return success_counter
    return success_counter


def rollout(env, model, task_oracle, subtask, val_annotations, plans, debug):
    """
    Run the actual rollout on one subtask (which is one natural language instruction).
    """
    if debug:
        print(f"{subtask} ", end="")
        time.sleep(0.5)

    obs = env.get_obs()

    # get lang annotation for subtask
    lang_annotation = val_annotations[subtask][0]
    model.reset()
    start_info = env.get_info()

    print("lang_annotation:", lang_annotation)

    for step in range(EP_LEN):
        action = model.step(obs, lang_annotation)

        if isinstance(action, int) and action == -1:
            print("Ran out of hard coded goal images")
            break 

        obs, _, _, current_info = env.step(action)
        rgb_obs = obs["rgb_obs"]["rgb_static"]
        if debug:
            img = env.render(mode="rgb_array")
            join_vis_lang(img, lang_annotation)
            # time.sleep(0.1)
        if step == 0:
            # for tsne plot, only if available
            collect_plan(model, plans, subtask)

        # check if current step solves a task
        current_task_info = task_oracle.get_task_info_for_set(start_info, current_info, {subtask})
        if len(current_task_info) > 0:
            if debug:
                print(colored("success", "green"), end=" ")
            print(colored("success", "green"), end=" ")
            print("step:", step)
            print("current_task_info:", current_task_info)
            model.real_goal_images.append(rgb_obs)
            model.obs_image_seq.append(rgb_obs)
            model.save_info(True)
            return True
    if debug:
        print(colored("fail", "red"), end=" ")

    
    model.real_goal_images.append(rgb_obs)
    model.obs_image_seq.append(rgb_obs)
    model.save_info(False)
    print(colored("fail", "red"), end=" ")
    return False


def main():
    # seed_everything(0, workers=True)  # type:ignore ###UNCOMMENT




    parser = argparse.ArgumentParser(description="Evaluate a trained model on multistep sequences with language goals.")
    parser.add_argument("--dataset_path", type=str, help="Path to the dataset root directory.")

    # arguments for loading default model
    parser.add_argument(
        "--train_folder", type=str, help="If calvin_agent was used to train, specify path to the log dir."
    )
    parser.add_argument(
        "--checkpoints",
        type=str,
        default=None,
        help="Comma separated list of epochs for which checkpoints will be loaded",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path of the checkpoint",
    )
    parser.add_argument(
        "--last_k_checkpoints",
        type=int,
        help="Specify the number of checkpoints you want to evaluate (starting from last). Only used for calvin_agent.",
    )

    # arguments for loading custom model or custom language embeddings
    # parser.add_argument(
    #     "--custom_model", action="store_true", help="Use this option to evaluate a custom model architecture."
    # )
    parser.add_argument(
        "--custom_model", type=int, default=1, help="Use this option to evaluate a custom model architecture."
    )

    parser.add_argument(
        "--diffusion_model_framework",
        type=str,
        default="jax",
        choices=["jax", "pytorch"],
        help="Comma separated list of epochs for which checkpoints will be loaded",
    )

    parser.add_argument(
        "--diffusion_model_checkpoint_path", type=str, help="Use this option to evaluate a custom model architecture."
    )

    parser.add_argument(
        "--gc_policy_checkpoint_path", type=str, help="Use this option to evaluate a custom model architecture."
    )


    parser.add_argument(
        "--gc_vf_checkpoint_path", type=str, help="Use this option to evaluate a custom model architecture."
    )

    parser.add_argument(
        "--s3_save_uri", type=str, help="Use this option to evaluate a custom model architecture."
    )

    parser.add_argument(
        "--save_to_s3", type=int, default=1, help="Use this option to evaluate a custom model architecture."
    )


    parser.add_argument(
        "--num_denoising_steps", type=int, default=200, help="Use this option to evaluate a custom model architecture."
    )

    # parser.add_argument(
    #     "--agent_type", type=str, help="Use this option to evaluate a custom model architecture."
    # )

    parser.add_argument(
        "--use_temporal_ensembling", type=int, default=1, help="Use this option to evaluate a custom model architecture."
    )

    parser.add_argument(
        "--num_samples", type=int, default=1, help="Use this option to evaluate a custom model architecture."
    )

    parser.add_argument(
        "--hard_coded_goal_images_path", type=str, default=None, help="Use this option to evaluate a custom model architecture."
    )

    parser.add_argument(
        "--hard_coded_goal_idxs", type=str, default=None, help="Use this option to evaluate a custom model architecture."
    )

    # parser.add_argument("--debug", action="store_true", help="Print debug info and visualize environment.")
    parser.add_argument("--debug", type=int, default=0)

    # parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--eval_log_dir", default=None, type=str, help="Where to log the evaluation results.")

    parser.add_argument("--device", default=0, type=int, help="CUDA device")
    args = parser.parse_args()

    # random.seed(args.seed)
    # np.random.seed(args.seed)
    # torch.manual_seed(args.seed)


    # evaluate a custom model
    if args.custom_model:
        model = CustomModel(args.use_temporal_ensembling, args.diffusion_model_checkpoint_path, args.gc_policy_checkpoint_path, args.gc_vf_checkpoint_path, args.num_denoising_steps, num_samples=args.num_samples, hard_coded_goal_images_path=args.hard_coded_goal_images_path, hard_coded_goal_idxs=args.hard_coded_goal_idxs, diffusion_model_framework=args.diffusion_model_framework)
        env = make_env(args.dataset_path)
        evaluate_policy(model, env, debug=args.debug, eval_log_dir=model.log_dir)
    else:
        assert False 
        assert "train_folder" in args

        checkpoints = []
        if args.checkpoints is None and args.last_k_checkpoints is None and args.checkpoint is None:
            print("Evaluating model with last checkpoint.")
            checkpoints = [get_last_checkpoint(Path(args.train_folder))]
        elif args.checkpoints is not None:
            print(f"Evaluating model with checkpoints {args.checkpoints}.")
            checkpoints = get_checkpoints_for_epochs(Path(args.train_folder), args.checkpoints)
        elif args.checkpoints is None and args.last_k_checkpoints is not None:
            print(f"Evaluating model with last {args.last_k_checkpoints} checkpoints.")
            checkpoints = get_all_checkpoints(Path(args.train_folder))[-args.last_k_checkpoints :]
        elif args.checkpoint is not None:
            checkpoints = [Path(args.checkpoint)]

        env = None
        for checkpoint in checkpoints:
            epoch = checkpoint.stem.split("=")[1]
            model, env, _ = get_default_model_and_env(
                args.train_folder,
                args.dataset_path,
                checkpoint,
                env=env,
                device_id=args.device,
            )
            evaluate_policy(model, env, epoch, eval_log_dir=args.eval_log_dir, debug=args.debug, create_plan_tsne=True)
            # evaluate_policy(model, env, epoch, eval_log_dir=model.log_dir, debug=args.debug, create_plan_tsne=True)

    

    if int(os.getenv("DEBUG")):
        args.s3_save_uri = os.path.join(args.s3_save_uri, "el_trasho")


    if args.save_to_s3:
        s3_callback = S3SyncCallback(model.log_dir, os.path.join(args.s3_save_uri, "/".join(model.log_dir.split("/")[1:])) + "/")
        s3_callback.on_train_epoch_end()
    

if __name__ == "__main__":
    main()
