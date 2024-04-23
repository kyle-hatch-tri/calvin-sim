import os 
import numpy as np 
from pathlib import Path

# from calvin_env.envs.play_table_env import get_env

# dataset_path = "mini_dataset"
# def make_env(dataset_path):
#     val_folder = Path(dataset_path) / "validation"
#     print("Before get_env")
#     env = get_env(val_folder, show_gui=False)
#     print("After get_env")

#     # insert your own env wrapper
#     # env = Wrapper(env)
#     return env


# env = make_env(dataset_path)

# import ipdb; ipdb.set_trace()


from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv
from libero.libero.utils import get_libero_path


benchmark_dict = benchmark.get_benchmark_dict()

task_suite_name = "libero_10"
task_name = "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it"

task_suite = benchmark_dict[task_suite_name]()
# retrieve a specific task
task_names = [task.name for task in task_suite.tasks]

assert task_name in task_names, f"\"{task_name}\" not in task_names: {task_names}"

    
task_id = task_names.index(task_name)
task = task_suite.get_task(task_id)
assert task.name == task_name, f"task.name: {task.name} != task_name: {task_name}"

init_states = task_suite.get_task_init_states(task_id) # for benchmarking purpose, we fix the a set of initial states

task_description = task.language
task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
print(f"[info] retrieving task {task_id} from suite {task_suite_name}, the " + \
    f"language instruction is {task_description}, and the bddl file is {task_bddl_file}")


# step over the environment
env_args = {
    "bddl_file_name": task_bddl_file,
    "camera_heights": 128,
    "camera_widths": 128,
    "control_freq":20,
}  


env = OffScreenRenderEnv(**env_args)
env.seed(0)


env.set_init_state(init_states[0])
# env.set_init_state(states[0])



reset_success = False
while not reset_success:
    try:
        env.reset()
        reset_success = True
    except:
        continue

dummy_action = [0.] * 7
for step in range(10):
    obs, reward, done, info = env.step(dummy_action)
    img = obs["agentview_image"]

import ipdb; ipdb.set_trace()

"""
Make a wrapper that keeps a dict of the several different envs, 
"""