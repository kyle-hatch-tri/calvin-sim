import os 
import numpy as np 
from glob import glob 
from tqdm import tqdm, trange 
from jaxrl_m.data.text_processing import text_processors

from calvin_agent.evaluation.gcbc_train_config import get_config
from calvin_agent.evaluation import diffusion_gc_policy
import cv2

from calvin_agent.evaluation import jax_diffusion_model


def visualize_outputs(args):
    os.chdir("..")


    diffusion_model = jax_diffusion_model.DiffusionModel(50, num_samples=1)

    rgb_image_obs = cv2.imread(args.input_image)
    rgb_image_obs = rgb_image_obs[..., ::-1]

    goal_images, _ = diffusion_model.generate(args.language_goal, rgb_image_obs, return_inference_time=True, prompt_w=args.prompt_w, context_w=args.context_w)

    print("args.language_goal:", args.language_goal)

    cv2.imwrite(os.path.join("random_util_scripts", "instruct_pix2pix_images", f"{args.desc}_p{args.prompt_w}_c{args.context_w}.png"), goal_images[0][..., ::-1])

def parse_arguments():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate a trained model on multistep sequences with language goals.")
    # parser.add_argument("--logdir", type=str, help="Path to the dataset root directory.")
    parser.add_argument("--input_image", type=str, help="Path to the dataset root directory.")
    parser.add_argument("--desc", type=str, help="Path to the dataset root directory.")
    parser.add_argument("--language_goal", type=str, help="Path to the dataset root directory.")
    # parser.add_argument("--slice", type=int, default=None, help="Path to the dataset root directory.")
    parser.add_argument("--prompt_w", type=float, default=7.5, help="Path to the dataset root directory.")
    parser.add_argument("--context_w", type=float, default=1.5, help="Path to the dataset root directory.")
    # parser.add_argument("--agent_config_string", type=str, help="Path to the dataset root directory.")
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
# export DIFFUSION_MODEL_CHECKPOINT=/home/kylehatch/Desktop/hidql/susie-calvin-checkpoints/susie_test/test1_400smthlibs2_2024.02.23_14.20.24/40000/params_ema
export DIFFUSION_MODEL_CHECKPOINT=/home/kylehatch/Desktop/hidql/susie-calvin-checkpoints/susie_test/public_model/checkpoint_only/params_ema

python3 -u visualize_high_level_policy_instruct_pix2pix_tasks.py \
--input_image /home/kylehatch/Desktop/hidql/calvin-sim/random_util_scripts/instruct_pix2pix_images/pexels-pixabay-371589.jpg \
--language_goal "store the grasped block in the sliding cabinet" \
--desc public_slider \
--prompt_w 7.5 \
--context_w 1.5


python3 -u visualize_high_level_policy_instruct_pix2pix_tasks.py \
--input_image /home/kylehatch/Desktop/hidql/calvin-sim/random_util_scripts/instruct_pix2pix_images/pexels-pixabay-371589.jpg \
--language_goal "Add boats on the water" \
--desc public_default \
--prompt_w 7.5 \
--context_w 1.5


python3 -u visualize_high_level_policy_instruct_pix2pix_tasks.py \
--input_image /home/kylehatch/Desktop/hidql/calvin-sim/random_util_scripts/instruct_pix2pix_images/rgb_obs.png \
--language_goal "store the grasped block in the sliding cabinet" \
--desc public_slider_rgbobs \
--prompt_w 7.5 \
--context_w 1.5



python3 -u visualize_high_level_policy_instruct_pix2pix_tasks.py \
--input_image /home/kylehatch/Desktop/hidql/calvin-sim/random_util_scripts/instruct_pix2pix_images/mona_lisa.png \
--language_goal "Make it a marble Roman sculpture" \
--desc public_default_mona_lisa \
--prompt_w 7.5 \
--context_w 1.5
"""