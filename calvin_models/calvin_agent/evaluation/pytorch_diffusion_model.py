
import os 
import numpy as np 
import torch
from PIL import Image

from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import (AutoencoderKL, DDPMScheduler,
                       StableDiffusionInstructPix2PixPipeline,
                       UNet2DConditionModel)
from diffusers.training_utils import EMAModel
from diffusers import StableDiffusionInstructPix2PixPipeline

class PytorchDiffusionModel:
    def __init__(self):
        self.generator = torch.Generator(device="cuda").manual_seed(42) ### set the rng I think

        weight_dtype = torch.float16
        revision = None

        model_id = os.getenv("DIFFUSION_MODEL_CHECKPOINT")
        print(f"Loading diffusion model weights from \"{model_id}\"...")
        self.pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            model_id, 
            revision=revision,
            torch_dtype=weight_dtype, 
            requires_safety_checker=False,
            safety_checker=None
        )

        self.pipeline = self.pipeline.to("cuda")
        self.pipeline.set_progress_bar_config(disable=True)
        

    def generate(self, language_command : str, image_obs : np.ndarray):
        # Resize image to 256x256
        image_obs = np.array(Image.fromarray(image_obs).resize((256, 256))).astype(np.uint8)

        with torch.autocast("cuda", enabled=True):
            sample = self.pipeline(
                    language_command, # prompt,
                    image=image_obs / 255, # denormalize(original_image), # diffuser's image processor expects [0,1] range
                    num_inference_steps=20, # or go to 200? 
                    image_guidance_scale=1.5,
                    guidance_scale=7,
                    generator=self.generator,
                ).images[0]
            
            # What is the eta thing from the jax model?
            
        return np.array(Image.fromarray(np.array(sample)).resize((200, 200))).astype(np.uint8)