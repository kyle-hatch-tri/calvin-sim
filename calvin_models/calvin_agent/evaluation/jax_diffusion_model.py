from susie.model import create_sample_fn, create_vae_encode_decode_fn
from susie.jax_utils import initialize_compilation_cache
# from susie_functions.model import create_sample_fn
# from susie_functions.jax_utils import initialize_compilation_cache
import numpy as np
from PIL import Image
import os

import time

class DiffusionModel:
    def __init__(self, num_denoising_steps=200, num_samples=1):
        initialize_compilation_cache()

        self.num_samples = num_samples

        print("os.getenv(\"DIFFUSION_MODEL_CHECKPOINT\"):", os.getenv("DIFFUSION_MODEL_CHECKPOINT"))

        self.sample_fn = create_sample_fn(
            os.getenv("DIFFUSION_MODEL_CHECKPOINT"),
            "kvablack/dlimp-diffusion/9n9ped8m",
            # num_timesteps=200,
            num_timesteps=num_denoising_steps,
            prompt_w=7.5,
            context_w=1.5,
            eta=0.0,
            pretrained_path="runwayml/stable-diffusion-v1-5:flax",
            num_samples=self.num_samples,
        )

        # path: str,
        # wandb_run_name: Optional[str] = None,
        # eta: float = 0.0,
        # pretrained_path: str = "runwayml/stable-diffusion-v1-5:flax",
        # num_samples: int = 1,

        self.vae_encode_decode_fn = create_vae_encode_decode_fn(
            os.getenv("DIFFUSION_MODEL_CHECKPOINT"),
            "kvablack/dlimp-diffusion/9n9ped8m",
            eta=0.0,
            pretrained_path="runwayml/stable-diffusion-v1-5:flax",
            num_samples=self.num_samples,
        )




    def generate(self, language_command : str, image_obs : np.ndarray, return_inference_time=False):
        # Resize image to 256x256
        
        image_obs = np.array(Image.fromarray(image_obs).resize((256, 256))).astype(np.uint8)

        # image_obs = np.stack([np.array(Image.fromarray(img).resize((256, 256))) for img in image_obs], axis=0).astype(np.uint8)

        t0 = time.time()
        sample = self.sample_fn(image_obs, language_command, prompt_w=7.5, context_w=1.5)

        # sample = self.sample_fn(image_obs[0], language_command, prompt_w=7.5, context_w=1.5)

        t1 = time.time()
        # print(f"\t>>t1 - t0: {t1 - t0:.3f}")
        # return np.array(Image.fromarray(sample).resize((200, 200))).astype(np.uint8)

        samples = np.array([np.array(Image.fromarray(s).resize((200, 200))).astype(np.uint8) for s in sample])

        if return_inference_time:
            return samples, t1 - t0
        else:
            return samples
        
    # def generate(self, language_command : str, image_obs : np.ndarray, return_inference_time=False):
    #     # Resize image to 256x256
        
    #     image_obs = np.array(Image.fromarray(image_obs).resize((256, 256))).astype(np.uint8)

    #     # image_obs = np.stack([np.array(Image.fromarray(img).resize((256, 256))) for img in image_obs], axis=0).astype(np.uint8)

    #     t0 = time.time()
    #     sample, encoded_decoded = self.sample_fn(image_obs, language_command, prompt_w=7.5, context_w=1.5)

    #     # sample = self.sample_fn(image_obs[0], language_command, prompt_w=7.5, context_w=1.5)

    #     t1 = time.time()
    #     # print(f"\t>>t1 - t0: {t1 - t0:.3f}")
    #     # return np.array(Image.fromarray(sample).resize((200, 200))).astype(np.uint8)

    #     samples = np.array([np.array(Image.fromarray(s).resize((200, 200))).astype(np.uint8) for s in sample])

    #     encoded_decoded = np.array([np.array(Image.fromarray(e).resize((200, 200))).astype(np.uint8) for e in encoded_decoded])

    #     if return_inference_time:
    #         return samples, encoded_decoded, t1 - t0
    #     else:
    #         return samples, encoded_decoded
        

    def vae_encode_decode(self, image_obs : np.ndarray, noise_scale=0, return_inference_time=False):
        # Resize image to 256x256
        
        image_obs = np.array(Image.fromarray(image_obs).resize((256, 256))).astype(np.uint8)

        # image_obs = np.stack([np.array(Image.fromarray(img).resize((256, 256))) for img in image_obs], axis=0).astype(np.uint8)

        t0 = time.time()
        sample = self.vae_encode_decode_fn(image_obs, noise_scale=noise_scale)
        t1 = time.time()
        # print(f"\t>>t1 - t0: {t1 - t0:.3f}")
        # return np.array(Image.fromarray(sample).resize((200, 200))).astype(np.uint8)

        samples = np.array([np.array(Image.fromarray(s).resize((200, 200))).astype(np.uint8) for s in sample])

        if return_inference_time:
            return samples, t1 - t0
        else:
            return samples
        

    

if __name__ == "__main__":
    model = DiffusionModel()