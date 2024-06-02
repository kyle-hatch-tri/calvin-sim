from functools import partial
from typing import Any

import copy 
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.core import FrozenDict

from jaxrl_m.common.common import JaxRLTrainState, ModuleDict, nonpytree_field
from jaxrl_m.common.encoding import LCGCEncodingWrapper
from jaxrl_m.common.typing import Batch, PRNGKey
from jaxrl_m.networks.actor_critic_nets import ValueCritic
from jaxrl_m.networks.mlp import MLP



# import os 
# import cv2
# def save_video(output_video_file, frames):
#      # Extract frame dimensions
#     height, width, _ = frames.shape[1:]

#     # Define the codec and create VideoWriter object
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can also use other codecs such as 'XVID'
#     fps = 30  # Adjust the frame rate as needed

#     os.makedirs(os.path.dirname(output_video_file), exist_ok=True)
#     video_writer = cv2.VideoWriter(output_video_file, fourcc, fps, (width, height))

#     # Write each frame to the video file
#     for frame in frames:
#         bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
#         video_writer.write(bgr_frame)

#     # Release the video writer object
#     video_writer.release()

class LCGCProgressVFAgent(flax.struct.PyTreeNode):
    debug_metrics_rng: PRNGKey
    state: JaxRLTrainState
    lr_schedule: Any = nonpytree_field()
    config: dict = nonpytree_field()
    
    @partial(jax.jit, static_argnames="pmap_axis")
    def update(self, batch: Batch, pmap_axis: str = None):

        def loss_fn(params, rng):
            # plain_lang = []
            # with open("./lcgc_progress_vf_debug/batch_language.txt", "r") as f:
            #     for line in f:
            #         plain_lang.append(line.strip())

            batch_size = batch["observations"]["image"].shape[0]

            pos_idx_end = int(batch_size * self.config["frac_pos"])

            neg_wrong_lang_goal_idx_start = pos_idx_end
            neg_wrong_lang_goal_idx_end = neg_wrong_lang_goal_idx_start + int(batch_size * self.config["frac_neg_wrong_lang"])

            neg_reverse_direction_idx_start = neg_wrong_lang_goal_idx_end
            neg_reverse_direction_idx_end = neg_reverse_direction_idx_start + int(batch_size * self.config["frac_neg_reverse_direction"])

            neg_wrong_goalimg_idx_start = neg_reverse_direction_idx_end

            pos_obs_images = batch["observations"]["image"][:pos_idx_end]
            pos_goal_images = batch["goals"]["image"][:pos_idx_end]
            pos_goal_language = batch["goals"]["language"][:pos_idx_end]
            # pos_plain_lang = plain_lang[:pos_idx_end]

            neg_wrong_lang_obs_images = batch["observations"]["image"][neg_wrong_lang_goal_idx_start:neg_wrong_lang_goal_idx_end]
            neg_wrong_lang_goal_images = batch["goals"]["image"][neg_wrong_lang_goal_idx_start:neg_wrong_lang_goal_idx_end]
            neg_wrong_lang_goal_language = batch["goals"]["language"][neg_wrong_lang_goal_idx_start:neg_wrong_lang_goal_idx_end]
            # neg_wrong_lang_plain_lang = plain_lang[neg_wrong_lang_goal_idx_start:neg_wrong_lang_goal_idx_end]

            # neg_reverse_direction_obs_images = batch["observations"]["image"][neg_reverse_direction_idx_start:neg_reverse_direction_idx_end] # these get defined later 
            # neg_reverse_direction_goal_images = batch["goals"]["image"][neg_reverse_direction_idx_start:neg_reverse_direction_idx_end]
            neg_reverse_direction_goal_language = batch["goals"]["language"][neg_reverse_direction_idx_start:neg_reverse_direction_idx_end]
            # neg_reverse_direction_plain_lang = plain_lang[neg_reverse_direction_idx_start:neg_reverse_direction_idx_end]

            neg_wrong_goalimg_obs_images = batch["observations"]["image"][neg_wrong_goalimg_idx_start:]
            neg_wrong_goalimg_goal_images = batch["goals"]["image"][neg_wrong_goalimg_idx_start:]
            neg_wrong_goalimg_goal_language = batch["goals"]["language"][neg_wrong_goalimg_idx_start:]
            # neg_wrong_goalimg_plain_lang = plain_lang[neg_wrong_goalimg_idx_start:]

            # print("\nself.config["frac_pos"]:", self.config["frac_pos"], " int(batch_size * self.config["frac_pos"]):", int(batch_size * self.config["frac_pos"]))
            # print("self.config["frac_neg_wrong_lang"]:", self.config["frac_neg_wrong_lang"], " int(batch_size * self.config["frac_neg_wrong_lang"]):", int(batch_size * self.config["frac_neg_wrong_lang"]))
            # print("self.config["frac_neg_reverse_direction"]:", self.config["frac_neg_reverse_direction"], " int(batch_size * self.config["frac_neg_reverse_direction"]):", int(batch_size * self.config["frac_neg_reverse_direction"]))
            # print("self.config["frac_neg_wrong_goalimg"]:", self.config["frac_neg_wrong_goalimg"], " int(batch_size * self.config["frac_neg_wrong_goalimg"]):", int(batch_size * self.config["frac_neg_wrong_goalimg"]))

            # print("\nbatch_size:", batch_size)
            # print("pos_idx_end:", pos_idx_end)
            # print("neg_wrong_lang_goal_idx_start:", neg_wrong_lang_goal_idx_start, " neg_wrong_lang_goal_idx_end:", neg_wrong_lang_goal_idx_end)
            # print("neg_reverse_direction_idx_start:", neg_reverse_direction_idx_start, " neg_reverse_direction_idx_end:", neg_reverse_direction_idx_end)
            # print("neg_wrong_goalimg_idx_start:", neg_wrong_goalimg_idx_start)

            # print("\npos_obs_images.shape[0]:", pos_obs_images.shape[0])
            # print("neg_wrong_lang_obs_images.shape[0]:", neg_wrong_lang_obs_images.shape[0])
            # print("neg_wrong_goalimg_obs_images.shape[0]:", neg_wrong_goalimg_obs_images.shape[0])





            # create negative examples where the language goals are randomly shuffled, so that the obs and goal do not match the language instruction (s_t, s_{t+k}, l) --> (s_t, s_{t+k}, l')
            # print("\nneg_wrong_lang_goal_language:", neg_wrong_lang_goal_language)
            # print("\nneg_wrong_lang_plain_lang:", neg_wrong_lang_plain_lang)
            neg_wrong_lang_goal_language_idxs = jnp.arange(neg_wrong_lang_goal_language.shape[0])
            # print("neg_wrong_lang_goal_language_idxs:", neg_wrong_lang_goal_language_idxs)
            rng, key = jax.random.split(rng)
            neg_wrong_lang_goal_language_idxs = jax.random.permutation(key, neg_wrong_lang_goal_language_idxs, axis=0)
            # print("neg_wrong_lang_goal_language_idxs:", neg_wrong_lang_goal_language_idxs)
            neg_wrong_lang_goal_language = neg_wrong_lang_goal_language[neg_wrong_lang_goal_language_idxs]
            # print("neg_wrong_lang_goal_language:", neg_wrong_lang_goal_language)
            # neg_wrong_lang_plain_lang = [neg_wrong_lang_plain_lang[idx] for idx in neg_wrong_lang_goal_language_idxs] ###DEBUG
            # print("neg_wrong_lang_plain_lang:", neg_wrong_lang_plain_lang)


            # create negative examples where the current observation image and the goal image are swapped (backwards progress). (s_t, s_{t+k}, l) --> (s_{t+k}, s_t, l)
            neg_reverse_direction_obs_images = batch["goals"]["image"][neg_reverse_direction_idx_start:neg_reverse_direction_idx_end] # these are deliberately swapped
            neg_reverse_direction_goal_images = batch["observations"]["image"][neg_reverse_direction_idx_start:neg_reverse_direction_idx_end] # these are deliberately swapped



            # create negative examples where the goal imgs are randomly shuffled, so that the goal image does not match the current obs and language instruction (s_t, s_{t+k}, l) --> (s_t, s_{t+k}', l)
            # print("\nneg_wrong_goalimg_goal_images[:, 0:5, 0, 0]:\n", neg_wrong_goalimg_goal_images[:, 0:5, 0, 0])
            neg_wrong_goalimg_goal_images_idxs = jnp.arange(neg_wrong_goalimg_goal_images.shape[0])
            # print("neg_wrong_goalimg_goal_images_idxs:\n", neg_wrong_goalimg_goal_images_idxs)
            rng, key = jax.random.split(rng)
            neg_wrong_goalimg_goal_images_idxs = jax.random.permutation(key, neg_wrong_goalimg_goal_images_idxs, axis=0)
            # print("neg_wrong_goalimg_goal_images_idxs:\n", neg_wrong_goalimg_goal_images_idxs)
            neg_wrong_goalimg_goal_images = neg_wrong_goalimg_goal_images[neg_wrong_goalimg_goal_images_idxs]
            # print("neg_wrong_goalimg_goal_images[:, 0:5, 0, 0]:\n", neg_wrong_goalimg_goal_images[:, 0:5, 0, 0])
            



            obs_images = jnp.concatenate([pos_obs_images, neg_wrong_lang_obs_images, neg_reverse_direction_obs_images, neg_wrong_goalimg_obs_images], axis=0)
            goal_images = jnp.concatenate([pos_goal_images, neg_wrong_lang_goal_images, neg_reverse_direction_goal_images, neg_wrong_goalimg_goal_images], axis=0)
            goal_language = jnp.concatenate([pos_goal_language, neg_wrong_lang_goal_language, neg_reverse_direction_goal_language, neg_wrong_goalimg_goal_language], axis=0)
            # all_plain_lang = pos_plain_lang + neg_wrong_lang_plain_lang + neg_reverse_direction_plain_lang + neg_wrong_goalimg_plain_lang


            new_batch = {"observations":{"image":obs_images},
                         "goals":{"image":goal_images,
                                  "language":goal_language}}
            
            batch_size_pos = pos_obs_images.shape[0]
            batch_size_neg = batch_size - batch_size_pos
            labels_pos = jnp.ones((batch_size_pos,), dtype=int)
            # labels_neg = jnp.zeros((batch_size_neg,), dtype=int)
            if self.config["loss_fn"] == "bce":
                labels_neg = jnp.zeros((batch_size_neg,), dtype=int)
            elif self.config["loss_fn"] == "hinge":
                labels_neg = jnp.ones((batch_size_neg,), dtype=int) * -1
            else:
                raise ValueError(f'Unsupported loss_fn: {self.config["loss_fn"]}')
            
            labels = jnp.concatenate([labels_pos, labels_neg], axis=0)

            # jax.debug.print("[update] labels: {labels}", labels=labels)



            
            rng, key = jax.random.split(rng)
            logits = self.state.apply_fn(
                {"params": params},
                (new_batch["observations"], new_batch["goals"]),
                # temperature=1.0,
                train=True,
                rngs={"dropout": key},
                name="value",
            )


            pos_idx_end = int(batch_size * self.config["frac_pos"])

            neg_wrong_lang_goal_idx_start = pos_idx_end
            neg_wrong_lang_goal_idx_end = neg_wrong_lang_goal_idx_start + int(batch_size * self.config["frac_neg_wrong_lang"])

            neg_reverse_direction_idx_start = neg_wrong_lang_goal_idx_end
            neg_reverse_direction_idx_end = neg_reverse_direction_idx_start + int(batch_size * self.config["frac_neg_reverse_direction"])

            neg_wrong_goalimg_idx_start = neg_reverse_direction_idx_end

            # combined = np.concatenate([obs_images, goal_images], axis=2)
            # new_frames = []
            # for j, frame in enumerate(combined):
            #     # Choose font and scale
            #     font = cv2.FONT_HERSHEY_SIMPLEX
            #     font_scale = 0.5
            #     font_color = (0, 255, 0)  # White color in BGR
            #     line_type = 2  # Line thickness

            #     # Add text to the image
            #     frame = cv2.putText(frame, f'[{labels[j]}]', (10, 15), font, font_scale, font_color, line_type)

            #     if j < pos_idx_end:
            #         frame = cv2.putText(frame, f'pos', (10, 30), font, font_scale, font_color, line_type)
            #     elif j >= neg_wrong_lang_goal_idx_start and j < neg_wrong_lang_goal_idx_end:
            #         frame = cv2.putText(frame, f'wrong lang', (10, 30), font, font_scale, font_color, line_type)
            #     elif j >= neg_reverse_direction_idx_start and j < neg_reverse_direction_idx_end:
            #         frame = cv2.putText(frame, f'reverse', (10, 30), font, font_scale, font_color, line_type)
            #     elif j >= neg_wrong_goalimg_idx_start:
            #         frame = cv2.putText(frame, f'wrong goalimg', (10, 30), font, font_scale, font_color, line_type)
            #     else:
            #         raise ValueError(f"{j} doesn't fit into any of the categories somehow.")
                

            #     frame = cv2.putText(frame, f'{all_plain_lang[j]}', (10, 180), font, 0.25, font_color, 1)

            #     new_frames.append(frame)

            # save_video("./lcgc_progress_vf_debug/update_debug.mp4", np.array(new_frames))

            if self.config["loss_fn"] == "bce":
                loss = optax.sigmoid_binary_cross_entropy(logits, labels)
            elif self.config["loss_fn"] == "hinge":
                loss = optax.hinge_loss(logits, labels)
            else:
                raise ValueError(f'Unsupported loss_fn: {self.config["loss_fn"]}')

            loss_pos = loss[:pos_idx_end].mean()
            loss_neg_wrong_lang = loss[neg_wrong_lang_goal_idx_start:neg_wrong_lang_goal_idx_end].mean()
            loss_neg_reverse_direction = loss[neg_reverse_direction_idx_start:neg_reverse_direction_idx_end].mean()
            loss_neg_wrong_goalimg = loss[neg_wrong_goalimg_idx_start:].mean()

            logits_pos = logits[:pos_idx_end].mean()
            logits_neg_wrong_lang = logits[neg_wrong_lang_goal_idx_start:neg_wrong_lang_goal_idx_end].mean()
            logits_neg_reverse_direction = logits[neg_reverse_direction_idx_start:neg_reverse_direction_idx_end].mean()
            logits_neg_wrong_goalimg = logits[neg_wrong_goalimg_idx_start:].mean()

            

            


            

            return (
                loss.mean(),
                {
                    "loss": loss.mean(),
                    "loss_pos": loss_pos,
                    "loss_neg_wrong_lang": loss_neg_wrong_lang,
                    "loss_neg_reverse_direction": loss_neg_reverse_direction,
                    "loss_neg_wrong_goalimg": loss_neg_wrong_goalimg,

                    "logits": logits.mean(),
                    "logits_pos": logits_pos,
                    "logits_neg_wrong_lang": logits_neg_wrong_lang,
                    "logits_neg_reverse_direction": logits_neg_reverse_direction,
                    "logits_neg_wrong_goalimg": logits_neg_wrong_goalimg,

                },
            )

        # compute gradients and update params
        new_state, info = self.state.apply_loss_fns(
            loss_fn, pmap_axis=pmap_axis, has_aux=True
        )

        # log learning rates
        info["lr"] = self.lr_schedule(self.state.step)

        return self.replace(state=new_state), info

    @jax.jit
    def get_debug_metrics(self, batch, **kwargs):
        # v = self.state.apply_fn(
        #     {"params": self.state.params},
        #     (batch["observations"], batch["goals"]),
        #     # temperature=1.0,
        #     name="value",
        # )
        # pi_actions = dist.mode()
        # log_probs = dist.log_prob(batch["actions"])
        # mse = ((pi_actions - batch["actions"]) ** 2).sum(-1)

        # return {"mse": mse, "log_probs": log_probs, "pi_actions": pi_actions}
        batch_size = batch["observations"]["image"].shape[0]

        pos_idx_end = int(batch_size * self.config["frac_pos"])

        neg_wrong_lang_goal_idx_start = pos_idx_end
        neg_wrong_lang_goal_idx_end = neg_wrong_lang_goal_idx_start + int(batch_size * self.config["frac_neg_wrong_lang"])

        neg_reverse_direction_idx_start = neg_wrong_lang_goal_idx_end
        neg_reverse_direction_idx_end = neg_reverse_direction_idx_start + int(batch_size * self.config["frac_neg_reverse_direction"])

        neg_wrong_goalimg_idx_start = neg_reverse_direction_idx_end

        pos_obs_images = batch["observations"]["image"][:pos_idx_end]
        pos_goal_images = batch["goals"]["image"][:pos_idx_end]
        pos_goal_language = batch["goals"]["language"][:pos_idx_end]

        neg_wrong_lang_obs_images = batch["observations"]["image"][neg_wrong_lang_goal_idx_start:neg_wrong_lang_goal_idx_end]
        neg_wrong_lang_goal_images = batch["goals"]["image"][neg_wrong_lang_goal_idx_start:neg_wrong_lang_goal_idx_end]
        neg_wrong_lang_goal_language = batch["goals"]["language"][neg_wrong_lang_goal_idx_start:neg_wrong_lang_goal_idx_end]

        # neg_reverse_direction_obs_images = batch["observations"]["image"][neg_reverse_direction_idx_start:neg_reverse_direction_idx_end] # these get defined later 
        # neg_reverse_direction_goal_images = batch["goals"]["image"][neg_reverse_direction_idx_start:neg_reverse_direction_idx_end]
        neg_reverse_direction_goal_language = batch["goals"]["language"][neg_reverse_direction_idx_start:neg_reverse_direction_idx_end]

        neg_wrong_goalimg_obs_images = batch["observations"]["image"][neg_wrong_goalimg_idx_start:]
        neg_wrong_goalimg_goal_images = batch["goals"]["image"][neg_wrong_goalimg_idx_start:]
        neg_wrong_goalimg_goal_language = batch["goals"]["language"][neg_wrong_goalimg_idx_start:]

        # print("\nself.config["frac_pos"]:", self.config["frac_pos"], " int(batch_size * self.config["frac_pos"]):", int(batch_size * self.config["frac_pos"]))
        # print("self.config["frac_neg_wrong_lang"]:", self.config["frac_neg_wrong_lang"], " int(batch_size * self.config["frac_neg_wrong_lang"]):", int(batch_size * self.config["frac_neg_wrong_lang"]))
        # print("self.config["frac_neg_reverse_direction"]:", self.config["frac_neg_reverse_direction"], " int(batch_size * self.config["frac_neg_reverse_direction"]):", int(batch_size * self.config["frac_neg_reverse_direction"]))
        # print("self.config["frac_neg_wrong_goalimg"]:", self.config["frac_neg_wrong_goalimg"], " int(batch_size * self.config["frac_neg_wrong_goalimg"]):", int(batch_size * self.config["frac_neg_wrong_goalimg"]))

        # print("\nbatch_size:", batch_size)
        # print("pos_idx_end:", pos_idx_end)
        # print("neg_wrong_lang_goal_idx_start:", neg_wrong_lang_goal_idx_start, " neg_wrong_lang_goal_idx_end:", neg_wrong_lang_goal_idx_end)
        # print("neg_reverse_direction_idx_start:", neg_reverse_direction_idx_start, " neg_reverse_direction_idx_end:", neg_reverse_direction_idx_end)
        # print("neg_wrong_goalimg_idx_start:", neg_wrong_goalimg_idx_start)

        # print("\npos_obs_images.shape[0]:", pos_obs_images.shape[0])
        # print("neg_wrong_lang_obs_images.shape[0]:", neg_wrong_lang_obs_images.shape[0])
        # print("neg_wrong_goalimg_obs_images.shape[0]:", neg_wrong_goalimg_obs_images.shape[0])





        # create negative examples where the language goals are randomly shuffled, so that the obs and goal do not match the language instruction (s_t, s_{t+k}, l) --> (s_t, s_{t+k}, l')
        # print("\nneg_wrong_lang_goal_language:", neg_wrong_lang_goal_language)
        neg_wrong_lang_goal_language_idxs = jnp.arange(neg_wrong_lang_goal_language.shape[0])
        # print("neg_wrong_lang_goal_language_idxs:", neg_wrong_lang_goal_language_idxs)
        rng, key = jax.random.split(self.debug_metrics_rng)
        neg_wrong_lang_goal_language_idxs = jax.random.permutation(key, neg_wrong_lang_goal_language_idxs, axis=0)
        # print("neg_wrong_lang_goal_language_idxs:", neg_wrong_lang_goal_language_idxs)
        neg_wrong_lang_goal_language = neg_wrong_lang_goal_language[neg_wrong_lang_goal_language_idxs]
        # print("neg_wrong_lang_goal_language:", neg_wrong_lang_goal_language)



        # create negative examples where the current observation image and the goal image are swapped (backwards progress). (s_t, s_{t+k}, l) --> (s_{t+k}, s_t, l)
        neg_reverse_direction_obs_images = batch["goals"]["image"][neg_reverse_direction_idx_start:neg_reverse_direction_idx_end] # these are deliberately swapped
        neg_reverse_direction_goal_images = batch["observations"]["image"][neg_reverse_direction_idx_start:neg_reverse_direction_idx_end] # these are deliberately swapped



        # create negative examples where the goal imgs are randomly shuffled, so that the goal image does not match the current obs and language instruction (s_t, s_{t+k}, l) --> (s_t, s_{t+k}', l)
        # print("\nneg_wrong_goalimg_goal_images[:, 0:5, 0, 0]:\n", neg_wrong_goalimg_goal_images[:, 0:5, 0, 0])
        neg_wrong_goalimg_goal_images_idxs = jnp.arange(neg_wrong_goalimg_goal_images.shape[0])
        # print("neg_wrong_goalimg_goal_images_idxs:\n", neg_wrong_goalimg_goal_images_idxs)
        rng, key = jax.random.split(rng)
        neg_wrong_goalimg_goal_images_idxs = jax.random.permutation(key, neg_wrong_goalimg_goal_images_idxs, axis=0)
        # print("neg_wrong_goalimg_goal_images_idxs:\n", neg_wrong_goalimg_goal_images_idxs)
        neg_wrong_goalimg_goal_images = neg_wrong_goalimg_goal_images[neg_wrong_goalimg_goal_images_idxs]
        # print("neg_wrong_goalimg_goal_images[:, 0:5, 0, 0]:\n", neg_wrong_goalimg_goal_images[:, 0:5, 0, 0])
        



        obs_images = jnp.concatenate([pos_obs_images, neg_wrong_lang_obs_images, neg_reverse_direction_obs_images, neg_wrong_goalimg_obs_images], axis=0)
        goal_images = jnp.concatenate([pos_goal_images, neg_wrong_lang_goal_images, neg_reverse_direction_goal_images, neg_wrong_goalimg_goal_images], axis=0)
        goal_language = jnp.concatenate([pos_goal_language, neg_wrong_lang_goal_language, neg_reverse_direction_goal_language, neg_wrong_goalimg_goal_language], axis=0)

        new_batch = {"observations":{"image":obs_images},
                        "goals":{"image":goal_images,
                                "language":goal_language}}
        
        batch_size_pos = pos_obs_images.shape[0]
        batch_size_neg = batch_size - batch_size_pos
        labels_pos = jnp.ones((batch_size_pos,), dtype=int)

        if self.config["loss_fn"] == "bce":
            labels_neg = jnp.zeros((batch_size_neg,), dtype=int)
        elif self.config["loss_fn"] == "hinge":
            labels_neg = jnp.ones((batch_size_neg,), dtype=int) * -1
        else:
            raise ValueError(f'Unsupported loss_fn: {self.config["loss_fn"]}')

        labels = jnp.concatenate([labels_pos, labels_neg], axis=0)

        # jax.debug.print("[update] labels: {labels}", labels=labels)
        
        rng, key = jax.random.split(rng)
        logits = self.state.apply_fn(
            {"params": self.state.params},
            (new_batch["observations"], new_batch["goals"]),
            # temperature=1.0,
            # train=True,
            # rngs={"dropout": key},
            name="value",
        )


        pos_idx_end = int(batch_size * self.config["frac_pos"])

        neg_wrong_lang_goal_idx_start = pos_idx_end
        neg_wrong_lang_goal_idx_end = neg_wrong_lang_goal_idx_start + int(batch_size * self.config["frac_neg_wrong_lang"])

        neg_reverse_direction_idx_start = neg_wrong_lang_goal_idx_end
        neg_reverse_direction_idx_end = neg_reverse_direction_idx_start + int(batch_size * self.config["frac_neg_reverse_direction"])

        neg_wrong_goalimg_idx_start = neg_reverse_direction_idx_end

        if self.config["loss_fn"] == "bce":
            loss = optax.sigmoid_binary_cross_entropy(logits, labels)
        elif self.config["loss_fn"] == "hinge":
            loss = optax.hinge_loss(logits, labels)
        else:
            raise ValueError(f'Unsupported loss_fn: {self.config["loss_fn"]}')

        loss_pos = loss[:pos_idx_end].mean()
        loss_neg_wrong_lang = loss[neg_wrong_lang_goal_idx_start:neg_wrong_lang_goal_idx_end].mean()
        loss_neg_reverse_direction = loss[neg_reverse_direction_idx_start:neg_reverse_direction_idx_end].mean()
        loss_neg_wrong_goalimg = loss[neg_wrong_goalimg_idx_start:].mean()

        logits_pos = logits[:pos_idx_end].mean()
        logits_neg_wrong_lang = logits[neg_wrong_lang_goal_idx_start:neg_wrong_lang_goal_idx_end].mean()
        logits_neg_reverse_direction = logits[neg_reverse_direction_idx_start:neg_reverse_direction_idx_end].mean()
        logits_neg_wrong_goalimg = logits[neg_wrong_goalimg_idx_start:].mean()
        

        return {   
                "loss": loss.mean(),
                "loss_pos": loss_pos,
                "loss_neg_wrong_lang": loss_neg_wrong_lang,
                "loss_neg_reverse_direction": loss_neg_reverse_direction,
                "loss_neg_wrong_goalimg": loss_neg_wrong_goalimg,

                "logits": logits.mean(),
                "logits_pos": logits_pos,
                "logits_neg_wrong_lang": logits_neg_wrong_lang,
                "logits_neg_reverse_direction": logits_neg_reverse_direction,
                "logits_neg_wrong_goalimg": logits_neg_wrong_goalimg,
            }
    

    @jax.jit
    def value_function(self, observations, goals):
        logits = self.state.apply_fn(
            {"params": self.state.params},
            (observations, goals),
            # temperature=1.0,
            # train=True,
            # rngs={"dropout": key},
            name="value",
        )

        return logits


# value_function({"image" : stacked_image_obs}, {"image" : goal_images, "language":stacked_language_instruction})

    @classmethod
    def create(
        cls,
        rng: PRNGKey,
        observations: FrozenDict,
        actions: jnp.ndarray,
        goals: FrozenDict,
        # Model architecture
        encoder_def: nn.Module,
        shared_goal_encoder: bool = True,
        early_goal_concat: bool = False,
        use_proprio: bool = False,
        network_kwargs: dict = {"hidden_dims": [256, 256]},
        # policy_kwargs: dict = {
        #     "tanh_squash_distribution": False,
        #     "state_dependent_std": False,
        #     "dropout": 0.0,
        # },
        # Optimizer
        learning_rate: float = 3e-4,
        warmup_steps: int = 1000,
        decay_steps: int = 1000000,
        frac_pos: float = 0.5,
        frac_neg_wrong_lang: float = 0.2,
        frac_neg_reverse_direction: float = 0.2,
        frac_neg_wrong_goalimg: float = 0.1, # not directly used

        loss_fn: str = "bce",
    ):
        if early_goal_concat:
            # passing None as the goal encoder causes early goal concat
            goal_encoder_def = None
        else:
            if shared_goal_encoder:
                goal_encoder_def = encoder_def
            else:
                goal_encoder_def = copy.deepcopy(encoder_def)

        encoder_def = LCGCEncodingWrapper(
            encoder=encoder_def,
            goal_encoder=goal_encoder_def,
            use_proprio=use_proprio,
            stop_gradient=False,
        )

        network_kwargs["activate_final"] = False #?
        networks = {
            # "actor": Policy(
            #     encoder_def,
            #     MLP(**network_kwargs),
            #     action_dim=actions.shape[-1],
            #     **policy_kwargs
            # )
            "value": ValueCritic(encoder_def, MLP(**network_kwargs)),
        }

        model_def = ModuleDict(networks)

        lr_schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=learning_rate,
            warmup_steps=warmup_steps,
            decay_steps=decay_steps,
            end_value=0.0,
        )
        tx = optax.adam(lr_schedule)

        rng, init_rng = jax.random.split(rng)
        params = model_def.init(init_rng, value=[(observations, goals)])["params"]

        rng, create_rng = jax.random.split(rng)
        state = JaxRLTrainState.create(
            apply_fn=model_def.apply,
            params=params,
            txs=tx,
            target_params=params,
            rng=create_rng,
        )


        rng, debug_metrics_rng = jax.random.split(rng)

        assert abs(frac_pos + frac_neg_wrong_lang + frac_neg_reverse_direction + frac_neg_wrong_goalimg - 1.0) < 1e-5, f"frac_pos: {frac_pos}, frac_neg_wrong_lang: {frac_neg_wrong_lang}, frac_neg_reverse_direction: {frac_neg_reverse_direction}, frac_neg_wrong_goalimg: {frac_neg_wrong_goalimg} | frac_pos + frac_neg_wrong_lang + frac_neg_reverse_direction + frac_neg_wrong_goalimg = {frac_pos + frac_neg_wrong_lang + frac_neg_reverse_direction + frac_neg_wrong_goalimg}"
        assert loss_fn in ["bce", "hinge"], f"Unsupported loss_fn: {loss_fn}"

        print("loss_fn:", loss_fn)


        config = flax.core.FrozenDict(
            dict(
                frac_pos=frac_pos,
                frac_neg_wrong_lang=frac_neg_wrong_lang,
                frac_neg_reverse_direction=frac_neg_reverse_direction,
                frac_neg_wrong_goalimg=frac_neg_wrong_goalimg,

                loss_fn=loss_fn,
            )
        )

        return cls(debug_metrics_rng, state, lr_schedule, config)


"""
How to print out the language instructions for debugging? 


Debugging options:
- could change all the language instructions for the true things to be the same language encoding, and all the negative ones 
    to be the same language encoding


try both with and w/o early goal concat
use dropout? 

Get rid of the load language thing
- except that this only matters for action chunking?
Make something that samples forward and backward? Or just do that in the loss function? 

Run with load language?
"""


            # batch_size_pos = int(batch_size * self.config["frac_pos"])
            # pos_obs_images = batch["observations"]["image"][:pos_idx_end]
            # pos_goal_images = batch["goals"]["image"][:pos_idx_end]
            # pos_goal_language = batch["goals"]["language"][:pos_idx_end]

            # neg_obs_images = batch["observations"]["image"][batch_size_pos:]
            # neg_goal_images = batch["goals"]["image"][batch_size_pos:]
            # neg_goal_language = batch["goals"]["language"][batch_size_pos:]

            # # batch_size_neg = neg_obs_images.shape[0]
            # # # batch_size_neg_wrong_lang = batch_size_neg // 2
            # # batch_size_neg_wrong_lang = int(batch_size_neg * self.frac_of_neg_wrong_lang)

            # # create negative examples where the language goals are randomly shuffled, so that the obs and goal do not match the language instruction (s_t, s_{t+k}, l) --> (s_t, s_{t+k}, l')
            # neg_wrong_lang_obs_images = neg_obs_images[:batch_size_neg_wrong_lang]
            # neg_wrong_lang_goal_images = neg_goal_images[:batch_size_neg_wrong_lang]
            # neg_wrong_lang_goal_language = neg_goal_language[:batch_size_neg_wrong_lang]
            # neg_wrong_lang_goal_language_idxs = jnp.arange(batch_size_neg_wrong_lang)
            # import ipdb; ipdb.set_trace()
            # neg_wrong_lang_goal_language_idxs = jax.random.permutation(key, neg_wrong_lang_goal_language_idxs, axis=0)
            # neg_wrong_lang_goal_language = neg_wrong_lang_goal_language[neg_wrong_lang_goal_language_idxs]

            # # create negative examples where the current observation image and the goal image are swapped (backwards progress). (s_t, s_{t+k}, l) --> (s_{t+k}, s_t, l)
            # neg_reverse_direction_obs_images = neg_goal_images[batch_size_neg_wrong_lang:] # these are deliberately swapped
            # neg_reverse_direction_goal_images = neg_obs_images[batch_size_neg_wrong_lang:] # these are deliberately swapped
            # neg_reverse_direction_goal_language = neg_goal_language[batch_size_neg_wrong_lang:]

            # obs_images = jnp.concatenate([pos_obs_images, neg_wrong_lang_obs_images, neg_reverse_direction_obs_images], axis=0)
            # goal_images = jnp.concatenate([pos_goal_images, neg_wrong_lang_goal_images, neg_reverse_direction_goal_images], axis=0)
            # goal_language = jnp.concatenate([pos_goal_language, neg_wrong_lang_goal_language, neg_reverse_direction_goal_language], axis=0)

            # new_batch = {"observations":{"image":obs_images},
            #              "goals":{"image":goal_images,
            #                       "language":goal_language}}
            
            # labels_pos = jnp.ones((batch_size_pos,), dtype=int)
            # labels_neg = jnp.zeros((batch_size_neg,), dtype=int)
            # labels = jnp.concatenate([labels_pos, labels_neg], axis=0)