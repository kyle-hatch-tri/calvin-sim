import json
from jaxrl_m.vision import encoders
from jaxrl_m.data.calvin_dataset import CalvinDataset
import jax
from jaxrl_m.agents import agents
import numpy as np
import os
import orbax.checkpoint


# encoder_kwargs=dict(
#                     pooling_method="avg",
#                     add_spatial_coordinates=True,
#                     act="swish",
#                 )
    
gc_ddpm_bc_kwargs = dict(
        score_network_kwargs=dict(
            time_dim=32,
            num_blocks=3,
            dropout_rate=0.1,
            hidden_dim=256,
            use_layer_norm=True,
        ),
        #language_conditioned=True,
        early_goal_concat=True,
        shared_goal_encoder=True,
        use_proprio=False,
        beta_schedule="cosine",
        diffusion_steps=20,
        action_samples=1,
        repeat_last_step=0,
        learning_rate=3e-4,
        warmup_steps=2000,
        actor_decay_steps=int(2e6),
    )


gc_bc_kwargs = dict(
        network_kwargs=dict(
            dropout_rate=0.1,
            hidden_dims=[256, 256],
            use_layer_norm=True,
        ),
        policy_kwargs=dict(tanh_squash_distribution=False, 
                            state_dependent_std=False,
                        #    dropout=0.0,
                            ),
        #language_conditioned=True,
        early_goal_concat=True,
        shared_goal_encoder=True,
        use_proprio=False,
        learning_rate=3e-4,
        warmup_steps=2000,
    )


gc_iql_kwargs = dict(
        network_kwargs=dict(
            dropout_rate=0.1,
            hidden_dims=[256, 256],
            use_layer_norm=True,
        ),
        policy_kwargs=dict(tanh_squash_distribution=False, 
                            state_dependent_std=False,
                        #    dropout=0.0,
                            ),
        #language_conditioned=True,
        early_goal_concat=True,
        shared_goal_encoder=True,
        use_proprio=False,
        learning_rate=3e-4,
        warmup_steps=2000,

        actor_decay_steps=int(2e6),
        negative_proportion=0.0,
        shared_encoder=False,
        discount=0.95,
        expectile=0.9,
        temperature=1.0,
        target_update_rate=0.002,
        dropout_target_networks=True,
    )


gc_iql2_kwargs = gc_iql_kwargs.copy()
gc_iql2_kwargs["discount"] = 0.99

gc_iql3_kwargs = gc_iql_kwargs.copy()
gc_iql3_kwargs["target_update_rate"] = 0.005

gc_iql4_kwargs = gc_iql_kwargs.copy()
gc_iql4_kwargs["discount"] = 0.99
gc_iql4_kwargs["expectile"] = 0.7

gc_iql5_kwargs = gc_iql_kwargs.copy()
gc_iql5_kwargs["discount"] = 0.99
gc_iql5_kwargs["expectile"] = 0.7
gc_iql5_kwargs["temperature"] = 3


AGENT_KWARGS = dict(gc_ddpm_bc=gc_ddpm_bc_kwargs,
                    gc_bc=gc_bc_kwargs,
                    gc_iql=gc_iql_kwargs,
                    gc_iql2=gc_iql2_kwargs,
                    gc_iql3=gc_iql3_kwargs,
                    gc_iql4=gc_iql4_kwargs,
                    gc_iql5=gc_iql5_kwargs,)


class GCPolicy:
    def __init__(self, agent_type, resume_path, normalize_actions=False, use_temporal_ensembling=True):
        self.use_temporal_ensembling = use_temporal_ensembling

        print("agent_type:", agent_type)

        # We need to first create a dataset object to supply to the agent
        train_paths = [[
            "mini_dataset/0.tfrecord",
            "mini_dataset/1.tfrecord"
        ]]

        self.agent_type = agent_type

        if "ddpm" in self.agent_type:
            act_pred_horizon = 4
            obs_horizon = 1
        else:
            act_pred_horizon = None
            obs_horizon = None

        dataset_kwargs = dict(
            shuffle_buffer_size=25000,
            prefetch_num_batches=20,
            augment=True,
            augment_next_obs_goal_differently=False,
            augment_kwargs=dict(
                random_resized_crop=dict(scale=[0.8, 1.0], ratio=[0.9, 1.1]),
                random_brightness=[0.2],
                random_contrast=[0.8, 1.2],
                random_saturation=[0.8, 1.2],
                random_hue=[0.1],
                augment_order=[
                    "random_resized_crop",
                    "random_brightness",
                    "random_contrast",
                    "random_saturation",
                    "random_hue",
                ],
            ),
            # normalize_actions=False if ("noactnorm" in self.agent_type or "nam" in self.agent_type)else True, # TODO better way to handle this
            normalize_actions=normalize_actions,
            goal_relabeling_strategy="delta_goals",
            goal_relabeling_kwargs=dict(goal_delta=[0, 24]),
            relabel_actions=False,
            act_pred_horizon=act_pred_horizon,
            obs_horizon=obs_horizon,
        )

        print("normalize_actions:", normalize_actions)

        ACT_MEAN = [
            2.9842544e-04,
            -2.6099570e-04,
            -1.5863389e-04,
            5.8916201e-05,
            -4.4560504e-05,
            8.2349771e-04,
            9.4075650e-02,
        ]

        ACT_STD = [
            0.27278143,
            0.23548537,
            0.2196189,
            0.15881406,
            0.17537235,
            0.27875036,
            1.0049515,
        ]

        PROPRIO_MEAN = [ # We don't actually use proprio so we're using dummy values for this
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]

        PROPRIO_STD = [ # We don't actually use proprio so we're using dummy values for this
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
        ]

        ACTION_PROPRIO_METADATA = {
            "action": {
                "mean": ACT_MEAN,
                "std": ACT_STD,
                # TODO compute these
                "min": ACT_MEAN,
                "max": ACT_STD
            },
            # TODO compute these
            "proprio": {
                "mean": PROPRIO_MEAN,
                "std": PROPRIO_STD,
                "min": PROPRIO_MEAN,
                "max": PROPRIO_STD
            }
        }

        action_metadata = {
            "mean": ACT_MEAN,
            "std": ACT_STD,
        }

        train_data = CalvinDataset(
            train_paths,
            42,
            batch_size=256,
            num_devices=1,
            train=True,
            action_proprio_metadata=ACTION_PROPRIO_METADATA,
            sample_weights=None,
            **dataset_kwargs,
        )
        train_data_iter = train_data.iterator()
        example_batch = next(train_data_iter)

        # Next let's initialize the agent

        print("agent_type:", self.agent_type)

        agent_kwargs = AGENT_KWARGS[self.agent_type]
        # agent_kwargs=dict(
        #     score_network_kwargs=dict(
        #         time_dim=32,
        #         num_blocks=3,
        #         dropout_rate=0.1,
        #         hidden_dim=256,
        #         use_layer_norm=True,
        #     ),
        #     early_goal_concat=True,
        #     shared_goal_encoder=True,
        #     use_proprio=False,
        #     beta_schedule="cosine",
        #     diffusion_steps=20,
        #     action_samples=1,
        #     repeat_last_step=0,
        #     learning_rate=3e-4,
        #     warmup_steps=2000,
        #     actor_decay_steps=int(2e6),
        # )
            

        encoder_def = encoders["resnetv1-34-bridge"](**{"act" : "swish", "add_spatial_coordinates" : "true", "pooling_method" : "avg"})
        rng = jax.random.PRNGKey(42)
        rng, construct_rng = jax.random.split(rng)
        # agent = agents["gc_ddpm_bc"].create(
        
        
        agent_type_key = self.agent_type
        if "iql" in agent_type_key and agent_type_key[-1].isdigit():
            agent_type_key = agent_type_key[:-1]

        agent = agents[agent_type_key].create(
            rng=construct_rng,
            observations=example_batch["observations"],
            goals=example_batch["goals"],
            actions=example_batch["actions"],
            encoder_def=encoder_def,
            **agent_kwargs,
        )

        print("Loading checkpoint...") 
        # resume_path = os.getenv("GC_POLICY_CHECKPOINT")
        restored = orbax.checkpoint.PyTreeCheckpointer().restore(resume_path, item=agent)
        if agent is restored:
            raise FileNotFoundError(f"Cannot load checkpoint from {resume_path}")
        print("Checkpoint successfully loaded")
        agent = restored

        # save the loaded agent
        self.agent = agent
        self.action_statistics = action_metadata

        # Prepare action buffer for temporal ensembling
        self.action_buffer = np.zeros((4, 4, 7))
        self.action_buffer_mask = np.zeros((4, 4), dtype=bool)

        # self.idx = 1 ### DEBUG ###

    def reset(self):
        self.action_buffer = np.zeros((4, 4, 7))
        self.action_buffer_mask = np.zeros((4, 4), dtype=bool)

    def value_function_ranking(self, image_obs : np.ndarray, goal_images : np.ndarray):
        assert len(goal_images.shape) == len(image_obs.shape) + 1
        stacked_image_obs = np.repeat(image_obs[None], goal_images.shape[0], axis=0)
        v = self.agent.value_function({"image" : stacked_image_obs}, {"image" : goal_images})
        return np.array(v.tolist()) 

    def predict_action(self, image_obs : np.ndarray, goal_image : np.ndarray):
        if "ddpm" in self.agent_type:
            action = self.agent.sample_actions(
                                {"image" : image_obs[np.newaxis, ...]}, 
                                {"image" : goal_image}, 
                                seed=jax.random.PRNGKey(42), 
                                temperature=0.0, 
                            )
            action = np.array(action.tolist()) 
        else:
            action = self.agent.sample_actions(
                                {"image" : image_obs}, 
                                {"image" : goal_image}, 
                                seed=jax.random.PRNGKey(42), 
                                temperature=0.0, 
                            )
            action = np.array(action.tolist())

            # Since the t=[1,2,3] action predictions are never used,
            # can use the same buffer machinery for the one step policies 
            # by just tiling the action along a new time 4 times 
            action = action[None, :] * np.ones((4, 7))


        if self.use_temporal_ensembling:
            # action = np.ones_like(action) * self.idx ### DEBUG ###
            # # Along the action horizon dimension
            # action[1] *= 10
            # action[2] *= 100
            # # Along the action dim dimension
            # action[:, -2] *= -1
            # # action[:, -1] *= -10
            # self.idx += 1 ### DEBUG ###


            # print("\n\n" * 5 + "=" * 30)
            # print("\naction:", action)
            # print("action.shape:", action.shape)

            # Scale action
            #action = np.array(self.action_statistics["std"]) * action + np.array(self.action_statistics["mean"])

            # Shift action buffer
            self.action_buffer[1:, :, :] = self.action_buffer[:-1, :, :]
            self.action_buffer_mask[1:, :] = self.action_buffer_mask[:-1, :]
            self.action_buffer[:, :-1, :] = self.action_buffer[:, 1:, :]
            self.action_buffer_mask[:, :-1] = self.action_buffer_mask[:, 1:]
            self.action_buffer_mask = self.action_buffer_mask * np.array([[True, True, True, True],
                                                                        [True, True, True, False],
                                                                        [True, True, False, False],
                                                                        [True, False, False, False]], dtype=bool)

            # Add to action buffer
            self.action_buffer[0] = action
            self.action_buffer_mask[0] = np.array([True, True, True, True], dtype=bool)
            
            # Ensemble temporally to predict action
            action_prediction = np.sum(self.action_buffer[:, 0, :] * self.action_buffer_mask[:, 0:1], axis=0) / np.sum(self.action_buffer_mask[:, 0], axis=0)
            # action.shape = (action_horizon, action_dim)
            # self.action_buffer.shape = (buffer_length, action_horizon, action_dim)
            # self.action_buffer_mask.shape = (buffer_length, action_horizon)

            # print("\nself.action_buffer:", self.action_buffer)
            # print("self.action_buffer.shape:", self.action_buffer.shape)
            
            # print("\nself.action_buffer_mask:", self.action_buffer_mask)
            # print("self.action_buffer_mask.shape:", self.action_buffer_mask.shape)

            # print("\naction_prediction:", action_prediction)
            # print("action_prediction.shape:", action_prediction.shape)
        else:
            action_prediction = action[0]

        # Make gripper action either -1 or 1
        if action_prediction[-1] < 0:
            action_prediction[-1] = -1
        else:
            action_prediction[-1] = 1

        # print("\naction_prediction:", action_prediction)
        # print("action_prediction.shape:", action_prediction.shape)
        return action_prediction



"""
Think the action buffer stuff is doing this:

keeps a buffer of shape (4, 4, 7) = (buffer_length, action_horizon, action_dim)
predicts an action of shape (4, 7) = (action_horizon, action_dim)
Moves the buffer down along the first dim, and then inserts the new action into the first slot of dim=0


When computing the action, just takes the first action predicted from the last 4 action predictions (buffer), and averages that
Doesn't ever use the future action predictions

IE
each query of the policy net predicts an action for t=[0, 1, 2, 3]
Stores the last 4 action predictions for all of t=[0, 1, 2, 3]
When computing the action, just takes the t=0 action prediction from the last 4 action predictions (buffer), and averages that
"""