import os
import numpy as np 
from glob import glob 
from tqdm import tqdm, trange
import cv2
import requests
import base64
import io
from PIL import Image
from itertools import combinations
import time 
from collections import defaultdict
from multiprocessing import Pool, Manager


def visualize_outputs(args):
    frames = []

    if "timestep" in args.ep_dir:
        timestep_dirs = [args.ep_dir]
        
    else:
        timestep_dirs = sorted(glob(os.path.join(args.ep_dir, "unfiltered_goal_images", "timestep_*")))
        

    print("timestep_dirs:", timestep_dirs)

    for timestep_dir in timestep_dirs:

        goal_images = []
        goal_image_files = sorted(glob(os.path.join(timestep_dir, "goal_image_*.png")))
        
        for i, goal_image_file in enumerate(goal_image_files):
            
            goal_image = cv2.imread(goal_image_file)
            goal_image = goal_image[..., ::-1]
            goal_images.append(goal_image)

        goal_images = np.stack(goal_images, axis=0)

        
        rgb_image_obs = cv2.imread(os.path.join(timestep_dir, "rgb_obs.png"))
        rgb_image_obs = rgb_image_obs[..., ::-1]

        with open(os.path.join(os.path.dirname(os.path.dirname(timestep_dir)), "language_task.txt"), "r") as f:
            _ = f.readline()
            language_goal = f.readline().strip()

        best_goal_img_idx, chat_gpt_answer, yes_goal_image_idxs = ask_chat_gpt2(rgb_image_obs, goal_images, language_goal, timestep_dir)


        # sorted_idxs = np.argsort(v)[::-1]

        # ordered_goal_images = goal_images[sorted_idxs]
        # ordered_vs = v[sorted_idxs]

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_color = (0, 255, 0)  # White color in BGR
        line_type = 2  # Line thickness

        best_img = goal_images[best_goal_img_idx]


        
        if best_goal_img_idx in yes_goal_image_idxs:
            best_img = cv2.putText(best_img, f'yes img', (10, 15), font, font_scale, font_color, line_type)
            
        else:
            best_img = cv2.putText(best_img, f'no img', (10, 15), font, font_scale, font_color, line_type)

        best_img = cv2.putText(best_img, f'best_goal_img_idx {best_goal_img_idx}', (10, 50), font, font_scale, font_color, line_type)

        # rgb_image_obs = cv2.putText(rgb_image_obs, f'current obs', (10, 15), font, font_scale, font_color, line_type)

        frame = [rgb_image_obs, best_img]

        
        for i in range(goal_images.shape[0]):
            if i == best_goal_img_idx:
                continue 
            img = goal_images[i]

            if i in yes_goal_image_idxs:
                img = cv2.putText(img, f'yes img', (10, 15), font, font_scale, font_color, line_type)
            else:
                img = cv2.putText(img, f'no img', (10, 15), font, font_scale, font_color, line_type)
            frame.append(img)

        frame = np.concatenate(frame, axis=1)

        frame = cv2.putText(frame, f'"{chat_gpt_answer}"', (10, 185), font, font_scale, font_color, line_type)

        frames.append(frame)


    if len(frames) > 1:
        print("saving video")
        save_video("./goal_images_chat_gpt/traj.mp4", np.array(frames))
    else:
        print("saving image")
        assert len(frames) == 1, f"len(frames): {len(frames)}"
        os.makedirs("./goal_images_chat_gpt", exist_ok=True)
        cv2.imwrite("./goal_images_chat_gpt/traj.png", np.array(frames[0])[..., ::-1])

def visualize_outputs2(args):
    frames = []

    if "timestep" in args.ep_dir:
        timestep_dirs = [args.ep_dir]
        
    else:
        timestep_dirs = sorted(glob(os.path.join(args.ep_dir, "unfiltered_goal_images", "timestep_*")))
        

    print("timestep_dirs:", timestep_dirs)

    savedir = "goal_images_chat_gpt"
    os.makedirs(savedir, exist_ok=True)

    for timestep_dir in timestep_dirs:

        timestep_no = timestep_dir.split("_")[-1]

        goal_images = []
        goal_image_files = sorted(glob(os.path.join(timestep_dir, "goal_image_*.png")))
        
        for i, goal_image_file in enumerate(goal_image_files):
            
            goal_image = cv2.imread(goal_image_file)
            goal_image = goal_image[..., ::-1]
            goal_image = cv2.resize(goal_image, (512, 512))
            goal_images.append(goal_image)

        goal_images = np.stack(goal_images, axis=0)

        
        rgb_image_obs = cv2.imread(os.path.join(timestep_dir, "rgb_obs.png"))
        rgb_image_obs = rgb_image_obs[..., ::-1]
        rgb_image_obs = cv2.resize(rgb_image_obs, (512, 512))

        with open(os.path.join(os.path.dirname(os.path.dirname(timestep_dir)), "language_task.txt"), "r") as f:
            _ = f.readline()
            language_goal = f.readline().strip()

        answers, win_counts = ask_chat_gpt3(rgb_image_obs, goal_images, language_goal, timestep_dir)



        frame = [rgb_image_obs]

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_color = (0, 255, 0)  # White color in BGR
        line_type = 2  # Line thickness


        max_win_count = 0
        for count in win_counts.values():
            if count > max_win_count:
                max_win_count = count 

        
        for i in range(goal_images.shape[0]):
            img = goal_images[i]

            img = cv2.putText(img, f'[{i}]    win_count: {win_counts[i]}', (25, 25), font, font_scale, font_color, line_type)

            if win_counts[i] == max_win_count:
                img = cv2.rectangle(img, (0, 0), (img.shape[1] - 1, img.shape[0] - 1), font_color, 10)

            frame.append(img)

        frame = np.concatenate(frame, axis=1)
        frames.append(frame)

        # with open(os.path.join(savedir, f"chat_gpt_answers_{timestep_no}.txt"), "w") as f:
        #     for i, answer in enumerate(answers):
        #         f.write("=" * 30 + f" goal image {i} " + "=" * 30 + "\n")
        #         f.write(f"{answer}\n\n\n")
        print(f"WIN COUNTS")
        for i, count in win_counts.items():
            print(f"\tgoal image {i}: {count}")

        with open(os.path.join(savedir, f"chat_gpt_answers_{timestep_no}.txt"), "w") as f:
            f.write(f"WIN COUNTS\n")
            for i, count in win_counts.items():
                f.write(f"\tgoal image {i}: {count}\n")

            f.write("\n\n\n")

            for (i, j), answer in answers.items():
                f.write("=" * 30 + f" goal images ({i}, {j}) " + "=" * 30 + "\n")
                f.write(f"{answer}\n\n\n")

            


    
    if len(frames) > 1:
        print("saving video")
        save_video(f"{savedir}/traj.mp4", np.array(frames))
    else:
        print("saving image")
        assert len(frames) == 1, f"len(frames): {len(frames)}"
        cv2.imwrite(f"{savedir}/timestep_{timestep_no}.png", np.array(frames[0])[..., ::-1])


def visualize_outputs3(args):
    frames = []

    if "timestep" in args.ep_dir:
        timestep_dirs = [args.ep_dir]
        
    else:
        timestep_dirs = sorted(glob(os.path.join(args.ep_dir, "unfiltered_goal_images", "timestep_*")))
        

    print("timestep_dirs:", timestep_dirs)

    savedir = "goal_images_chat_gpt"
    os.makedirs(savedir, exist_ok=True)

    for timestep_dir in tqdm(timestep_dirs):

        timestep_no = timestep_dir.split("_")[-1]

        goal_images = []
        goal_image_files = sorted(glob(os.path.join(timestep_dir, "goal_image_*.png")))
        
        for i, goal_image_file in enumerate(goal_image_files):
            
            goal_image = cv2.imread(goal_image_file)
            goal_image = goal_image[..., ::-1]
            goal_image = cv2.resize(goal_image, (512, 512))
            goal_images.append(goal_image)

        goal_images = np.stack(goal_images, axis=0)

        
        rgb_image_obs = cv2.imread(os.path.join(timestep_dir, "rgb_obs.png"))
        rgb_image_obs = rgb_image_obs[..., ::-1]
        rgb_image_obs = cv2.resize(rgb_image_obs, (512, 512))

        with open(os.path.join(os.path.dirname(os.path.dirname(timestep_dir)), "language_task.txt"), "r") as f:
            _ = f.readline()
            language_goal = f.readline().strip()

        win_counts, infos, total_query_time = ask_chat_gpt5(rgb_image_obs, goal_images, language_goal, timestep_dir)



        frame = [rgb_image_obs]

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_color = (0, 255, 0)  # White color in BGR
        line_type = 2  # Line thickness


        max_win_count = 0
        for count in win_counts.values():
            if count > max_win_count:
                max_win_count = count 

        
        for i in range(goal_images.shape[0]):
            img = goal_images[i]

            img = cv2.putText(img, f'[{i}]    win_count: {win_counts[i]}', (25, 40), font, font_scale, font_color, line_type)

            if win_counts[i] == max_win_count:
                img = cv2.rectangle(img, (0, 0), (img.shape[1] - 1, img.shape[0] - 1), font_color, 10)

            frame.append(img)

        frame = np.concatenate(frame, axis=1)
        frames.append(frame)

        # print(f"WIN COUNTS")
        # for i, count in win_counts.items():
        #     print(f"\tgoal image {i}: {count}")

        with open(os.path.join(savedir, f"chat_gpt_answers_{timestep_no}.txt"), "w") as f:
            f.write(f"WIN COUNTS\n")
            for i, count in win_counts.items():
                f.write(f"\tgoal image {i}: {count}\n")

            f.write(f"\\nTotal query time: {total_query_time:.3f}s.\n\n")

            f.write("\n\n\n")
            

            for (i, j), info in infos.items():
                f.write("=" * 30 + f" goal images ({i}, {j}) " + "=" * 30 + "\n")
                f.write(f"timestep_dir: {timestep_dir}\n")
                f.write(f"language goal: \"{language_goal}\".\n")
                f.write(f"Query time: {info['query1 time']:.3f}s, {info['query2 time']:.3f}s.\n")
                answer = info["answer"]
                f.write(f"{answer}\n\n\n")

            


    
    if len(frames) > 1:
        print("saving video")
        episode = args.ep_dir.split("/")[-1]
        save_video(f"{savedir}/{episode}.mp4", np.array(frames))
    else:
        print("saving image")
        assert len(frames) == 1, f"len(frames): {len(frames)}"
        cv2.imwrite(f"{savedir}/timestep_{timestep_no}.png", np.array(frames[0])[..., ::-1])


def ask_chat_gpt5(rgb_image_obs, goal_images, language_goal, timestep_dir, n_attempts=10):

    base64_rgb_obs = encode_image(rgb_image_obs)

    prompt1 = f"We have a computer simulation of a robot tabletop manipulation environment. \
In this simulated environment, we are using a machine learning algorithm to control the robot so that \
it completes the following task: \"{language_goal}.\" The robot is partway through completing the task. We have an image of the environment taken from a third person camera. We will call this image the image observation. \
We also have a second image. This image is generated by a neural network, and shows the robot what it should do a few seconds into the future in order to come closer to completing the task \"{language_goal}\". This image is called the goal image. The robot will \
compare the image observation and the goal image, and figure out how it needs to move its gripper and what objects it needs to move or manipulate in order to make the environment match what is shown in the goal image. \
Since the goal image shows what the environment should look like a few seconds into the future, it will have some differences from the image observation. These differences will primarily be the location of the robot gripper and position of the robot arm, as well \
as the locations and positions of any objects that the robot is directly manipulating. However, sometimes the neural network will make errors when generating the goal image, which will cause the robot to become confused. These errors are when the goal image has harmful inconsistencies with the image observation. \
The following is a list of the most common types of harmful inconsistencies to look out for: \
1. Hallucinated objects. This is when there are objects that appear in the goal image but that do not appear in the image observation. Note, however, that the position of the robot gripper and arm in the image \
observation may be covering up objects that are behind them. So, if the position of the robot arm or gripper changes between the goal image and the image observation, there may be objects that were previously \
covered by the the robot arm or gripper in the image observation that are now visible in the goal image. These should not be considered hallucinations. Also note that not all hallucinations will be relevant/harmful. Hallucinated objects that are in an area of the image that the robot is not interacting with, or \
hallucinated objects that are not relevant to completing the task \"{language_goal}\" are unlikely to confuse the robot and so should not be considered relevant inconsistencies. \
2. Changes in object shape or color. This is when one or more objects in the goal image don't match their original shapes or colors in the image observation. Note, however, that slight changes in shape or color are okay and should not be considered inconsistencies \
(such as a shift to a slightly different shade of the same color or a slight shape elongations or distortions), but major changes in shape or color (such as an object becoming a completely different shape or becoming a completely different color) are not okay and should be considered inconsistencies. \
Note, however, that in both of these cases, these inconsistencies are only likely to be harmful if they involve objects that are directly related to completing the task \"{language_goal}\" or involve objects that the robot gripper is attempting to manipulate. Inconsistencies in the background of the image \
or in parts of the image that the robot gripper is not interacting with are most likely not harmful and should not be taken into consideration. Similarly, inconsistencies in the image not related to completing the task \"{language_goal}\", or not related to the objects that the robot gripper is attempting to manipulate \
should not be taken into consideration. Now, here is the image observation of the environment taken from the third person camera and here are two the goal images generated by the neural network. The first image is the image observation, and the second two images are the goal images. \
Please take special care not to confuse the ordering of the two goal images. \
Which of the two goal images is least likely to contain relevant/harmful inconsistencies with the image observation that would cause the robot to become confused? Please include \"First\" or \"Second\" in your answer (or \"Same\" if there is no significant difference between level of relevant/harmful inconsistencies). \
Please take special care not to confuse the ordering of the two goal images."


# The first image is the image observation, and the second image contains the two goal images, \
# stacked side-by-side. \
# Please take special care not to confuse the ordering of the two goal images. \
# Which of the two goal images is least likely to contain relevant/harmful inconsistencies with the image observation that would cause the robot to become confused? Is it the goal image on the left side, or the goal image on the right side? Please include \"Left\" or \"Right\" in your answer. \
# Please take special care not to confuse the ordering of the two goal images."

    prompt2 = f"Now please answer in just one word: \"First\", \"Second\", or \"Same\"."



    manager = Manager()

    # answers = {}
    win_counts = manager.dict()
    infos = manager.dict()
    function_inputs = []

    idx_pairs = list(combinations(range(goal_images.shape[0]), 2))

    # initialize the dictionaries
    for i in range(goal_images.shape[0]):
        win_counts[i] = 0
    for i, j in idx_pairs:
        infos[i, j] = manager.dict()

    for i, j in idx_pairs:
        goal_image1 = goal_images[i]
        goal_image2 = goal_images[j]

        
        base64_goal_image1 = encode_image(goal_image1)
        base64_goal_image2 = encode_image(goal_image2)
        # base64_goal_image = encode_image(np.concatenate([goal_image1, goal_image2], axis=1))



        function_inputs.append((win_counts, infos, i, j, base64_rgb_obs, base64_goal_image1, base64_goal_image2, prompt1, prompt2))
        # function_inputs.append((i, j, base64_rgb_obs, base64_goal_image1, base64_goal_image2, prompt1, prompt2))


    t0 = time.time()
    with Pool(len(function_inputs)) as pool: # We have one process per input because we are io bound, not cpu bound
        pool.starmap(compare_pair_of_goal_images, function_inputs)
    t1 = time.time()

    total_query_time = t1 - t0

    win_counts = dict(win_counts)
    infos = {key:dict(val) for key, val in infos.items()}

    for (i, j), info in infos.items():
        print("=" * 30 + f" goal images ({i}, {j}) " + "=" * 30 + "\n")
        print("timestep_dir:", timestep_dir)
        print(f"language goal: \"{language_goal}\".")
        print(f"Query time: {info['query1 time']:.3f}s, {info['query2 time']:.3f}s.")
        answer = info["answer"]
        print(f"{answer}\n\n\n")

    print(f"Total query time: {total_query_time:.3f}s.")
    print("WIN COUNTS")
    for i, count in win_counts.items():
        print(f"\tgoal image {i}: {count}")

    


    # print("\n\n" + "=" * 30 + f"Goal image {i} vs. {j}" + "=" * 30)
    # print("timestep_dir:", timestep_dir)
    # print(f"language goal: \"{language_goal}\".")
    # print("Chat GPT answer:\n", answer)
    # print(f"Query time: {info['query1 time']:.3f}s, {info['query1 time']:.3f}s.")


    return win_counts, infos, total_query_time


def compare_pair_of_goal_images(win_counts, infos, i, j, base64_rgb_obs, base64_goal_image1, base64_goal_image2, prompt1, prompt2):
    # global win_counts, infos
    # i, j, base64_rgb_obs, base64_goal_image1, base64_goal_image2, prompt1, prompt2 = function_input
    content = [
        {"type": "text","text": prompt1},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_rgb_obs}", "detail": "high"}},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_goal_image1}", "detail": "high"}},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_goal_image2}", "detail": "high"}},
        # {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_goal_image}", "detail": "high"}},

    ]

    messages = [{"role": "user", "content":content,}] 

    t0 = time.time()
    answer1 = query_chat_gpt(messages)   
    t1 = time.time()

    messages.append({"role":"assistant", "content":[{"type": "text","text": answer1},]})
    messages.append({"role":"user", "content":[{"type": "text","text": prompt2},]})

    t2 = time.time()
    answer2 = query_chat_gpt(messages)   
    t3 = time.time()

    answer2 = answer2.strip()
    answer2 = answer2.strip(".\n\"\',")

    if answer2.lower() == "first":
        win_counts[i] += 1 
    elif answer2.lower() == "second":
        win_counts[j] += 1
    elif answer2.lower() == "same": 
        win_counts[i] += 1  # assumption here is that they are more likely to be "same" if they are good, not if they are both bad
        win_counts[j] += 1
    else:
        raise ValueError(f"answer2.lower() not in [first, second, same]. answer2: \"{answer2.lower()}\".")

    answer = f"> {answer1} \n\n> {answer2}"

    

    # answers[(i, j)] = answer
    infos[(i, j)]["answer"] = answer
    infos[(i, j)]["query1 time"] = t1 - t0 
    infos[(i, j)]["query2 time"] = t3 - t2



def ask_chat_gpt3(rgb_image_obs, goal_images, language_goal, timestep_dir, n_attempts=10):
    # resize images to 512x512 
    # chat gpt 4 turbo 
    # detail level high 


    example_rgb_image_obs = cv2.imread(os.path.join("/home/kylehatch/Desktop/hidql/results/single_task/calvin/public_model/checkpoint_only/calvin/gcdiffusion/auggoaldiff/seed_0/20240227_194024/checkpoint_150000/chat_gpt_dummy_filter_4s_2024.04.10_15.47.24/ep31/unfiltered_goal_images/timestep_040", "rgb_obs.png"))
    example_rgb_image_obs = example_rgb_image_obs[..., ::-1]
    example_rgb_image_obs = cv2.resize(example_rgb_image_obs, (512, 512))



    base64_example_rgb_image_obs = encode_image(example_rgb_image_obs)


    # example_good_image = cv2.imread(os.path.join("/home/kylehatch/Desktop/hidql/results/single_task/calvin/public_model/checkpoint_only/calvin/gcdiffusion/auggoaldiff/seed_0/20240227_194024/checkpoint_150000/chat_gpt_dummy_filter_4s_2024.04.10_15.47.24/ep31/unfiltered_goal_images/timestep_040", "goal_image_001.png"))
    # example_good_image = example_good_image[..., ::-1]
    # example_good_image = cv2.resize(example_good_image, (512, 512))
    # base64_example_good_image = encode_image(example_good_image)

    # example_bad_image = cv2.imread(os.path.join("/home/kylehatch/Desktop/hidql/results/single_task/calvin/public_model/checkpoint_only/calvin/gcdiffusion/auggoaldiff/seed_0/20240227_194024/checkpoint_150000/chat_gpt_dummy_filter_4s_2024.04.10_15.47.24/ep31/unfiltered_goal_images/timestep_040", "goal_image_003.png"))
    # example_bad_image = example_bad_image[..., ::-1]
    # example_bad_image = cv2.resize(example_bad_image, (512, 512))
    # base64_example_bad_image = encode_image(example_bad_image)


    

    base64_rgb_obs = encode_image(rgb_image_obs)

    prompt1 = f"We have a computer simulation of a robot tabletop manipulation environment. \
In this simulated environment, we are using a machine learning algorithm to control the robot so that \
it completes the following task: \"{language_goal}.\" The robot is partway through completing the task. We have an image of the environment taken from a third person camera. We will call this image the image observation. \
We also have a second image. This image is generated by a neural network, and shows the robot what it should do a few seconds into the future in order to come closer to completing the task \"{language_goal}\". This image is called the goal image. The robot will \
compare the image observation and the goal image, and figure out how it needs to move its gripper and what objects it needs to move or manipulate in order to make the environment match what is shown in the goal image. \
Since the goal image shows what the environment should look like a few seconds into the future, it will have some differences from the image observation. These differences will primarily be the location of the robot gripper and position of the robot arm, as well \
as the locations and positions of any objects that the robot is directly manipulating. However, sometimes the neural network will make errors when generating the goal image, which will cause the robot to become confused. These errors are when the goal image has harmful inconsistencies with the image observation. \
The following is a list of the most common types of harmful inconsistencies to look out for: \
1. Hallucinated objects. This is when there are objects that appear in the goal image but that do not appear in the image observation. Note, however, that the position of the robot gripper and arm in the image \
observation may be covering up objects that are behind them. So, if the position of the robot arm or gripper changes between the goal image and the image observation, there may be objects that were previously \
covered by the the robot arm or gripper in the image observation that are now visible in the goal image. These should not be considered hallucinations. Also note that not all hallucinations will be relevant/harmful. Hallucinated objects that are in an area of the image that the robot is not interacting with, or \
hallucinated objects that are not relevant to completing the task \"{language_goal}\" are unlikely to confuse the robot and so should not be considered relevant inconsistencies. \
2. Changes in object shape or color. This is when one or more objects in the goal image don't match their original shapes or colors in the image observation. Note, however, that slight changes in shape or color are okay and should not be considered inconsistencies \
(such as a shift to a slightly different shade of the same color or a slight shape elongations or distortions), but major changes in shape or color (such as an object becoming a completely different shape or becoming a completely different color) are not okay and should be considered inconsistencies. \
Note, however, that in both of these cases, these inconsistencies are only likely to be harmful if they involve objects that are directly related to completing the task \"{language_goal}\" or involve objects that the robot gripper is attempting to manipulate. Inconsistencies in the background of the image \
or in parts of the image that the robot gripper is not interacting with are most likely not harmful and should not be taken into consideration. Similarly, inconsistencies in the image not related to completing the task \"{language_goal}\", or not related to the objects that the robot gripper is attempting to manipulate \
should not be taken into consideration. Now, here is the image observation of the environment taken from the third person camera and here are two the goal images generated by the neural network. The first image is the image observation, and the second two images are the goal images. \
Please take special care not to confuse the ordering of the two goal images. \
Which of the two goal images is least likely to contain relevant/harmful inconsistencies with the image observation that would cause the robot to become confused? Please include \"First\" or \"Second\" in your answer (or \"Same\" if there is no significant difference between level of relevant/harmful inconsistencies). \
Please take special care not to confuse the ordering of the two goal images."


# The first image is the image observation, and the second image contains the two goal images, \
# stacked side-by-side. \
# Please take special care not to confuse the ordering of the two goal images. \
# Which of the two goal images is least likely to contain relevant/harmful inconsistencies with the image observation that would cause the robot to become confused? Is it the goal image on the left side, or the goal image on the right side? Please include \"Left\" or \"Right\" in your answer. \
# Please take special care not to confuse the ordering of the two goal images."

    prompt2 = f"Now please answer in just one word: \"First\", \"Second\", or \"Same\"."

    answers = {}

    win_counts = defaultdict(int)


    idx_pairs = list(combinations(range(goal_images.shape[0]), 2))

    for i, j in idx_pairs:
        goal_image1 = goal_images[i]
        goal_image2 = goal_images[j]

        
        base64_goal_image1 = encode_image(goal_image1)
        base64_goal_image2 = encode_image(goal_image2)
        # base64_goal_image = encode_image(np.concatenate([goal_image1, goal_image2], axis=1))

        content = [
            {"type": "text","text": prompt1},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_rgb_obs}", "detail": "high"}},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_goal_image1}", "detail": "high"}},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_goal_image2}", "detail": "high"}},
            # {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_goal_image}", "detail": "high"}},

        ]

        messages = [{"role": "user", "content":content,}]

        t0 = time.time()
        answer1 = query_chat_gpt(messages)   
        t1 = time.time()

        messages.append({"role":"assistant", "content":[{"type": "text","text": answer1},]})
        messages.append({"role":"user", "content":[{"type": "text","text": prompt2},]})

        t2 = time.time()
        answer2 = query_chat_gpt(messages)   
        t3 = time.time()

        answer2 = answer2.strip()
        answer2 = answer2.strip(".\n\"\',")

        if answer2.lower() == "first":
            win_counts[i] += 1 
        elif answer2.lower() == "second":
            win_counts[j] += 1
        elif answer2.lower() == "same": 
            win_counts[i] += 1  # assumption here is that they are more likely to be "same" if they are good, not if they are both bad
            win_counts[j] += 1
        else:
            raise ValueError(f"answer2.lower() not in [first, second, same]. answer2: \"{answer2.lower()}\".")

        # answer = answer1 + "\n" + answer2

        answer = f"> {answer1} \n\n> {answer2}"

        print("\n\n" + "=" * 30 + f"Goal image {i} vs. {j}" + "=" * 30)
        print("timestep_dir:", timestep_dir)
        print(f"language goal: \"{language_goal}\".")
        print("Chat GPT answer:\n", answer)
        print(f"Query time: {t1 - t0:.3f}s, {t3 - t2:.3f}s.")

        answers[(i, j)] = answer

        

    return answers, win_counts


def ask_chat_gpt4(rgb_image_obs, goal_images, language_goal, timestep_dir, n_attempts=10):
    # resize images to 512x512 
    # chat gpt 4 turbo 
    # detail level high 


    example_rgb_image_obs = cv2.imread(os.path.join("/home/kylehatch/Desktop/hidql/results/single_task/calvin/public_model/checkpoint_only/calvin/gcdiffusion/auggoaldiff/seed_0/20240227_194024/checkpoint_150000/chat_gpt_dummy_filter_4s_2024.04.10_15.47.24/ep31/unfiltered_goal_images/timestep_040", "rgb_obs.png"))
    example_rgb_image_obs = example_rgb_image_obs[..., ::-1]
    example_rgb_image_obs = cv2.resize(example_rgb_image_obs, (512, 512))

    white_header = np.ones((50, 512, 3), dtype=np.uint8) * 255
    example_rgb_image_obs = np.concatenate([white_header, example_rgb_image_obs], axis=0)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (0, 0, 0)  # White color in BGR
    line_type = 4  # Line thickness

    example_rgb_image_obs = cv2.putText(example_rgb_image_obs, f'OBSERVATION IMAGE', (30, 30), font, font_scale, font_color, line_type)
    # cv2.imwrite("example_rgb_image_obs.png", example_rgb_image_obs[..., ::-1])

    base64_example_rgb_image_obs = encode_image(example_rgb_image_obs)


    # example_good_image = cv2.imread(os.path.join("/home/kylehatch/Desktop/hidql/results/single_task/calvin/public_model/checkpoint_only/calvin/gcdiffusion/auggoaldiff/seed_0/20240227_194024/checkpoint_150000/chat_gpt_dummy_filter_4s_2024.04.10_15.47.24/ep31/unfiltered_goal_images/timestep_040", "goal_image_001.png"))
    # example_good_image = example_good_image[..., ::-1]
    # example_good_image = cv2.resize(example_good_image, (512, 512))
    # base64_example_good_image = encode_image(example_good_image)

    # example_bad_image = cv2.imread(os.path.join("/home/kylehatch/Desktop/hidql/results/single_task/calvin/public_model/checkpoint_only/calvin/gcdiffusion/auggoaldiff/seed_0/20240227_194024/checkpoint_150000/chat_gpt_dummy_filter_4s_2024.04.10_15.47.24/ep31/unfiltered_goal_images/timestep_040", "goal_image_003.png"))
    # example_bad_image = example_bad_image[..., ::-1]
    # example_bad_image = cv2.resize(example_bad_image, (512, 512))
    # base64_example_bad_image = encode_image(example_bad_image)


    

    base64_rgb_obs = encode_image(rgb_image_obs)

    prompt1 = f"We have a computer simulation of a robot tabletop manipulation environment. \
In this simulated environment, we are using a machine learning algorithm to control the robot so that \
it completes the following task: \"{language_goal}.\" The robot is partway through completing the task. We have an image of the environment taken from a third person camera. We will call this image the image observation. \
We also have a second image. This image is generated by a neural network, and shows the robot what it should do a few seconds into the future in order to come closer to completing the task \"{language_goal}\". This image is called the goal image. The robot will \
compare the image observation and the goal image, and figure out how it needs to move its gripper and what objects it needs to move or manipulate in order to make the environment match what is shown in the goal image. \
Since the goal image shows what the environment should look like a few seconds into the future, it will have some differences from the image observation. These differences will primarily be the location of the robot gripper and position of the robot arm, as well \
as the locations and positions of any objects that the robot is directly manipulating. However, sometimes the neural network will make errors when generating the goal image, which will cause the robot to become confused. These errors are when the goal image has harmful inconsistencies with the image observation. \
The following is a list of the most common types of harmful inconsistencies to look out for: \
1. Hallucinated objects. This is when there are objects that appear in the goal image but that do not appear in the image observation. Note, however, that the position of the robot gripper and arm in the image \
observation may be covering up objects that are behind them. So, if the position of the robot arm or gripper changes between the goal image and the image observation, there may be objects that were previously \
covered by the the robot arm or gripper in the image observation that are now visible in the goal image. These should not be considered hallucinations. Also note that not all hallucinations will be relevant/harmful. Hallucinated objects that are in an area of the image that the robot is not interacting with, or \
hallucinated objects that are not relevant to completing the task \"{language_goal}\" are unlikely to confuse the robot and so should not be considered relevant inconsistencies. \
2. Changes in object shape or color. This is when one or more objects in the goal image don't match their original shapes or colors in the image observation. Note, however, that slight changes in shape or color are okay and should not be considered inconsistencies \
(such as a shift to a slightly different shade of the same color or a slight shape elongations or distortions), but major changes in shape or color (such as an object becoming a completely different shape or becoming a completely different color) are not okay and should be considered inconsistencies. \
Note, however, that in both of these cases, these inconsistencies are only likely to be harmful if they involve objects that are directly related to completing the task \"{language_goal}\" or involve objects that the robot gripper is attempting to manipulate. Inconsistencies in the background of the image \
or in parts of the image that the robot gripper is not interacting with are most likely not harmful and should not be taken into consideration. Similarly, inconsistencies in the image not related to completing the task \"{language_goal}\", or not related to the objects that the robot gripper is attempting to manipulate \
should not be taken into consideration. Now, here is the image observation of the environment taken from the third person camera and here are two the goal images generated by the neural network. The first image is the image observation, and it has been marked with a header that says \
\"OBSERVATION IMAGE\". The second two images are the goal images, and have been marked with headers that say \"GOAL IMAGE 1\" and \"GOAL IMAGE 2\", respectively. \
Which of the two goal images is least likely to contain relevant/harmful inconsistencies with the image observation that would cause the robot to become confused? Please include \"GOAL IMAGE 1\" or \"GOAL IMAGE 2\" in your answer." 

    answers = {}


    idx_pairs = list(combinations(range(goal_images.shape[0]), 2))

    for i, j in idx_pairs:
        goal_image1 = goal_images[i]
        goal_image2 = goal_images[j]

        goal_image1 = np.concatenate([white_header, goal_image1], axis=0)
        goal_image1 = cv2.putText(goal_image1, f'GOAL IMAGE 1', (30, 30), font, font_scale, font_color, line_type)
        # cv2.imwrite("goal_image1.png", goal_image1[..., ::-1])
        

        goal_image2 = np.concatenate([white_header, goal_image2], axis=0)
        goal_image2 = cv2.putText(goal_image2, f'GOAL IMAGE 2', (30, 30), font, font_scale, font_color, line_type)
        # cv2.imwrite("goal_image2.png", goal_image2[..., ::-1])

        base64_goal_image1 = encode_image(goal_image1)
        base64_goal_image2 = encode_image(goal_image2)
        # base64_goal_image = encode_image(np.concatenate([goal_image1, goal_image2], axis=1))

        content = [
            {"type": "text","text": prompt1},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_rgb_obs}", "detail": "high"}},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_goal_image1}", "detail": "high"}},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_goal_image2}", "detail": "high"}},
            # {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_goal_image}", "detail": "high"}},

        ]

        t0 = time.time()
        answer = query_chat_gpt(content)
        t1 = time.time()

        print(f"\nGoal image {i} vs. {j}")
        print("timestep_dir:", timestep_dir)
        print(f"language goal: \"{language_goal}\".")
        print("Chat GPT answer:", answer)
        print(f"Query time: {t1 - t0:.3f}s.")

        answers[(i, j)] = answer

        


    return answers

def ask_chat_gpt2(rgb_image_obs, goal_images, language_goal, timestep_dir, n_attempts=10):
    # resize images to 512x512 
    # chat gpt 4 turbo 
    # detail level high 


    example_rgb_image_obs = cv2.imread(os.path.join("/home/kylehatch/Desktop/hidql/results/single_task/calvin/public_model/checkpoint_only/calvin/gcdiffusion/auggoaldiff/seed_0/20240227_194024/checkpoint_150000/chat_gpt_dummy_filter_4s_2024.04.10_15.47.24/ep31/unfiltered_goal_images/timestep_040", "rgb_obs.png"))
    example_rgb_image_obs = example_rgb_image_obs[..., ::-1]
    example_rgb_image_obs = cv2.resize(example_rgb_image_obs, (512, 512))
    base64_example_rgb_image_obs = encode_image(example_rgb_image_obs)


    example_good_image = cv2.imread(os.path.join("/home/kylehatch/Desktop/hidql/results/single_task/calvin/public_model/checkpoint_only/calvin/gcdiffusion/auggoaldiff/seed_0/20240227_194024/checkpoint_150000/chat_gpt_dummy_filter_4s_2024.04.10_15.47.24/ep31/unfiltered_goal_images/timestep_040", "goal_image_001.png"))
    example_good_image = example_good_image[..., ::-1]
    example_good_image = cv2.resize(example_good_image, (512, 512))
    base64_example_good_image = encode_image(example_good_image)

    example_bad_image = cv2.imread(os.path.join("/home/kylehatch/Desktop/hidql/results/single_task/calvin/public_model/checkpoint_only/calvin/gcdiffusion/auggoaldiff/seed_0/20240227_194024/checkpoint_150000/chat_gpt_dummy_filter_4s_2024.04.10_15.47.24/ep31/unfiltered_goal_images/timestep_040", "goal_image_003.png"))
    example_bad_image = example_bad_image[..., ::-1]
    example_bad_image = cv2.resize(example_bad_image, (512, 512))
    base64_example_bad_image = encode_image(example_bad_image)


    

    base64_rgb_obs = encode_image(rgb_image_obs)

    prompt1 = f"We have a computer simulation of a robot tabletop manipulation environment. \
In this simulated environment, we are using a machine learning algorithm to control the robot so that \
it completes the following task: \"{language_goal}.\" The robot is partway through completing the task. We have an image of the environment taken from a third person camera. We will call this image the image observation. \
We also have a second image. This image is generated by a neural network, and shows the robot what it should do a few seconds into the future in order to come closer to completing the task \"{language_goal}\". This image is called the goal image. The robot will \
compare the image observation and the goal image, and figure out how it needs to move its gripper and what objects it needs to move or manipulate in order to make the environment match what is shown in the goal image. \
Since the goal image shows what the environment should look like a few seconds into the future, it will have some differences from the image observation. These differences will primarily be the location of the robot gripper and position of the robot arm, as well \
as the locations and positions of any objects that the robot is directly manipulating. However, sometimes the neural network will make errors when generating the goal image, which will cause the robot to become confused. These errors are when the goal image has harmful inconsistencies with the image observation. \
The following is a list of the most common types of harmful inconsistencies to look out for: \
1. Hallucinated objects. This is when there are objects that appear in the goal image but that do not appear in the image observation. Note, however, that the position of the robot gripper and arm in the image \
observation may be covering up objects that are behind them. So, if the position of the robot arm or gripper changes between the goal image and the image observation, there may be objects that were previously \
covered by the the robot arm or gripper in the image observation that are now visible in the goal image. These should not be considered hallucinations. Also note that not all hallucinations will be relevant/harmful. Hallucinated objects that are in an area of the image that the robot is not interacting with, or \
hallucinated objects that are not relevant to completing the task \"{language_goal}\" are unlikely to confuse the robot and so should not be considered relevant inconsistencies. \
2. Changes in object shape or color. This is when one or more objects in the goal image don't match their original shapes or colors in the image observation. Note, however, that slight changes in shape or color are okay and should not be considered inconsistencies \
(such as a shift to a slightly different shade of the same color or a slight shape elongations or distortions), but major changes in shape or color (such as an object becoming a completely different shape or becoming a completely different color) are not okay and should be considered inconsistencies. \
Note, however, that in both of these cases, these inconsistencies are only likely to be harmful if they involve objects that are directly related to completing the task \"{language_goal}\" or involve objects that the robot gripper is attempting to manipulate. Inconsistencies in the background of the image \
or in parts of the image that the robot gripper is not interacting with are most likely not harmful and should not be taken into consideration. Similarly, inconsistencies in the image not related to completing the task \"{language_goal}\", or not related to the objects that the robot gripper is attempting to manipulate \
should not be taken into consideration. \
Here is an example of an image observation of the environment taken from the third person camera and a good goal image generated by the neural network. The first image is the image observation, and the second image is the goal image. While this good goal image may contain some slight inconsistencies or irrelevant inconsistencies with the image observation, it does not contain any \
inconsistencies that are harmful to the robot completing its task."
    
    prompt2 = f"In contrast, here is an example of an image observation and a bad goal image generated by the neural network. The first image is the image observation, and the second image is the goal image. This goal image contains some inconsistencies with the image observation \
that are relevant to the task, and could cause the robot to become confused and hinder it from completing its task."


    prompt3 = f"Now, here is the image observation of the environment taken from the third person camera and here is the goal image generated by the neural network. The first image is the image observation, and the second image is the goal image. \
Are there any relevant/harmful inconsistencies between the image observation and the goal image that would cause the robot to become confused? Please include \"Yes\" or \"No\" in your answer."


    answers = []

    
    for i, goal_image in enumerate(goal_images):
        
        base64_goal_image = encode_image(goal_image)
        content = [
            # {"type": "text","text": prompt1},
            # {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_rgb_obs}", "detail": "high"}},
            # {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_goal_image}", "detail": "high"}},
            {"type": "text","text": prompt1},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_example_rgb_image_obs}", "detail": "high"}},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_example_good_image}", "detail": "high"}},

            {"type": "text","text": prompt2},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_example_rgb_image_obs}", "detail": "high"}},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_example_bad_image}", "detail": "high"}},

            {"type": "text","text": prompt3},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_rgb_obs}", "detail": "high"}},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_goal_image}", "detail": "high"}},
        ]

        t0 = time.time()
        answer = query_chat_gpt(content)
        t1 = time.time()

        print(f"\nGoal image {i}")
        print("timestep_dir:", timestep_dir)
        print(f"language goal: \"{language_goal}\".")
        print("Chat GPT answer:", answer)
        print(f"Query time: {t1 - t0:.3f}s.")

        answers.append(answer)

        


    return answers


def ask_chat_gpt(rgb_image_obs, goal_images, language_goal, timestep_dir, n_attempts=10):
    base64_rgb_obs = encode_image(rgb_image_obs)

    prompt1 = f"We have a computer simulation of a robot tabletop manipulation environment. \
In this simulated environment, we are using a machine learning algorithm to control the robot so that \
it completes the following task: \"{language_goal}.\" The machine learning algorithm works as follows: \
at each time-step, an image of the environment is taken from a third person camera. This image is \
given to a generative neural network that generates a goal image of what the robot should accomplish \
twenty time-steps into the future in order to come closer to completing the task, \"{language_goal}.\" \
A low level controller then tries to control the robot to reach the generated goal image. After twenty time-steps, \
a new image is taken from the camera, and the generative neural network produces a new goal image for the low level \
controller to reach. This is repeated until the task, \"{language_goal}\" is completed. The robot is partway through \
completing the task. Here is the image from the third person camera showing the current state of the simulated environment."
    
    prompt2 = f"Here is a candidate goal image generated by the neural network showing possible goals of what the robot \
should do over the next twenty time-steps in order to get closer to completing the task, \"{language_goal}.\" However, generated goal images may \
be of high or low quality, meaning that some generated goal images are good at guiding the low level policy towards completing the task, while some \
generated goal images will have errors that will cause the low level controller not to successfully complete the task. For example, sometimes the neural network may generate goal images that have \
hallucinated objects that do not actually exist in the environment. Sometimes the neural network may generate images that show it \
completing the task incorrectly, or that show it completing a different task. Or sometimes the neural network may generate goal \
images that are sub-optimal for other reasons. Given this information, is the following generated goal image likely to lead the low level controller closer to completing the task, \
\"{language_goal}\"? Please include \"Yes\" or \"No\" in your answer."
    
    prompt3 = f"Here are two candidate goal images generated by the neural network showing possible goals of what the robot \
should do over the next twenty time-steps in order to get closer to completing the task, \"{language_goal}.\" However, generated goal images may \
be of high or low quality, meaning that some generated goal images are good at guiding the low level policy towards completing the task, while some \
generated goal images will have errors that will cause the low level controller not to successfully complete the task. For example, sometimes the neural network may generate goal images that have \
hallucinated objects that do not actually exist in the environment. Sometimes the neural network may generate images that show it \
completing the task incorrectly, or that show it completing a different task. Or sometimes the neural network may generate goal \
images that are sub-optimal for other reasons. Given this information, which goal image is more likely to lead the low level controller closer to successfuly completing the task, \
\"{language_goal}\"? Please include \"Image 1\" or \"Image 2\" in your answer."
    

    prompt4 = f"Here are two candidate goal images generated by the neural network showing possible goals of what the robot \
should do over the next twenty time-steps in order to get closer to completing the task, \"{language_goal}\" (The images are stacked together, so that one of the images is on the left, and one of the images \
is on the right). However, generated goal images may \
be of high or low quality, meaning that some generated goal images are good at guiding the low level policy towards completing the task, while some \
generated goal images will have errors that will cause the low level controller not to successfully complete the task. For example, sometimes the neural network may generate goal images that have \
hallucinated objects that do not actually exist in the environment. Sometimes the neural network may generate images that show it \
completing the task incorrectly, or that show it completing a different task. Or sometimes the neural network may generate goal \
images that are sub-optimal for other reasons. Given this information, which goal image is more likely to lead the low level controller closer to successfuly completing the task, \
\"{language_goal}\"? Please include \"Left\" or \"Right\" in your answer, and please make sure the first word of your answer is either \"Left\" or \"Right\"."
    for attempt_idx in range(n_attempts):
        try:

            yes_goal_images = {}

            
            for i, goal_image in enumerate(goal_images):
                base64_goal_image = encode_image(goal_image)
                content = [
                    {"type": "text","text": prompt1},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_rgb_obs}"}},
                    {"type": "text","text": prompt2},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_goal_image}"}},
                ]

                t0 = time.time()
                answer = query_chat_gpt(content)
                t1 = time.time()

                print(f"\nGoal image {i}")
                print("timestep_dir:", timestep_dir)
                print(f"language goal: \"{language_goal}\".")
                print("Chat GPT answer:", answer)
                print(f"Query time: {t1 - t0:.3f}s.")

                

                # if "yes" in answer.lower():
                #     assert "no" not in answer.lower(), f"answer: \"{answer}\"."
                #     yes_goal_images[i] = goal_image
                # else:
                #     assert "no" in answer.lower(), f"answer: \"{answer}\"."
                if answer.lower()[:3] == "yes":
                    yes_goal_images[i] = goal_image

                else:
                    # assert "no" in answer.lower(), f"answer: \"{answer}\"."
                    if "no" not in answer.lower():
                        print(f"\"no\" not found in answer to yes/no query after {attempt_idx + 1}/{n_attempts} attempts: \"{answer}\".")
                        raise ValueError(f"\"no\" not found in answer to yes/no query after {attempt_idx + 1}/{n_attempts} attempts: \"{answer}\".")

            break 
        except:
            return 0, f"\"no\" not found in answer to yes/no query after {attempt_idx + 1}/{n_attempts} attempts: \"{answer}\".", []


    # ordered_keys = sorted(list(yes_goal_images.keys()))
    # best_goal_img_idx = ordered_keys[0]
    # for curr_goal_img_idx in ordered_keys[1:]:
    #     if eval(best_goal_img_idx) < eval(curr_goal_img_idx):
    #         best_goal_img_idx = curr_goal_img_idx
            
    print("list(yes_goal_images.keys()):", list(yes_goal_images.keys()))
            
    if len(list(yes_goal_images.keys())) > 1:

        for attempt_idx in range(n_attempts):
            try:

                ordered_keys = sorted(list(yes_goal_images.keys()))
                best_goal_img_idx = ordered_keys[0]
                for curr_goal_img_idx in ordered_keys[1:]:
                    # goal_image1 = yes_goal_images[best_goal_img_idx]
                    # base64_goal_image1 = encode_image(goal_image1)


                    # goal_image2 = yes_goal_images[curr_goal_img_idx]
                    # base64_goal_image2 = encode_image(goal_image2)

                    # content = [
                    #     {"type": "text","text": prompt1},
                    #     {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_rgb_obs}"}},
                    #     {"type": "text","text": prompt3},
                    #     {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_goal_image1}"}},
                    #     {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_goal_image2}"}},
                    # ]

                    goal_image1 = yes_goal_images[best_goal_img_idx]
                    goal_image2 = yes_goal_images[curr_goal_img_idx]


                    base64_goal_images = encode_image(np.concatenate([goal_image1, goal_image2], axis=1))

                    content = [
                        {"type": "text","text": prompt1},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_rgb_obs}"}},
                        {"type": "text","text": prompt4},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_goal_images}"}},
                    ]

                    t0 = time.time()
                    answer = query_chat_gpt(content)
                    t1 = time.time()

                    print(f"\nGoal image {best_goal_img_idx} vs. goal image {curr_goal_img_idx}")
                    print("timestep_dir:", timestep_dir)
                    print(f"language goal: \"{language_goal}\".")
                    print("Chat GPT answer:", answer)
                    print(f"Query time: {t1 - t0:.3f}s.")

                    if answer.lower()[:5] == "right":
                        best_goal_img_idx = curr_goal_img_idx
                        
                    else:
                        # assert "left" in answer.lower(), f"answer: \"{answer}\"."
                        if "left" not in answer.lower():
                            print(f"\"left\" not found in answer to pairwise comparison after {attempt_idx + 1}/{n_attempts} attempts: \"{answer}\".")
                            raise ValueError(f"\"left\" not found in answer to pairwise comparison after {attempt_idx + 1}/{n_attempts} attempts: \"{answer}\".")



                break  
            except:
                return 0, f"\"left\" not found in answer to pairwise comparison after {attempt_idx + 1}/{n_attempts} attempts: \"{answer}\".", []

    elif len(list(yes_goal_images.keys())) == 1:
        best_goal_img_idx = list(yes_goal_images.keys())[0]
        answer = "single yes"
    elif len(list(yes_goal_images.keys())) == 0:
        best_goal_img_idx = 0
        answer = "zero yes"



    return best_goal_img_idx, answer, list(yes_goal_images.keys())



    

    # prompt = f"What is the largest mountain range in North America?"

    # answer = query_chat_gpt(prompt)

    


def query_chat_gpt(messages, dryrun=False):
    # api_key = "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"
    # api_key = "sk-proj-VtrcFJQBB1W4MEGrdibfT3BlbkFJoSpSwukCtDlxcADLAGLc"
    api_key = "sk-proj-mr8ZIOp3MVVEVbCNYgFFT3BlbkFJCdSIulmzOATndgxhFDuU" # new one from Blake
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    payload = {
        # "model": "gpt-4-vision-preview",
        "model": "gpt-4-turbo",
        # "messages": [
        #     {
        #         "role": "user",
        #         "content":content,
        #     }
        # ],
        "messages":messages,
        # "max_tokens": 300,
        "max_tokens": 1200,
    }

    if dryrun:
        return "Did not run due to dryrun=True"

    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers=headers,
        json=payload,
    )
    answer = None
    try:
        answer = response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"Exception parsing response.json(): {response.json()}, error: {e}")
        answer = "Error"
    
    return answer



def encode_image(
    # image_filepath: str,
    image,
    image_format: str = "PNG",
    max_side_length: int = 512,
    debug: bool = False,
):
    image = Image.fromarray(image)
    # Calculate the new size, keeping aspect ratio.
    ratio = min(max_side_length / image.width, max_side_length / image.height)
    new_size = (int(image.width * ratio), int(image.height * ratio))
    resized_image = image.resize(new_size, Image.LANCZOS)

    # if debug:
        # debug_dir = Path(
        #     "/home/blakewulfe/data/datasets/r2d2/visualization/attempt_2/data/debug"
        # )
        # debug_filepath = debug_dir / "_".join(image_filepath.parts[-6:])
        # resized_image.save(debug_filepath, format=image_format)

    buffered = io.BytesIO()
    resized_image.save(buffered, format=image_format)
    image_bytes = buffered.getvalue()


    image_base64 = base64.b64encode(image_bytes)
    image_base64_str = image_base64.decode("utf-8")
    return image_base64_str
    
def save_video(output_video_file, frames):
     # Extract frame dimensions
    height, width, _ = frames.shape[1:]

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can also use other codecs such as 'XVID'
    fps = 30  # Adjust the frame rate as needed

    os.makedirs(os.path.dirname(output_video_file), exist_ok=True)
    video_writer = cv2.VideoWriter(output_video_file, fourcc, fps, (width, height))

    # Write each frame to the video file
    for frame in frames:
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video_writer.write(bgr_frame)

    # Release the video writer object
    video_writer.release()

def parse_arguments():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate a trained model on multistep sequences with language goals.")
    # parser.add_argument("--logdir", type=str, help="Path to the dataset root directory.")
    parser.add_argument("--ep_dir", type=str, help="Path to the dataset root directory.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    visualize_outputs3(args)

"""

python3 -u chat_gpt_goal_rankings.py \
--logdir /home/kylehatch/Desktop/hidql/calvin-sim/results/el_trasho/single_task/calvin/public_model/checkpoint_only/calvin/gcdiffusion/auggoaldiff/seed_0/20240227_194024/checkpoint_150000/2024.04.10_14.59.07


python3 -u chat_gpt_goal_rankings.py \
--timestep_dir /home/kylehatch/Desktop/hidql/calvin-sim/results/single_task/calvin/public_model/checkpoint_only/calvin/gcdiffusion/auggoaldiff/seed_0/20240227_194024/checkpoint_150000/chat_gpt_dummy_filter_4s_2024.04.10_15.47.24/ep3/unfiltered_goal_images/timestep_140




python3 -u chat_gpt_goal_rankings.py \
--ep_dir /home/kylehatch/Desktop/hidql/results/single_task/calvin/public_model/checkpoint_only/calvin/gcdiffusion/auggoaldiff/seed_0/20240227_194024/checkpoint_150000/chat_gpt_dummy_filter_4s_2024.04.10_15.47.24/ep31/unfiltered_goal_images/timestep_000



python3 -u chat_gpt_goal_rankings.py \
--ep_dir /home/kylehatch/Desktop/hidql/results/single_task/calvin/public_model/checkpoint_only/calvin/gcdiffusion/auggoaldiff/seed_0/20240227_194024/checkpoint_150000/chat_gpt_dummy_filter_4s_2024.04.10_15.47.24/ep0/unfiltered_goal_images/timestep_020

"""

"""
Integrate into the eval loop, and evaluate failure cases to see what other things to add to the prompt 

Could include prompts about wanting the steps shown to be incremental? 

Another important aspect is that it should show step by step progression
- if an object moves, it should first show the gripper coming close to that object or grasping or contacting that object.
- if it just shows it appearing in the drawer, could be hard for the low level controller to interpret how to get it there



"""




#     prompt1 = f"We have a computer simulation of a robot tabletop manipulation environment. \
# In this simulated environment, we are using a machine learning algorithm to control the robot so that \
# it completes the following task: \"{language_goal}.\" The robot is partway through completing the task. We have an image of the environment taken from a third person camera. We will call this image the image observation. \
# We also have a second image. This image is generated by a neural network, and shows the robot what it should do a few seconds into the future in order to come closer to completing the task \"{language_goal}\". This image is called the goal image. The robot will \
# compare the image observation and the goal image, and figure out how it needs to move its gripper and what objects it needs to move or manipulate in order to make the environment match what is shown in the goal image. \
# Since the goal image shows what the environment should look like a few seconds into the future, it will have some differences from the image observation. These differences will primarily be the location of the robot gripper and position of the robot arm, as well \
# as the locations and positions of any objects that the robot is directly manipulating. However, sometimes the neural network will make errors when generating the goal image, which will cause the robot to become confused. These errors are when the goal image has temporal inconsistencies with the image observation. \
# The following is a list of the most common types of inconsistencies to look out for: \
# 1. Hallucinated objects. This is when 

# For example, one type of inconsistency is when the goal image contains hallucinated objects. These are objects that do not appear in the image observation, but do appear in the goal image. Another type of inconsistency is when the neural network changes the shape or color of one or \
# more objects in the goal image, so that these objects don't match their original shapes or colors in the image observation. 


# Additionally, if an object changes position without being manipulated by the gripper, or if an object that is not relevant to completing the task \"{language_goal}\" changes position, \
# this is likely an inconsistency. However, note that since the robot is moving the position of its arm and gripper, changes in the position of the robot arm or gripper, or any objects that are being directly manipulated by the robot arm or gripper, should not be considered inconsistencies. \
# Also note that the position of the robot gripper and arm in the image observation may be covering up objects that are behind them. So, if the position of the robot arm or gripper changes between the goal image and the image observation, there may be objects that were previously \
# covered by the the robot arm or gripper in the image observation that are now visible in the goal image. These should not be considered inconsistencies. \
# Now, here is the image observation of the environment taken from the third person camera and here is the goal image generated by the neural network. The first image is the image observation, and the second image is the goal image. \
# Are there any inconsistencies between the image observation and the goal image that would cause the robot to become confused? Please include \"Yes\" or \"No\" in your answer."

# # Now, here is the image observation of the environment taken from the third person camera and the goal image generated by the neural network. The image observation and the goal image are stacked together, so that the image observation is on the left side and the goal image is on the right side. \
# # Are there any inconsistencies between the image observation and the goal image that would cause the robot to become confused? Please include \"Yes\" or \"No\" in your answer."




#     prompt1 = f"We have a computer simulation of a robot tabletop manipulation environment. \
# In this simulated environment, we are using a machine learning algorithm to control the robot so that \
# it completes the following task: \"{language_goal}.\" The robot is partway through completing the task. We have an image of the environment taken from a third person camera. We will call this image the image observation. \
# We also have a second image. This image is generated by a neural network, and shows the robot what it should do a few seconds into the future in order to come closer to completing the task \"{language_goal}\". This image is called the goal image. The robot will \
# compare the image observation and the goal image, and figure out how it needs to move its gripper and what objects it needs to move or manipulate in order to make the environment match what is shown in the goal image. \
# Since the goal image shows what the environment should look like a few seconds into the future, it will have some differences from the image observation. These differences will primarily be the location of the robot gripper and position of the robot arm, as well \
# as the locations and positions of any objects that the robot is directly manipulating. However, sometimes the neural network will make errors when generating the goal image, which will cause the robot to become confused. These errors are when the goal image has temporal inconsistencies with the image observation. \
# The following is a list of the most common types of inconsistencies to look out for: \
# 1. Hallucinated objects. This is when there are objects that appear in the goal image but that do not appear in the image observation. Note, however, that the position of the robot gripper and arm in the image \
# observation may be covering up objects that are behind them. So, if the position of the robot arm or gripper changes between the goal image and the image observation, there may be objects that were previously \
# covered by the the robot arm or gripper in the image observation that are now visible in the goal image. These should not be considered hallucinations. \
# 2. Changes in object shape or color. This is when one or more objects in the goal image don't match their original shapes or colors in the image observation. Note, however, that slight changes in shape or color are okay and should not be considered inconsistencies \
# (such as a shift to a slightly different shade of the same color or a slightly different proportions of the same shape), but major changes in shape or color (such as an object becoming a completely different shape or becoming a completely different color) are not okay and should be considered inconsistencies. \
# 3. Jumps in object positions. This is when an object changes position without being manipulated by the gripper, or if an object that is not relevant to completing the task \"{language_goal}\" changes position. However, note that since the robot is moving the position of its arm \
# and gripper, changes in the position of the robot arm or gripper, or any objects that are being directly manipulated by the robot arm or gripper, should not be considered inconsistencies. Additionally, very slight changes in object positions should not be considered inconsistencies. \
# Now, here is the image observation of the environment taken from the third person camera and here is the goal image generated by the neural network. The first image is the image observation, and the second image is the goal image. \
# Are there any inconsistencies between the image observation and the goal image that would cause the robot to become confused? Please include \"Yes\" or \"No\" in your answer."

# # Now, here is the image observation of the environment taken from the third person camera and the goal image generated by the neural network. The image observation and the goal image are stacked together, so that the image observation is on the left side and the goal image is on the right side. \
# # Are there any inconsistencies between the image observation and the goal image that would cause the robot to become confused? Please include \"Yes\" or \"No\" in your answer."


# prompt1 = f"We have a computer simulation of a robot tabletop manipulation environment. \
# In this simulated environment, we are using a machine learning algorithm to control the robot so that \
# it completes the following task: \"{language_goal}.\" The robot is partway through completing the task. We have an image of the environment taken from a third person camera. We will call this image the image observation. \
# We also have a second image. This image is generated by a neural network, and shows the robot what it should do a few seconds into the future in order to come closer to completing the task \"{language_goal}\". This image is called the goal image. The robot will \
# compare the image observation and the goal image, and figure out how it needs to move its gripper and what objects it needs to move or manipulate in order to make the environment match what is shown in the goal image. \
# Since the goal image shows what the environment should look like a few seconds into the future, it will have some differences from the image observation. These differences will primarily be the location of the robot gripper and position of the robot arm, as well \
# as the locations and positions of any objects that the robot is directly manipulating. However, sometimes the neural network will make errors when generating the goal image, which will cause the robot to become confused. These errors are when the goal image has temporal inconsistencies with the image observation. \
# The following is a list of the most common types of inconsistencies to look out for: \
# 1. Hallucinated objects. This is when there are objects that appear in the goal image but that do not appear in the image observation. Note, however, that the position of the robot gripper and arm in the image \
# observation may be covering up objects that are behind them. So, if the position of the robot arm or gripper changes between the goal image and the image observation, there may be objects that were previously \
# covered by the the robot arm or gripper in the image observation that are now visible in the goal image. These should not be considered hallucinations. \
# 2. Changes in object shape or color. This is when one or more objects in the goal image don't match their original shapes or colors in the image observation. Note, however, that slight changes in shape or color are okay and should not be considered inconsistencies \
# (such as a shift to a slightly different shade of the same color or a slight shape elongations or distortions), but major changes in shape or color (such as an object becoming a completely different shape or becoming a completely different color) are not okay and should be considered inconsistencies. \
# 3. Jumps in object positions. This is when an object changes position without being manipulated by the gripper, or if an object that is not relevant to completing the task \"{language_goal}\" changes position. However, note that since the robot is moving the position of its arm \
# and gripper, changes in the position of the robot arm or gripper, or any objects that are being directly manipulated by the robot arm or gripper, should not be considered inconsistencies. Additionally, very slight changes in object positions should not be considered inconsistencies. \
# Note, however, that in all of these cases, these inconsistencies are only likely to be harmful if they involve objects that are directly related to completing the task \"{language_goal}\" or involve objects that the robot gripper is attempting to manipulate. Inconsistencies in the background of the image \
# or in parts of the image that the robot gripper is not interacting with are most likely not harmful and should not be taken into consideration. Similarly, inconsistencies in the image not related to completing the task \"{language_goal}\", or not related to the objects that the robot gripper is attempting to manipulate \
# should not be taken into consideration. \
# Now, here is the image observation of the environment taken from the third person camera and here is the goal image generated by the neural network. The first image is the image observation, and the second image is the goal image. \
# Are there any relevant inconsistencies between the image observation and the goal image that would cause the robot to become confused? Please include \"Yes\" or \"No\" in your answer."





#     example_rgb_image_obs = cv2.imread(os.path.join("/home/kylehatch/Desktop/hidql/results/single_task/calvin/public_model/checkpoint_only/calvin/gcdiffusion/auggoaldiff/seed_0/20240227_194024/checkpoint_150000/chat_gpt_dummy_filter_4s_2024.04.10_15.47.24/ep31/unfiltered_goal_images/timestep_000", "rgb_obs.png"))
#     example_rgb_image_obs = example_rgb_image_obs[..., ::-1]
#     example_rgb_image_obs = cv2.resize(example_rgb_image_obs, (512, 512))
#     base64_example_rgb_image_obs = encode_image(example_rgb_image_obs)


#     example_good_image = cv2.imread(os.path.join("/home/kylehatch/Desktop/hidql/results/single_task/calvin/public_model/checkpoint_only/calvin/gcdiffusion/auggoaldiff/seed_0/20240227_194024/checkpoint_150000/chat_gpt_dummy_filter_4s_2024.04.10_15.47.24/ep31/unfiltered_goal_images/timestep_000", "goal_image_001.png"))
#     example_good_image = example_good_image[..., ::-1]
#     example_good_image = cv2.resize(example_good_image, (512, 512))
#     base64_example_good_image = encode_image(example_good_image)

#     example_bad_image = cv2.imread(os.path.join("/home/kylehatch/Desktop/hidql/results/single_task/calvin/public_model/checkpoint_only/calvin/gcdiffusion/auggoaldiff/seed_0/20240227_194024/checkpoint_150000/chat_gpt_dummy_filter_4s_2024.04.10_15.47.24/ep31/unfiltered_goal_images/timestep_000", "goal_image_000.png"))
#     example_bad_image = example_bad_image[..., ::-1]
#     example_bad_image = cv2.resize(example_bad_image, (512, 512))
#     base64_example_bad_image = encode_image(example_bad_image)


    

#     base64_rgb_obs = encode_image(rgb_image_obs)

#     prompt1 = f"We have a computer simulation of a robot tabletop manipulation environment. \
# In this simulated environment, we are using a machine learning algorithm to control the robot so that \
# it completes the following task: \"{language_goal}.\" The robot is partway through completing the task. We have an image of the environment taken from a third person camera. We will call this image the image observation. \
# We also have a second image. This image is generated by a neural network, and shows the robot what it should do a few seconds into the future in order to come closer to completing the task \"{language_goal}\". This image is called the goal image. The robot will \
# compare the image observation and the goal image, and figure out how it needs to move its gripper and what objects it needs to move or manipulate in order to make the environment match what is shown in the goal image. \
# Since the goal image shows what the environment should look like a few seconds into the future, it will have some differences from the image observation. These differences will primarily be the location of the robot gripper and position of the robot arm, as well \
# as the locations and positions of any objects that the robot is directly manipulating. However, sometimes the neural network will make errors when generating the goal image, which will cause the robot to become confused. These errors are when the goal image has harmful inconsistencies with the image observation. \
# The following is a list of the most common types of harmful inconsistencies to look out for: \
# 1. Hallucinated objects. This is when there are objects that appear in the goal image but that do not appear in the image observation. Note, however, that the position of the robot gripper and arm in the image \
# observation may be covering up objects that are behind them. So, if the position of the robot arm or gripper changes between the goal image and the image observation, there may be objects that were previously \
# covered by the the robot arm or gripper in the image observation that are now visible in the goal image. These should not be considered hallucinations. \
# 2. Changes in object shape or color. This is when one or more objects in the goal image don't match their original shapes or colors in the image observation. Note, however, that slight changes in shape or color are okay and should not be considered inconsistencies \
# (such as a shift to a slightly different shade of the same color or a slight shape elongations or distortions), but major changes in shape or color (such as an object becoming a completely different shape or becoming a completely different color) are not okay and should be considered inconsistencies. \
# Note, however, that in both of these cases, these inconsistencies are only likely to be harmful if they involve objects that are directly related to completing the task \"{language_goal}\" or involve objects that the robot gripper is attempting to manipulate. Inconsistencies in the background of the image \
# or in parts of the image that the robot gripper is not interacting with are most likely not harmful and should not be taken into consideration. Similarly, inconsistencies in the image not related to completing the task \"{language_goal}\", or not related to the objects that the robot gripper is attempting to manipulate \
# should not be taken into consideration. \
# Here is an example of an image observation of the environment taken from the third person camera and a good goal image generated by the neural network. The first image is the image observation, and the second image is the goal image. While this good goal image may contain some slight inconsistencies or irrelevant inconsistencies with the image observation, it does not contain any \
# inconsistencies that are harmful to the robot completing its task."
    
#     prompt2 = f"In contrast, here is an example of an image observation and a bad goal image generated by the neural network. The first image is the image observation, and the second image is the goal image. This goal image contains some inconsistencies with the image observation \
# that are relevant to the task, and could cause the robot to become confused and hinder it from completing its task."


#     prompt3 = f"Now, here is the image observation of the environment taken from the third person camera and here is the goal image generated by the neural network. The first image is the image observation, and the second image is the goal image. \
# Are there any relevant inconsistencies between the image observation and the goal image that would cause the robot to become confused? Please include \"Yes\" or \"No\" in your answer."




#  prompt1 = f"We have a computer simulation of a robot tabletop manipulation environment. \
# In this simulated environment, we are using a machine learning algorithm to control the robot so that \
# it completes the following task: \"{language_goal}.\" The robot is partway through completing the task. We have an image of the environment taken from a third person camera. We will call this image the image observation. \
# We also have a second image. This image is generated by a neural network, and shows the robot what it should do a few seconds into the future in order to come closer to completing the task \"{language_goal}\". This image is called the goal image. The robot will \
# compare the image observation and the goal image, and figure out how it needs to move its gripper and what objects it needs to move or manipulate in order to make the environment match what is shown in the goal image. \
# Since the goal image shows what the environment should look like a few seconds into the future, it will have some differences from the image observation. These differences will primarily be the location of the robot gripper and position of the robot arm, as well \
# as the locations and positions of any objects that the robot is directly manipulating. However, sometimes the neural network will make errors when generating the goal image, which will cause the robot to become confused. These errors are when the goal image has harmful inconsistencies with the image observation. \
# The following is a list of the most common types of harmful inconsistencies to look out for: \
# 1. Hallucinated objects. This is when there are objects that appear in the goal image but that do not appear in the image observation. Note, however, that the position of the robot gripper and arm in the image \
# observation may be covering up objects that are behind them. So, if the position of the robot arm or gripper changes between the goal image and the image observation, there may be objects that were previously \
# covered by the the robot arm or gripper in the image observation that are now visible in the goal image. These should not be considered hallucinations. Also note that not all hallucinations will be relevant/harmful. Hallucinated objects that are in an area of the image that the robot is not interacting with, or \
# hallucinated objects that are not relevant to completing the task \"{language_goal}\" are unlikely to confuse the robot and so should not be considered relevant inconsistencies. \
# 2. Changes in object shape or color. This is when one or more objects in the goal image don't match their original shapes or colors in the image observation. Note, however, that slight changes in shape or color are okay and should not be considered inconsistencies \
# (such as a shift to a slightly different shade of the same color or a slight shape elongations or distortions), but major changes in shape or color (such as an object becoming a completely different shape or becoming a completely different color) are not okay and should be considered inconsistencies. \
# Note, however, that in both of these cases, these inconsistencies are only likely to be harmful if they involve objects that are directly related to completing the task \"{language_goal}\" or involve objects that the robot gripper is attempting to manipulate. Inconsistencies in the background of the image \
# or in parts of the image that the robot gripper is not interacting with are most likely not harmful and should not be taken into consideration. Similarly, inconsistencies in the image not related to completing the task \"{language_goal}\", or not related to the objects that the robot gripper is attempting to manipulate \
# should not be taken into consideration. \
# Here is an example of an image observation of the environment taken from the third person camera and a good goal image generated by the neural network. The first image is the image observation, and the second image is the goal image. While this good goal image may contain some slight inconsistencies or irrelevant inconsistencies with the image observation, it does not contain any \
# inconsistencies that are harmful to the robot completing its task."
    
#     prompt2 = f"In contrast, here is an example of an image observation and a bad goal image generated by the neural network. The first image is the image observation, and the second image is the goal image. This goal image contains some inconsistencies with the image observation \
# that are relevant to the task, and could cause the robot to become confused and hinder it from completing its task."


#     prompt3 = f"Now, here is the image observation of the environment taken from the third person camera and here is the goal image generated by the neural network. The first image is the image observation, and the second image is the goal image. \
# Are there any relevant/harmful inconsistencies between the image observation and the goal image that would cause the robot to become confused? Please include \"Yes\" or \"No\" in your answer."


#     answers = []

    
#     for i, goal_image in enumerate(goal_images):
        
#         base64_goal_image = encode_image(goal_image)
#         content = [
#             # {"type": "text","text": prompt1},
#             # {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_rgb_obs}", "detail": "high"}},
#             # {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_goal_image}", "detail": "high"}},
#             {"type": "text","text": prompt1},
#             {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_example_rgb_image_obs}", "detail": "high"}},
#             {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_example_good_image}", "detail": "high"}},

#             {"type": "text","text": prompt2},
#             {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_example_rgb_image_obs}", "detail": "high"}},
#             {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_example_bad_image}", "detail": "high"}},

#             {"type": "text","text": prompt3},
#             {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_rgb_obs}", "detail": "high"}},
#             {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_goal_image}", "detail": "high"}},
#         ]


# prompt1 = f"We have a computer simulation of a robot tabletop manipulation environment. \
# In this simulated environment, we are using a machine learning algorithm to control the robot so that \
# it completes the following task: \"{language_goal}.\" The robot is partway through completing the task. We have an image of the environment taken from a third person camera. We will call this image the image observation. \
# We also have a second image. This image is generated by a neural network, and shows the robot what it should do a few seconds into the future in order to come closer to completing the task \"{language_goal}\". This image is called the goal image. The robot will \
# compare the image observation and the goal image, and figure out how it needs to move its gripper and what objects it needs to move or manipulate in order to make the environment match what is shown in the goal image. \
# Since the goal image shows what the environment should look like a few seconds into the future, it will have some differences from the image observation. These differences will primarily be the location of the robot gripper and position of the robot arm, as well \
# as the locations and positions of any objects that the robot is directly manipulating. However, sometimes the neural network will make errors when generating the goal image, which will cause the robot to become confused. These errors are when the goal image has harmful inconsistencies with the image observation. \
# The following is a list of the most common types of harmful inconsistencies to look out for: \
# 1. Hallucinated objects. This is when there are objects that appear in the goal image but that do not appear in the image observation. Note, however, that the position of the robot gripper and arm in the image \
# observation may be covering up objects that are behind them. So, if the position of the robot arm or gripper changes between the goal image and the image observation, there may be objects that were previously \
# covered by the the robot arm or gripper in the image observation that are now visible in the goal image. These should not be considered hallucinations. Also note that not all hallucinations will be relevant/harmful. Hallucinated objects that are in an area of the image that the robot is not interacting with, or \
# hallucinated objects that are not relevant to completing the task \"{language_goal}\" are unlikely to confuse the robot and so should not be considered relevant inconsistencies. \
# 2. Changes in object shape or color. This is when one or more objects in the goal image don't match their original shapes or colors in the image observation. Note, however, that slight changes in shape or color are okay and should not be considered inconsistencies \
# (such as a shift to a slightly different shade of the same color or a slight shape elongations or distortions), but major changes in shape or color (such as an object becoming a completely different shape or becoming a completely different color) are not okay and should be considered inconsistencies. \
# Note, however, that in both of these cases, these inconsistencies are only likely to be harmful if they involve objects that are directly related to completing the task \"{language_goal}\" or involve objects that the robot gripper is attempting to manipulate. Inconsistencies in the background of the image \
# or in parts of the image that the robot gripper is not interacting with are most likely not harmful and should not be taken into consideration. Similarly, inconsistencies in the image not related to completing the task \"{language_goal}\", or not related to the objects that the robot gripper is attempting to manipulate \
# should not be taken into consideration. Now, here is the image observation of the environment taken from the third person camera and here are two the goal images generated by the neural network. The first image is the image observation, and the second image contains the two goal images, \
# stacked side-by-side. \
# Please take special care not to confuse the ordering of the two goal images. \
# Which of the two goal images is least likely to contain relevant/harmful inconsistencies with the image observation that would cause the robot to become confused? Is it the goal image on the left side, or the goal image on the right side? Please include \"Left\" or \"Right\" in your answer. \
# Please take special care not to confuse the ordering of the two goal images."

# # The first image is the image observation, and the second two images are the goal images. \
# # Which of the two goal images is least likely to contain relevant/harmful inconsistencies with the image observation that would cause the robot to become confused? Please include \"First\" or \"Second\" in your answer."