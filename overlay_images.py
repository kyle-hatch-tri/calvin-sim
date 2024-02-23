import os 
import cv2
import numpy as np 



# def overlay_edge_detections(image1_path, image2_path, output_path, transparency=0.5):
def overlay_edge_detections(image1, image2, output_path, transparency=0.5):
    # # Read the images
    # image1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
    # image2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)
    image1_gray = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)
    image2_gray = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)

    # Apply Canny edge detection to each image
    edges1 = cv2.Canny(image1_gray, 50, 150)
    edges2 = cv2.Canny(image2_gray, 50, 150)

    # # Resize edge maps to ensure they have the same dimensions
    # edges2 = cv2.resize(edges2, (edges1.shape[1], edges1.shape[0]))
    

    # # Create a mask for the edges of the second image
    # edges2_mask = np.stack([edges2] * 3, axis=-1) 

    # # Create a blended image with transparency
    # blended = cv2.addWeighted(image1, 1.0 - transparency, edges2_mask, transparency, 0)


    edges1_im = np.stack([edges1, np.zeros_like(edges1), np.zeros_like(edges1)], axis=-1)
    edges2_im = np.stack([np.zeros_like(edges2), edges2, np.zeros_like(edges2)], axis=-1)

    overlayed_edges = edges1_im + edges2_im
    # overlayed_real = 

    real_images = np.concatenate([image1, image2, np.zeros_like(image2)], axis=1)
    edges = np.concatenate([edges1_im, edges2_im, overlayed_edges], axis=1)
    output_image = np.concatenate([real_images, edges], axis=0)

    # Save the result
    cv2.imwrite(output_path, output_image[..., ::-1])

# Example usage
image1_path = '/home/kylehatch/Desktop/hidql/calvin-sim/experiments/jax_model/public_checkpoint/only_checkpoint/public_checkpoint/only_checkpoint/no_vf/checkpoint_none/50_denoising_steps/1_samples/tmpensb/2024.02.14_17.17.41/ep0/goal_images.npy'
image2_path = '/home/kylehatch/Desktop/hidql/calvin-sim/experiments/jax_model/public_checkpoint/only_checkpoint/public_checkpoint/only_checkpoint/no_vf/checkpoint_none/50_denoising_steps/1_samples/tmpensb/2024.02.14_17.18.52/ep7/goal_images.npy'
output_dir = os.path.join("overlay_images")
os.makedirs(output_dir, exist_ok=True)

goal_images1 = np.load(image1_path)
goal_images2 = np.load(image2_path)

for i, (goal_image1, goal_image2 )in enumerate(zip(goal_images1, goal_images2)):
    output_path = os.path.join(output_dir, f"overlay_{i:02d}.png")
    print(output_path)

    overlay_edge_detections(goal_image1, goal_image2, output_path, transparency=0.7)