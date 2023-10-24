# NeRF: Neural Radiance Fields 

This project provides one of the most simplified implementation of the famous Neural Radiance Fields paper <a href="https://www.matthewtancik.com/nerf">"NeRF
Representing Scenes as Neural Radiance Fields for View Synthesis"</a>. 
## Overview: 
### 1. **Images' Ray Computation**:
- Determine the origins and directions for each camera frame concerning the world coordinate frame.
- Develop a function (`get_rays()`) to return the origins and directions for rays of an image.
- Visualize the entire data setup using `plot_all_poses()`.

### 2. **Ray Sampling**:
- Sample points along a ray using the equation \( r = o + t \cdot d \).
- Implement `stratified_sampling()` to generate points from each ray.

### 3. **NeRF MLP Design**:
- Design a neural network as per the NeRF paper to map the position and direction of a point along a ray to its color and density.
- Complete `nerf_model()` and the data preparation function `get_batch()`.
![](https://github.com/satyajeetburla/NeRF/blob/main/nerf1.PNG)

### 4. **Volume Rendering**:
- Use the volumetric rendering formula from the NeRF paper to approximate pixel colors based on sampled point colors and densities along a ray.
- Implement the `volumetric_rendering()` function to calculate the final color for each ray.

### 5. **Image Rendering**:
- Integrate all prior steps in the function `one_forward_pass()` to render a complete image from a dataset.
![](https://github.com/satyajeetburla/NeRF/blob/main/nerf2.PNG)

### 6. **Training**:
- Set up a training loop with predefined hyperparameters.
- Utilize the Adam optimizer and mean square error for training.
- Aim for a PSNR of 25 after 3,000 iterations.
## Result
![URL_OF_THE_IMAGE](https://github.com/satyajeetburla/NeRF/blob/main/Output%20Image.PNG)

