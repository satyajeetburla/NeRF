import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import time
def positional_encoding(x, num_frequencies, incl_input=True):
    
    """
    Apply positional encoding to the input.
    
    Args:
    x (torch.Tensor): Input tensor to be positionally encoded. 
      The dimension of x is [N, D], where N is the number of input coordinates,
      and D is the dimension of the input coordinate.
    num_frequencies (optional, int): The number of frequencies used in
     the positional encoding (default: 6).
    incl_input (optional, bool): If True, concatenate the input with the 
        computed positional encoding (default: True).

    Returns:
    (torch.Tensor): Positional encoding of the input tensor. 
    """
    
    results = []
    if incl_input:
        results.append(x)
    #############################  TODO 1(a) BEGIN  ############################
    # encode input tensor and append the encoded tensor to the list of results.
    D = x.shape[1]
    L = num_frequencies
    import math
    # Encode input tensor and append the encoded tensor to the list of results.
    for i in range(L):
        freq = 2 ** i
        sin_encoding = torch.sin(freq * math.pi * x)
        cos_encoding = torch.cos(freq * math.pi * x)
        results.append(sin_encoding)
        results.append(cos_encoding)
    #############################  TODO 1(a) END  ##############################
    return torch.cat(results, dim=-1)
def get_rays(height, width, intrinsics, Rcw, Tcw):
    
    """
    Compute the origin and direction of rays passing through all pixels of an image (one ray per pixel).

    Args:
    height: the height of an image.
    width: the width of an image.
    intrinsics: camera intrinsics matrix of shape (3, 3).
    Rcw: Rotation matrix of shape (3,3) from camera to world coordinates.
    Tcw: Translation vector of shape (3,1) that transforms

    Returns:
    ray_origins (torch.Tensor): A tensor of shape (height, width, 3) denoting the centers of
      each ray. Note that desipte that all ray share the same origin, here we ask you to return 
      the ray origin for each ray as (height, width, 3).
    ray_directions (torch.Tensor): A tensor of shape (height, width, 3) denoting the
      direction of each ray.
    """

    #device = intrinsics.device
    # ray_directions = torch.zeros((height, width, 3), device=device)  # placeholder
    # ray_origins = torch.zeros((height, width, 3), device=device)  # placeholder
    ray_directions = torch.zeros((height, width, 3)) 
    ray_origins = torch.zeros((height, width, 3)) 
    #############################  TODO 2.1 BEGIN  ##########################  
    for i in range(height):
      for j in range(width):
        ray_origins[i,j] = Tcw.resize(3) 
        ray_directions[i,j] = (Rcw @ np.linalg.inv(intrinsics)@np.array([j,i,1])).resize(3)
    #############################  TODO 2.1 END  ############################
    return ray_origins, ray_directions

def stratified_sampling(ray_origins, ray_directions, near, far, samples):

    """
    Sample 3D points on the given rays. The near and far variables indicate the bounds of sampling range.

    Args:
    ray_origins: Origin of each ray in the "bundle" as returned by the
      get_rays() function. Shape: (height, width, 3).
    ray_directions: Direction of each ray in the "bundle" as returned by the
      get_rays() function. Shape: (height, width, 3).
    near: The 'near' extent of the bounding volume.
    far:  The 'far' extent of the bounding volume.
    samples: Number of samples to be drawn along each ray.
  
    Returns:
    ray_points: Query 3D points along each ray. Shape: (height, width, samples, 3).
    depth_points: Sampled depth values along each ray. Shape: (height, width, samples).
    """

    #############################  TODO 2.2 BEGIN  ############################
   
    sample_points = torch.arange(1, samples+1)
    depth_points = near + ((sample_points -1)/samples)*(far-near) 
    ray_points = ray_origins[...,None,:] + depth_points[...,None]*ray_directions[...,None,:]
 
    #############################  TODO 2.2 END  ############################
    return ray_points, depth_points
class nerf_model(nn.Module):
    
    """
    Define a NeRF model comprising eight fully connected layers and following the
    architecture described in the NeRF paper. 
    """

    def __init__(self, filter_size=256, num_x_frequencies=6, num_d_frequencies=3):
        super().__init__()

        #############################  TODO 2.3 BEGIN  ############################
        D1 = 3
        D2 = 3
        L1 = num_x_frequencies
        L2 = num_d_frequencies

        imp1 = D1 + 2*L1*D1
        imp2 = D2 + 2*L2*D2
 

        self.fc1 = nn.Linear(imp1, filter_size)
        self.fc2 = nn.Linear(filter_size, filter_size)
        self.fc3 = nn.Linear(filter_size, filter_size)
        self.fc4 = nn.Linear(filter_size, filter_size)
        self.fc5 = nn.Linear(filter_size , filter_size)
        self.fc6 = nn.Linear(filter_size+ imp1, filter_size)
        self.fc7 = nn.Linear(filter_size, filter_size)
        self.fc8 = nn.Linear(filter_size, filter_size)
        self.fc91 = nn.Linear(filter_size,  1)
        self.fc9 = nn.Linear(filter_size, filter_size)
        
        self.fc10 = nn.Linear(filter_size + imp2, 128)
        self.fc11 = nn.Linear(128, 3)

      #############################  TODO 2.3 END  ############################
    

    def forward(self, x, d):
        #############################  TODO 2.3 BEGIN  ############################
        inputx = x
        inputd = d
        out = self.fc1(inputx)
        out = F.relu(out)
        out = self.fc2(out)
        out = F.relu(out)
        out = self.fc3(out)
        out = F.relu(out)
        out = self.fc4(out)
        out = F.relu(out)
        out = self.fc5(out)
        out = F.relu(out)

        inpx1 = torch.cat([out, inputx], dim=-1)
        out = self.fc6(inpx1)
        out = F.relu(out)
        out = self.fc7(out)
        out = F.relu(out)
        out1 = self.fc8(out)
        out1 = F.relu(out1)
        sigma = self.fc91(out1)
        newout = self.fc9(out1)
        inpx2 = torch.cat([newout, inputd], dim=-1)
        out = self.fc10(inpx2)
        out = F.relu(out)
        out = self.fc11(out)
        rgb = torch.sigmoid(out)
        #############################  TODO 2.3 END  ############################
        return rgb, sigma
def get_batches(ray_points, ray_directions, num_x_frequencies, num_d_frequencies):
    
    def get_chunks(inputs, chunksize = 2**15):
        return [inputs[i:i + chunksize] for i in range(0, inputs.shape[0], chunksize)]
    
    """
    This function returns chunks of the ray points and directions to avoid memory errors with the
    neural network. It also applies positional encoding to the input points and directions before 
    dividing them into chunks, as well as normalizing and populating the directions.
    """
    # #############################  TODO 2.3 BEGIN  ############################

    ray_directions_norm = ray_directions / torch.norm(ray_directions, dim=-1, keepdim=True)

    ray_directions_flat = ray_directions_norm.unsqueeze(-2).repeat(1, 1, ray_points.shape[2], 1).view(-1, 3)

    ray_directions_encoded = positional_encoding(ray_directions_flat, num_d_frequencies)

    ray_points_flat = ray_points.reshape(-1, 3)

    ray_points_encoded = positional_encoding(ray_points_flat, num_x_frequencies)

    ray_points_batches = get_chunks(ray_points_encoded)

    ray_directions_batches = get_chunks(ray_directions_encoded)

   #############################  TODO 2.3 END  ############################

    return ray_points_batches, ray_directions_batches

def volumetric_rendering(rgb, s, depth_points):

    """
    Differentiably renders a radiance field, given the origin of each ray in the
    "bundle", and the sampled depth values along them.

    Args:
    rgb: RGB color at each query location (X, Y, Z). Shape: (height, width, samples, 3).
    sigma: Volume density at each query location (X, Y, Z). Shape: (height, width, samples).
    depth_points: Sampled depth values along each ray. Shape: (height, width, samples).
  
    Returns:
    rec_image: The reconstructed image after applying the volumetric rendering to every pixel.
    Shape: (height, width, 3)
    """
    
    #############################  TODO 2.4 BEGIN  ############################
    sigma = torch.nn.functional.relu(s)
    delta = depth_points[...,1:] - depth_points[...,:-1]
    pad =    torch.tensor([1e9]).expand(depth_points[..., :1].shape)  
    delta =  torch.cat([delta, pad],dim = -1)
    pd = torch.exp(-sigma*delta)
    m_pd = 1 - pd
    ti = torch.cumprod(1-m_pd+1e-9, dim =-1)
    ti = torch.roll(ti, shifts = 1,dims=-1)
    c_dash =m_pd * ti
    rec_image = (c_dash[..., None] * rgb).sum(dim=-2)
    #############################  TODO 2.4 END  ############################

    return rec_image

def one_forward_pass(height, width, intrinsics, pose, near, far, samples, model, num_x_frequencies, num_d_frequencies):
    
    #############################  TODO 2.5 BEGIN  ############################
    Rcw = pose[:3, :3].reshape((3,3))
    Tcw = pose[:3, -1].reshape((3,1))

    #compute all the rays from the image
    ray_origins, ray_directions= get_rays(height, width, intrinsics,Rcw ,Tcw )
    #sample the points from the rays
    ray_points, depth_points = stratified_sampling(ray_origins, ray_directions, near, far, samples)

    ray_points_batches, ray_directions_batches = get_batches(ray_points, ray_directions, num_x_frequencies, num_d_frequencies)
    # Forward pass the batches and concatenate the outputs at the end
    all_rgb = []
    all_sigma = []
    for i in range(len(ray_points_batches)):
          ray_points_i = ray_points_batches[i].float()
          ray_directions_i = ray_directions_batches[i].float()
          rgb_i, sigma_i  = model(ray_points_i,ray_directions_i)
          all_rgb.append(rgb_i)
          all_sigma.append(sigma_i)
    rgb = torch.concat(all_rgb).reshape((height,width,samples,3))
    sigma = torch.concat(all_sigma).reshape((height,width,samples))
    # Apply volumetric rendering to obtain the reconstructed image
    rec_image= volumetric_rendering(rgb, sigma, depth_points)
    #############################  TODO 2.5 END  ############################

    return rec_image