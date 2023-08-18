import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import time

def positional_encoding(x, num_frequencies=6, incl_input=True):
    
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

class model_2d(nn.Module):
    
    """
    Define a 2D model comprising of three fully connected layers,
    two relu activations and one sigmoid activation.
    """
    
    def __init__(self, filter_size=128, num_frequencies=6):
        super().__init__()
        #############################  TODO 1(b) BEGIN  ############################
        D =  2
        L = num_frequencies
        imp = D + 2*L*D
        self.fc1 = nn.Linear(imp, filter_size)
        self.fc2 = nn.Linear(filter_size, filter_size)
        self.fc3 = nn.Linear(filter_size, 3)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()


        #############################  TODO 1(b) END  ##############################        

    def forward(self, x):
        #############################  TODO 1(b) BEGIN  ############################
        x = self.fc1(x)
        x = self.relu(x)
        
        x = self.fc2(x)
        x = self.relu(x)
        
        x = self.fc3(x)
        x = self.sigmoid(x)
     

        #############################  TODO 1(b) END  ##############################  
        return x  
    
def train_2d_model(test_img, num_frequencies, device, model=model_2d, positional_encoding=positional_encoding, show=True):

    # Optimizer parameters
    lr = 5e-4
    iterations = 10000
    height, width = test_img.shape[:2]

    # Number of iters after which stats are displayed
    display = 2000  
    
    # Define the model and initialize its weights.
    model2d = model(num_frequencies=num_frequencies)
    print(model2d)
    print(num_frequencies)
    model2d.to(device)

    def weights_init(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)

    model2d.apply(weights_init)

    #############################  TODO 1(c) BEGIN  ############################
    # Define the optimizer
    optimizer = torch.optim.Adam(model2d.parameters(), lr=lr)
    #############################  TODO 1(c) END  ############################

    # Seed RNG, for repeatability
    seed = 5670
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Lists to log metrics etc.
    psnrs = []
    iternums = []

    t = time.time()
    t0 = time.time()

    #############################  TODO 1(c) BEGIN  ############################
    # Create the 2D normalized coordinates, and apply positional encoding to them
    def create_coords(height, width):
      """
      Creates a tensor of size (height, width, 2) containing normalized coordinates
      ranging from -1 to 1.
      """
      y_range = torch.linspace(-1, 1, height, device=device)
      x_range = torch.linspace(-1, 1, width, device=device)
      y_coords, x_coords = torch.meshgrid(y_range, x_range)
      coords = torch.stack([y_coords, x_coords], dim=-1)
      return coords
    coords = create_coords(height, width).to(device)
    if positional_encoding is not None:
        coords = positional_encoding(coords,num_frequencies)
    #############################  TODO 1(c) END  ############################

    for i in range(iterations+1):
        optimizer.zero_grad()
        #############################  TODO 1(c) BEGIN  ############################
        # Run one iteration

        pred = model2d(coords)

        pred = pred.view(height, width, 3)
        loss = F.mse_loss(pred, test_img)

        # Compute mean-squared error between the predicted and target images. Backprop!
        loss.backward()
        optimizer.step()


        #############################  TODO 1(c) END  ############################

        # Display images/plots/stats
        if i % display == 0 and show:
            #############################  TODO 1(c) BEGIN  ############################
            # Calculate psnr
            with torch.no_grad():

                mse = F.mse_loss(pred, test_img)
                psnr = 10 * torch.log10(1.0 / mse)

            #############################  TODO 1(c) END  ############################

            print("Iteration %d " % i, "Loss: %.4f " % loss.item(), "PSNR: %.2f" % psnr.item(), \
                "Time: %.2f secs per iter" % ((time.time() - t) / display), "%.2f secs in total" % (time.time() - t0))
            t = time.time()

            psnrs.append(psnr.item())
            iternums.append(i)

            plt.figure(figsize=(13, 4))
            plt.subplot(131)
            plt.imshow(pred.detach().cpu().numpy())
            plt.title(f"Iteration {i}")
            plt.subplot(132)
            plt.imshow(test_img.cpu().numpy())
            plt.title("Target image")
            plt.subplot(133)
            plt.plot(iternums, psnrs)
            plt.title("PSNR")
            plt.show()

    print('Done!')
    return pred.detach().cpu()