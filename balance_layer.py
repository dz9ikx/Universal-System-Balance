import torch
import torch.nn as nn
import numpy as np

class Balance0707Layer(nn.Module):
    """
    A dynamic activation layer based on the universal 0.707 invariant.
    It filters structural noise by applying a dynamic threshold derived from the RMS 
    of the input tensor, maximizing the number of distinguishable structural patterns.
    """
    def forward(self, x):
        # 1. Calculate RMS of activations (sigma)
        # We use dim=0 to calculate RMS per feature dimension if input is batched
        sigma = torch.sqrt(torch.mean(x**2, dim=0, keepdim=True))
        
        # 2. Define the dynamic threshold T = 0.707 * sigma (zeta_c * RMS)
        # We use a constant for zeta_c to be exact 1/sqrt(2)
        zeta_c = 1.0 / np.sqrt(2)
        threshold = zeta_c * sigma
        
        # 3. Create a mask: True where the signal magnitude exceeds the threshold
        # This highlights structural patterns vs. noise
        mask = (x.abs() > threshold).float()
        
        # 4. Apply the mask and apply energy compensation (sqrt(2) factor)
        # This maintains the overall energy balance in the network
        return x * mask * np.sqrt(2)

# Example usage of the layer:
if __name__ == '__main__':
    # Create a test tensor (e.g., 1000 samples, 5 features)
    input_tensor = torch.randn(1000, 5) * 2
    
    # Initialize our new layer
    balance_layer = Balance0707Layer()
    
    # Apply the layer
    output_tensor = balance_layer(input_tensor)
    
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output_tensor.shape}")
    print(f"Original Mean: {input_tensor.mean():.4f}, RMS: {torch.sqrt(torch.mean(input_tensor**2)):.4f}")
    print(f"Filtered Mean: {output_tensor.mean():.4f}, RMS: {torch.sqrt(torch.mean(output_tensor**2)):.4f}")

