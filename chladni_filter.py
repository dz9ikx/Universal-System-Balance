import numpy as np
from skimage import io, filters, color, exposure
import matplotlib.pyplot as plt

def apply_chladni_filter(image_path, zeta_c=1.0 / np.sqrt(2)):
    """
    Applies the Chladni filter based on the 0.707 invariant theory.
    Uses Difference of Gaussians (DoG) with a scale ratio of sqrt(2) 
    to extract the structural 'skeleton' of the image.
    """
    # 1. Load image and convert to grayscale
    image = io.imread(image_path)
    if len(image.shape) == 3:
        # Convert to grayscale if it's a color image
        image = color.rgb2gray(image)

    # 2. Define scales with the invariant ratio sqrt(2)
    sigma1 = 1.0
    sigma2 = sigma1 * np.sqrt(2)

    # 3. Calculate Difference of Gaussians (DoG)
    # This approximates the Laplacian of Gaussian operator
    dog_image = filters.gaussian(image, sigma=sigma1) - filters.gaussian(image, sigma=sigma2)

    # 4. Calculate the Root Mean Square (RMS) of the DoG image
    rms_dog = np.sqrt(np.mean(dog_image**2))

    # 5. Define the dynamic threshold T = zeta_c * RMS(DoG)
    threshold = zeta_c * rms_dog

    # 6. Apply the threshold to find structural edges/nodes
    # The 'Edge' image is binary: 1 where structure dominates noise, 0 otherwise
    edge_image = (np.abs(dog_image) > threshold).astype(float)

    return edge_image, dog_image

# Example usage (needs a sample image, e.g., 'sample_image.jpg' in the same folder):
if __name__ == '__main__':
    # You would need an image file named 'sample_image.jpg' in your local directory
    # For a simple test without external files, we can create a dummy image:
    
    # Create a synthetic image with a simple structure (e.g., a circle)
    x, y = np.indices((256, 256))
    center_x, center_y = 128, 128
    radius = 50
    synthetic_image = np.zeros((256, 256))
    synthetic_image[ (x - center_x)**2 + (y - center_y)**2 < radius**2 ] = 1
    synthetic_image += np.random.normal(0, 0.1, synthetic_image.shape) # Add noise

    # Save the synthetic image temporarily to run the filter
    io.imsave('synthetic_test_image.png', synthetic_image, check_contrast=False)
    
    # Apply the filter
    skeleton, dog_result = apply_chladni_filter('synthetic_test_image.png')

    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    ax = axes.ravel()
    
    ax[0].imshow(synthetic_image, cmap='gray')
    ax[0].set_title("Original (Synthetic Image + Noise)")
    ax[0].axis('off')
    
    ax[1].imshow(dog_result, cmap='viridis')
    ax[1].set_title("Difference of Gaussians (DoG)")
    ax[1].axis('off')

    ax[2].imshow(skeleton, cmap='gray')
    ax[2].set_title(f"Structural Skeleton ($\zeta_c$ Threshold)")
    ax[2].axis('off')
    
    plt.tight_layout()
    plt.show()

