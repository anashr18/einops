import torch

# Example tensor: [batch_size, channels, height, width]
batch_size, channels, height, width = 10, 3, 32, 32
images = torch.randn(batch_size, channels, height, width)
print(f"Original shape: {images.shape}")

# Reshape to split each channel into 4x4 blocks
block_size = 4
new_height = height // block_size
new_width = width // block_size

# Step 1: Reshape the tensor
images_reshaped = images.view(batch_size, channels, new_height, block_size, new_width, block_size)
print(f"Shape after initial reshape: {images_reshaped.shape}")

# Step 2: Permute the dimensions
images_reshaped = images_reshaped.permute(0, 1, 2, 4, 3, 5)
print(f"Shape after permutation: {images_reshaped.shape}")

# Step 3: Final reshape
images_reshaped = images_reshaped.contiguous().view(batch_size, channels, new_height * new_width, block_size, block_size)
print(f"Final shape: {images_reshaped.shape}")
