import os
import random
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define transformation for ImageNet
imagenet_transforms = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to match input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load ImageNet dataset
imagenet_dataset = datasets.ImageFolder(root="/Users/junghwang/Downloads/cs593p/CS593/categorized_train", transform=imagenet_transforms)

# Simulate temporal stacking of similar images
class ImageNetSimulatedVideos(torch.utils.data.Dataset):
    def __init__(self, dataset, num_frames=16):
        """
        Creates a dataset where each sample is a stack of `num_frames` images
        from the same class.
        """
        self.dataset = dataset
        self.num_frames = num_frames
        self.class_to_indices = self._group_by_class()

    def _group_by_class(self):
        """Groups dataset indices by class."""
        class_to_indices = {}
        for idx, (_, label) in enumerate(self.dataset.samples):
            if label not in class_to_indices:
                class_to_indices[label] = []
            class_to_indices[label].append(idx)
        return class_to_indices

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Randomly sample `num_frames` images from the same class
        _, label = self.dataset.samples[idx]
        indices = self.class_to_indices[label]
        sampled_indices = random.choices(indices, k=self.num_frames)
        frames = [self.dataset[i][0] for i in sampled_indices]  # Load images
        frames = torch.stack(frames, dim=0)  # Stack along temporal dimension
        return frames, label

# Simulated ImageNet dataset
imagenet_simulated = ImageNetSimulatedVideos(imagenet_dataset, num_frames=16)

# DataLoader for ImageNet simulated videos
imagenet_dataloader = DataLoader(imagenet_simulated, batch_size=8, shuffle=True, num_workers=4)
import torch
import torch.nn as nn
import torch.nn.functional as F

class RViT(nn.Module):
    def __init__(self, num_classes, hidden_dim, num_layers, frame_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Patch embedding with reduced stride and kernel size
        self.patch_embedding = nn.Conv3d(16, hidden_dim, kernel_size=(3, 8, 8), stride=(3, 4, 4), padding=(1, 2, 2))
        
        # Simplified learnable position encoding
        # Adjust the temporal dimension to match the expected input
        self.position_encoding = nn.Parameter(torch.randn(1, hidden_dim, 1, 56, 56), requires_grad=True)
        
        self.rvit_units = nn.ModuleList([RViTUnit(hidden_dim) for _ in range(num_layers)])
        self.classifier = nn.Linear(hidden_dim, num_classes)
        
        self.frame_reconstruction = nn.Sequential(
            nn.Conv3d(hidden_dim, 64, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.ReLU(),
            nn.Conv3d(64, frame_dim[0], kernel_size=(1, 3, 3), padding=(0, 1, 1))
        )
        self.temporal_upsample = nn.Upsample(size=(15, 224, 224), mode='trilinear', align_corners=False)

    def forward(self, x):
        # Patch embedding
        patches = self.patch_embedding(x)  # Shape: [batch_size, hidden_dim, depth, height, width]
        
        # Adjust position encoding to match the temporal dimension of `patches`
        pos_encoding = self.position_encoding.repeat(1, 1, patches.shape[2], 1, 1)  # Shape: [1, hidden_dim, depth, height, width]
        
        # Add position encoding
        patches += pos_encoding
        
        # Initialize recurrent state with spatial dimensions
        h = torch.zeros_like(patches)
        
        # Pass through RViT units
        for unit in self.rvit_units:
            h = unit(patches, h)
        
        # Use final recurrent state for classification
        h_last = h.mean(dim=(2, 3, 4))  # Global average pooling over spatial dimensions
        action_logits = self.classifier(h_last)
        
        # Frame reconstruction
        reconstructed_frame = self.frame_reconstruction(h)
        return action_logits, reconstructed_frame


class ScaledDotProductAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.Wq = nn.Conv3d(hidden_dim, hidden_dim, kernel_size=1)
        self.Wk = nn.Conv3d(hidden_dim, hidden_dim, kernel_size=1)
        self.Wv = nn.Conv3d(hidden_dim, hidden_dim, kernel_size=1)
        self.scale = hidden_dim ** -0.5

    def forward(self, x, h):
        # Compute queries, keys, and values
        q = self.Wq(x)
        k = self.Wk(h)
        v = self.Wv(h)
        
        # Compute scaled dot-product attention
        attn = torch.softmax((q * k).sum(dim=1, keepdim=True) * self.scale, dim=-1)
        output = attn * v  # Apply attention weights to values
        
        return output
        
class LinearAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.Wq = nn.Conv3d(hidden_dim, hidden_dim, kernel_size=1)
        self.Wk = nn.Conv3d(hidden_dim, hidden_dim, kernel_size=1)
        self.Wv = nn.Conv3d(hidden_dim, hidden_dim, kernel_size=1)
        self.Wo = nn.Conv3d(hidden_dim, hidden_dim, kernel_size=1)

    def forward(self, x, h):
        # Compute queries, keys, and values
        q = self.Wq(x)  # Shape: [batch_size, hidden_dim, depth, height, width]
        k = self.Wk(h)  # Shape: [batch_size, hidden_dim, depth, height, width]
        v = self.Wv(h)  # Shape: [batch_size, hidden_dim, depth, height, width]
        
        # Reshape q and k for attention computation
        q = q.flatten(start_dim=2)  # Shape: [batch_size, hidden_dim, depth*height*width]
        k = k.flatten(start_dim=2)  # Shape: [batch_size, hidden_dim, depth*height*width]
        v = v.flatten(start_dim=2)  # Shape: [batch_size, hidden_dim, depth*height*width]

        # Compute attention weights
        attn_weights = torch.bmm(q.transpose(1, 2), k)  # Shape: [batch_size, depth*height*width, depth*height*width]
        attn_weights = attn_weights / (k.size(1) ** 0.5)  # Scale by sqrt of hidden_dim
        attn_weights = torch.softmax(attn_weights, dim=-1)  # Apply softmax over last dimension

        # Apply attention weights to values
        attn_output = torch.bmm(v, attn_weights.transpose(1, 2))  # Shape: [batch_size, hidden_dim, depth*height*width]
        
        # Reshape back to 3D
        attn_output = attn_output.view_as(h)  # Shape: [batch_size, hidden_dim, depth, height, width]

        # Final projection to match the input shape
        output = self.Wo(attn_output)  # Shape: [batch_size, hidden_dim, depth, height, width]
        return output



class RViTUnit(nn.Module):
    def __init__(self, hidden_dim, dropout_rate=0.1):
        super().__init__()
        self.attention_gate = LinearAttention(hidden_dim)  # Use LinearAttention here
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        self.recurrent_dropout = nn.Dropout(dropout_rate)

    def forward(self, x, h):
        # Attention mechanism with linear attention
        attn_output = self.attention_gate(x, h)
        attn_output = self.recurrent_dropout(attn_output)
    
        # Reshape for LayerNorm
        batch_size, hidden_dim, depth, height, width = attn_output.shape
        h_flat = h.permute(0, 2, 3, 4, 1).reshape(-1, hidden_dim)
        attn_output_flat = attn_output.permute(0, 2, 3, 4, 1).reshape(-1, hidden_dim)
    
        # Apply LayerNorm
        h_new_flat = self.layer_norm1(h_flat + attn_output_flat)
    
        # Reshape back to original shape
        h_new = h_new_flat.reshape(batch_size, depth, height, width, hidden_dim).permute(0, 4, 1, 2, 3)
    
        # Apply FFN with LayerNorm
        h_new_flat = h_new.permute(0, 2, 3, 4, 1).reshape(-1, hidden_dim)
        h_new_flat = self.layer_norm2(h_new_flat + self.ffn(h_new_flat))
    
        h_new = h_new_flat.reshape(batch_size, depth, height, width, hidden_dim).permute(0, 4, 1, 2, 3)
        return h_new



# Initialize RViT model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = RViT(num_classes=len(imagenet_dataset.classes), hidden_dim=512, num_layers=2, frame_dim=(3, 224, 224)).to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Training loop for ImageNet simulated videos
num_epochs = 5
for epoch in range(num_epochs):
    print(f"Epoch [{epoch + 1}/{num_epochs}] - Pretraining on ImageNet")
    model.train()
    running_loss = 0.0
    running_corrects = 0

    for inputs, labels in imagenet_dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            logits, _ = model(inputs)  # Pass through RViT
            loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        # Update metrics
        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(logits, 1)
        running_corrects += torch.sum(preds == labels)

    epoch_loss = running_loss / len(imagenet_simulated)
    epoch_acc = running_corrects.double() / len(imagenet_simulated)
    print(f"Pretrain Loss: {epoch_loss:.4f} - Pretrain Accuracy: {epoch_acc:.4f}")

# Save pretrained weights
torch.save(model.state_dict(), "rvit_imagenet_pretrained.pth")
