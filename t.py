
# Dataset class for images
class ActionImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = []
        self.labels = []
        self.classes = os.listdir(root_dir)
        for label, action in enumerate(self.classes):
            action_dir = os.path.join(root_dir, action)
            for img_file in os.listdir(action_dir):
                img_path = os.path.join(action_dir, img_file)
                self.data.append(img_path)
                self.labels.append(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        label = self.labels[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
        img = Image.fromarray(img)  # Convert NumPy array to PIL image
        if self.transform:
            img = self.transform(img)
        return img, label


# Dataset class for videos
# class ActionVideoDataset(Dataset):
#     def __init__(self, video_dir, num_frames=16, transform=None):
#         self.video_dir = video_dir
#         self.num_frames = num_frames
#         self.transform = transform
#         self.video_files = [os.path.join(video_dir, f) for f in os.listdir(video_dir) if f.endswith('.mp4')]

#     def __len__(self):
#         return len(self.video_files)

#     def __getitem__(self, idx):
#         video_path = self.video_files[idx]
#         cap = cv2.VideoCapture(video_path)
#         frames = []
#         while len(frames) < self.num_frames:
#             ret, frame = cap.read()
#             if not ret:
#                 break
#             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             frame = Image.fromarray(frame)  # Convert NumPy array to PIL image
#             if self.transform:
#                 frame = self.transform(frame)
#             frames.append(frame)
#         cap.release()
#         # Pad missing frames with zeros if video is too short
#         while len(frames) < self.num_frames:
#             frames.append(torch.zeros_like(frames[0]))
#         frames = torch.stack(frames)
#         return frames, video_path
# class ActionVideoDataset(Dataset):
#     def __init__(self, video_dir, num_frames=16, transform=None):
#         self.video_dir = video_dir
#         self.num_frames = num_frames
#         self.transform = transform
#         self.classes = os.listdir(video_dir)  # List of action folder names
#         self.label_to_class = {i: cls for i, cls in enumerate(self.classes)}  # Map index to folder name
#         self.video_files = [(os.path.join(video_dir, cls, f), i) 
#                             for i, cls in enumerate(self.classes) 
#                             for f in os.listdir(os.path.join(video_dir, cls)) if f.endswith('.mp4')]

#     def __len__(self):
#         return len(self.video_files)

#     def __getitem__(self, idx):
#         video_path, label = self.video_files[idx]
#         cap = cv2.VideoCapture(video_path)
#         frames = []
#         while len(frames) < self.num_frames:
#             ret, frame = cap.read()
#             if not ret:
#                 break
#             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             frame = Image.fromarray(frame)
#             if self.transform:
#                 frame = self.transform(frame)
#             frames.append(frame)
#         cap.release()
#         while len(frames) < self.num_frames:
#             frames.append(torch.zeros_like(frames[0]))
#         frames = torch.stack(frames)
#         return frames, label, video_path

class ActionVideoDatasetSingle(Dataset):
    def __init__(self, video_dir, num_frames=16, transform=None):
        self.video_dir = video_dir
        self.num_frames = num_frames
        self.transform = transform
        self.classes = os.listdir(video_dir)  # List of action folder names
        self.label_to_class = {i: cls for i, cls in enumerate(self.classes)}  # Map index to folder name

        # Select one video per action label folder
        self.video_files = []
        for label, cls in enumerate(self.classes):
            action_folder = os.path.join(video_dir, cls)
            videos = [f for f in os.listdir(action_folder) if f.endswith('.mp4')]
            if videos:
                # Pick the first video from the folder (or a random video)
                self.video_files.append((os.path.join(action_folder, videos[0]), label))

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        video_path, label = self.video_files[idx]
        cap = cv2.VideoCapture(video_path)
        frames = []
        while len(frames) < self.num_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)  # Convert NumPy array to PIL image
            if self.transform:
                frame = self.transform(frame)
            frames.append(frame)
        cap.release()

        # Pad missing frames with zeros if the video is too short
        while len(frames) < self.num_frames:
            frames.append(torch.zeros_like(frames[0]))
        frames = torch.stack(frames)
        return frames, label, video_path


# Transforms
image_transform = Compose([
    Resize((224, 224)),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

video_transform = Compose([
    Resize((224, 224)),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load datasets
train_dataset = ActionImageDataset(root_dir='action_sp', transform=image_transform)
test_dataset = ActionVideoDatasetSingle(video_dir='resized_videos', transform=video_transform)

# Subset to train on 50 images
# train_subset = Subset(train_dataset, list(range(200)))
train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)

# Test loader
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)



# import os
# import cv2
# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
# from torchvision.transforms import Compose, ToTensor, Resize, Normalize, RandomHorizontalFlip, RandomRotation, ColorJitter
# from PIL import Image
# import numpy as np
# from collections import Counter

# # Dataset class for images
# class ActionImageDataset(Dataset):
#     def __init__(self, root_dir, transform=None):
#         self.root_dir = root_dir
#         self.transform = transform
#         self.data = []
#         self.labels = []
#         self.classes = os.listdir(root_dir)
#         for label, action in enumerate(self.classes):
#             action_dir = os.path.join(root_dir, action)
#             for img_file in os.listdir(action_dir):
#                 img_path = os.path.join(action_dir, img_file)
#                 self.data.append(img_path)
#                 self.labels.append(label)

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         img_path = self.data[idx]
#         label = self.labels[idx]
#         img = cv2.imread(img_path)
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
#         img = Image.fromarray(img)  # Convert NumPy array to PIL image
#         if self.transform:
#             img = self.transform(img)
#         return img, label


# # Dataset class for videos
# class ActionVideoDatasetSingle(Dataset):
#     def __init__(self, video_dir, num_frames=16, transform=None):
#         self.video_dir = video_dir
#         self.num_frames = num_frames
#         self.transform = transform
#         self.classes = os.listdir(video_dir)  # List of action folder names
#         self.label_to_class = {i: cls for i, cls in enumerate(self.classes)}  # Map index to folder name

#         # Select one video per action label folder
#         self.video_files = []
#         for label, cls in enumerate(self.classes):
#             action_folder = os.path.join(video_dir, cls)
#             videos = [f for f in os.listdir(action_folder) if f.endswith('.mp4')]
#             if videos:
#                 # Pick the first video from the folder (or a random video)
#                 self.video_files.append((os.path.join(action_folder, videos[0]), label))

#     def __len__(self):
#         return len(self.video_files)

#     def __getitem__(self, idx):
#         video_path, label = self.video_files[idx]
#         cap = cv2.VideoCapture(video_path)
#         frames = []
#         while len(frames) < self.num_frames:
#             ret, frame = cap.read()
#             if not ret:
#                 break
#             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             frame = Image.fromarray(frame)  # Convert NumPy array to PIL image
#             if self.transform:
#                 frame = self.transform(frame)
#             frames.append(frame)
#         cap.release()

#         # Pad missing frames with zeros if the video is too short
#         while len(frames) < self.num_frames:
#             frames.append(torch.zeros_like(frames[0]))
#         frames = torch.stack(frames)
#         return frames, label, video_path


# # Augmented transforms
# augmented_transform = Compose([
#     Resize((224, 224)),
#     RandomHorizontalFlip(),
#     RandomRotation(15),
#     ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
#     ToTensor(),
#     Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])

# # Regular transforms for test dataset
# video_transform = Compose([
#     Resize((224, 224)),
#     ToTensor(),
#     Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])

# # Load datasets
# train_dataset = ActionImageDataset(root_dir='action_sp', transform=augmented_transform)
# test_dataset = ActionVideoDatasetSingle(video_dir='resized_videos', transform=video_transform)

# # Class balancing with WeightedRandomSampler
# class_counts = Counter(train_dataset.labels)
# total_samples = sum(class_counts.values())
# class_weights = {label: total_samples / count for label, count in class_counts.items()}

# # Create a subset of the dataset
# train_subset = Subset(train_dataset, list(range(200)))

# # Adjust weights for the subset
# sample_weights = [class_weights[train_dataset.labels[idx]] for idx in train_subset.indices]
# sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(train_subset), replacement=True)

# # DataLoader with sampler for training
# train_loader = DataLoader(train_subset, batch_size=14, sampler=sampler)


# # Test loader
# test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)




# class RViT(nn.Module):
#     def __init__(self, num_classes, hidden_dim, num_layers, frame_dim):
#         super().__init__()
#         self.hidden_dim = hidden_dim
        
    #     # Patch embedding with reduced stride and kernel size
    #     # self.patch_embedding = nn.Conv3d(16, hidden_dim, kernel_size=(3, 8, 8), stride=(3, 4, 4), padding=(1, 2, 2))
    #     self.patch_embedding = nn.Conv3d(3, hidden_dim, kernel_size=(3, 8, 8), stride=(3, 4, 4), padding=(1, 2, 2))

    #     # Simplified learnable position encoding
    #     # Adjust the temporal dimension to match the expected input
    #     self.position_encoding = nn.Parameter(torch.randn(1, hidden_dim, 1, 56, 56), requires_grad=True)
        
    #     self.rvit_units = nn.ModuleList([RViTUnit(hidden_dim) for _ in range(num_layers)])
    #     self.classifier = nn.Linear(hidden_dim, num_classes)
        
    #     self.frame_reconstruction = nn.Sequential(
    #         nn.Conv3d(hidden_dim, 64, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
    #         nn.ReLU(),
    #         nn.Conv3d(64, frame_dim[0], kernel_size=(1, 3, 3), padding=(0, 1, 1))
    #     )
    #     self.temporal_upsample = nn.Upsample(size=(15, 224, 224), mode='trilinear', align_corners=False)

    # def forward(self, x):
    #     # Patch embedding
    #     patches = self.patch_embedding(x)  # Shape: [batch_size, hidden_dim, depth, height, width]
    #     # print("After patch embedding:", patches.shape)
        
    #     # Dynamically resize position encoding to match patches' spatial dimensions
    #     _, _, depth, height, width = patches.shape
    #     pos_encoding = F.interpolate(
    #         self.position_encoding, size=(depth, height, width), mode='trilinear', align_corners=False
    #     )  # Adjusted shape: [1, hidden_dim, depth, height, width]
    #     # print("Position encoding shape after interpolation:", pos_encoding.shape)
        
    #     # Add position encoding
        # patches += pos_encoding
        # # print("After adding position encoding:", patches.shape)

        # # Initialize recurrent state
        # h = torch.zeros_like(patches)
        # # print("Initialized recurrent state:", h.shape)

        # # Pass through RViT units
        # for i, unit in enumerate(self.rvit_units):
        #     h = unit(patches, h)
        #     # print(f"After RViT unit {i+1}:", h.shape)

        # # Classification
        # h_last = h.mean(dim=(2, 3, 4))  # Global average pooling over spatial dimensions
        # action_logits = self.classifier(h_last)
        # print("Action logits shape:", action_logits.shape)

        # # Frame reconstruction
        # reconstructed_frame = self.frame_reconstruction(h)
        # # print("Reconstructed frame shape:", reconstructed_frame.shape)

        # return action_logits, reconstructed_frame




import torch
import torch.nn as nn
import torch.nn.functional as F


class RViT(nn.Module):
    def __init__(self, num_classes, hidden_dim, num_layers, frame_dim):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Patch embedding
        self.patch_embedding = nn.Conv3d(3, hidden_dim, kernel_size=(3, 8, 8), stride=(3, 4, 4), padding=(1, 2, 2))
        self.position_encoding = nn.Parameter(torch.randn(1, hidden_dim, 1, 56, 56), requires_grad=True)

        # Spatiotemporal Transformer Units
        self.rvit_units = nn.ModuleList([RViTUnitWithTemporal(hidden_dim) for _ in range(num_layers)])
        
        # Classifier
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # Patch embedding
        patches = self.patch_embedding(x)
        
        # Add positional encoding
        _, _, depth, height, width = patches.shape
        pos_encoding = nn.functional.interpolate(self.position_encoding, size=(depth, height, width), mode='trilinear', align_corners=False)
        patches += pos_encoding

        # Flatten spatial dimensions for temporal processing
        batch_size, hidden_dim, depth, height, width = patches.size()
        patches = patches.permute(0, 2, 1, 3, 4).reshape(batch_size, depth, hidden_dim, -1)
        patches = patches.mean(-1)  # Average pooling across spatial dimensions (or keep it flattened)

        # Recurrent Spatiotemporal Processing
        for unit in self.rvit_units:
            patches = unit(patches)

        # Classification
        h_last = patches.mean(dim=1)  # Global average pooling over the sequence length
        logits = self.classifier(h_last)
        return logits




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

class SpatiotemporalAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.spatial_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=8)
        self.temporal_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=8)
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        # x: [batch_size, sequence_length, hidden_dim]
        # Spatial attention
        spatial_out, _ = self.spatial_attention(x, x, x)  # Self-attention
        spatial_out = self.layer_norm1(x + spatial_out)   # Add & Norm

        # Temporal attention
        temporal_out, _ = self.temporal_attention(spatial_out, spatial_out, spatial_out)
        temporal_out = self.layer_norm2(spatial_out + temporal_out)  # Add & Norm

        return temporal_out


class RViTUnitWithTemporal(nn.Module):
    def __init__(self, hidden_dim, dropout_rate=0.3):
        super().__init__()
        self.spatial_temporal_block = SpatiotemporalAttention(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        # Spatiotemporal attention
        attention_out = self.spatial_temporal_block(x)
        attention_out = self.dropout(attention_out)

        # Feedforward network
        ffn_out = self.ffn(attention_out)
        output = self.layer_norm(attention_out + ffn_out)  # Add & Norm

        return output






import os
from collections import Counter
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, RandomHorizontalFlip, RandomRotation, ColorJitter, Resize, ToTensor, Normalize

# Dataset class for images
class ActionVideoFromImagesDataset(Dataset):
    def __init__(self, root_dir, num_frames=16, transform=None):
        """
        Args:
            root_dir (str): Path to the root directory containing action label folders.
            num_frames (int): Number of frames to sample per pseudo-video.
            transform (callable, optional): Transform to apply to each frame.
        """
        self.root_dir = root_dir
        self.num_frames = num_frames
        self.transform = transform
        self.data = []
        self.labels = []
        self.classes = os.listdir(root_dir)

        # Load dataset: Store paths of image files and corresponding labels
        for label, action in enumerate(self.classes):
            action_dir = os.path.join(root_dir, action)
            image_files = sorted(os.listdir(action_dir))  # Sort for consistent sequence
            self.data.append((image_files, label, action_dir))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Returns:
            frames (torch.Tensor): Tensor of shape [C, D, H, W].
            label (int): Action label.
        """
        image_files, label, action_dir = self.data[idx]

        # Sample `num_frames` images
        if len(image_files) >= self.num_frames:
            selected_files = image_files[:self.num_frames]  # Sequential sampling
        else:
            selected_files = image_files  # Use all available images

        frames = []
        for file in selected_files:
            img_path = os.path.join(action_dir, file)
            img = Image.open(img_path).convert("RGB")  # Ensure RGB format
            if self.transform:
                img = self.transform(img)
            frames.append(img)

        # Pad with empty frames if fewer than `num_frames`
        while len(frames) < self.num_frames:
            frames.append(torch.zeros_like(frames[0]))

        # Stack frames along the temporal depth dimension
        frames = torch.stack(frames, dim=0)  # Shape: [D, C, H, W]
        frames = frames.permute(1, 0, 2, 3)  # Rearrange to [C, D, H, W]

        return frames, label



class ActionVideoDatasetSingle(Dataset):
    def __init__(self, video_dir, num_frames=16, transform=None):
        self.video_dir = video_dir
        self.num_frames = num_frames
        self.transform = transform
        self.classes = os.listdir(video_dir)  # List of action folder names
        self.label_to_class = {i: cls for i, cls in enumerate(self.classes)}  # Map index to folder name

        # Select one video per action label folder
        self.video_files = []
        for label, cls in enumerate(self.classes):
            action_folder = os.path.join(video_dir, cls)
            videos = [f for f in os.listdir(action_folder) if f.endswith('.mp4')]
            if videos:
                self.video_files.append((os.path.join(action_folder, videos[0]), label))

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        video_path, label = self.video_files[idx]
        cap = cv2.VideoCapture(video_path)
        frames = []
        while len(frames) < self.num_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)  # Convert NumPy array to PIL image
            if self.transform:
                frame = self.transform(frame)
            frames.append(frame)
        cap.release()

        # If fewer frames than num_frames, pad with zeros
        while len(frames) < self.num_frames:
            frames.append(torch.zeros_like(frames[0]))

        # Stack frames along the depth dimension
        frames = torch.stack(frames, dim=0)  # Shape: [D, C, H, W]
        frames = frames.permute(1, 0, 2, 3)  # Rearrange to [C, D, H, W]

        print(f"Frame shape: {frames.shape}")  # Should be [3, num_frames, height, width]
        return frames, label, video_path




# Transforms
image_transform = Compose([
    Resize((224, 224)),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

from torchvision.transforms import RandomCrop, RandomHorizontalFlip

video_transform = Compose([
    Resize((224, 224)),
    RandomHorizontalFlip(p=0.5),
    RandomCrop((200, 200), pad_if_needed=True),  # Adds spatial variety
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


test_dataset = ActionVideoDatasetSingle(video_dir='resized_videos', num_frames=16, transform=video_transform)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)  # Adjust batch_size if needed


# Parameters for class balancing
target_count = 15  # Define the target number of images per class

# Load datasets with augmentation for imbalance correction
train_dataset = ActionVideoFromImagesDataset(
    root_dir='action_sp',
    num_frames=16,
    transform=image_transform
)
train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
test_dataset = ActionVideoDatasetSingle(video_dir='resized_videos', transform=video_transform)

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Confirm updated dataset statistics
print(f"Number of training samples: {len(train_dataset)}")


import os
from PIL import Image
import torch
from torchvision.transforms import (
    Compose, RandomHorizontalFlip, RandomRotation, ColorJitter,
    RandomResizedCrop, GaussianBlur, RandomPerspective
)
from tqdm import tqdm

class BalancedAugmentor:
    def __init__(self, root_dir, target_count=100, transform=None):
        """
        Args:
            root_dir (str): Path to the original dataset root directory.
            target_count (int): Desired number of images per action label after augmentation.
            transform (callable, optional): Transform to apply for augmentation.
        """
        self.root_dir = root_dir
        self.target_count = target_count
        self.transform = transform if transform else self.default_augmentation()

    def default_augmentation(self):
        """
        Defines the augmentation pipeline for intense augmentation.
        """
        return Compose([
            RandomResizedCrop(size=(224, 224), scale=(0.8, 1.2)),  # Random crop and resize
            RandomHorizontalFlip(p=0.5),
            RandomRotation(degrees=30),  # Random rotation
            ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
            GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 2.0)),
            RandomPerspective(distortion_scale=0.5, p=0.5),
        ])

    def augment_class(self, label_dir, label, current_count):
        """
        Augments a specific class to reach the target number of images.
        Args:
            label_dir (str): Directory containing images for the specific class label.
            label (str): Name of the label.
            current_count (int): Current number of images in the class folder.
        """
        image_files = os.listdir(label_dir)
        num_to_augment = self.target_count - current_count

        if num_to_augment > 0:
            print(f"Augmenting class '{label}' with {num_to_augment} new images.")
            for i in range(num_to_augment):
                img_path = os.path.join(label_dir, image_files[i % len(image_files)])
                img = Image.open(img_path).convert("RGB")  # Ensure all images are in RGB mode

                # Apply augmentation
                augmented_img = self.transform(img)

                # Save augmented image
                augmented_img_path = os.path.join(label_dir, f"{label}_aug_{i}.jpg")
                augmented_img.save(augmented_img_path)

    def balance_dataset(self):
        """
        Balances all classes by augmenting underrepresented ones.
        """
        action_labels = os.listdir(self.root_dir)
        for label in tqdm(action_labels, desc="Balancing Classes"):
            label_dir = os.path.join(self.root_dir, label)
            if not os.path.isdir(label_dir):
                continue  # Skip if it's not a directory
            current_count = len(os.listdir(label_dir))
            if current_count < self.target_count:
                self.augment_class(label_dir, label, current_count)

# Parameters
root_dir = 'action_sp'  # Original dataset path
target_count = 100  # Desired number of images per action label

# Perform balancing augmentation
augmentor = BalancedAugmentor(root_dir, target_count=target_count)
augmentor.balance_dataset()

# Verify final counts
final_counts = {label: len(os.listdir(os.path.join(root_dir, label))) for label in os.listdir(root_dir)}
print(f"Final image counts per class: {final_counts}")



import os

def count_images_in_folder(folder_path, extensions=('jpg', 'jpeg', 'png')):
    total_images = 0
    
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(extensions):
                total_images += 1
                
            
    return total_images

# Example usage
folder_path = "action_sp"  # Replace with your folder path
total_images = count_images_in_folder(folder_path)
print(f"Total images: {total_images}")


import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

# Parameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)  # Assuming the model is defined
scaler = GradScaler()  # Mixed-precision training
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()  # Replace with your loss function
gradient_accumulation_steps = 4  # Accumulate gradients for 4 steps
num_epochs = 10

# Training Loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    optimizer.zero_grad()
    
    for step, (inputs, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")):
        inputs, labels = inputs.to(device), labels.to(device)
        if inputs.ndim == 4:  # If depth dimension is missing
            inputs = inputs.unsqueeze(2)
        
        with autocast():
            outputs = model(inputs)
            if isinstance(outputs, tuple):  # Extract logits if model returns a tuple
                outputs = outputs[0]
            loss = criterion(outputs, labels)
            loss = loss / gradient_accumulation_steps

        
        # Backpropagation with scaling
        scaler.scale(loss).backward()

        # Update weights after gradient accumulation steps
        if (step + 1) % gradient_accumulation_steps == 0 or (step + 1) == len(train_loader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        running_loss += loss.item() * gradient_accumulation_steps

    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")



import torch
import torch.nn as nn
import torch.nn.functional as F

class RViT(nn.Module):
    def __init__(self, num_classes, hidden_dim, num_layers, frame_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.patch_embedding = nn.Conv3d(3, hidden_dim, kernel_size=(3, 8, 8), stride=(3, 4, 4), padding=(1, 2, 2))
        self.position_encoding = nn.Parameter(torch.randn(1, hidden_dim, 1, 56, 56), requires_grad=True)
        self.rvit_units = nn.ModuleList([RViTUnit(hidden_dim) for _ in range(num_layers)])
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        patches = self.patch_embedding(x)
        _, _, depth, height, width = patches.shape
        pos_encoding = F.interpolate(self.position_encoding, size=(depth, height, width), mode='trilinear', align_corners=False)
        patches += pos_encoding
        h = torch.zeros_like(patches).to(patches.device)
        for unit in self.rvit_units:
            h = unit(patches, h)
        h_last = h.mean(dim=(2, 3, 4))
        logits = self.classifier(h_last)
        return logits, h

class RViTUnit(nn.Module):
    def __init__(self, hidden_dim, dropout_rate=0.3):
        super().__init__()
        self.attention_gate = LinearAttention(hidden_dim)
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, h):
        attn_output = self.attention_gate(x, h)
        attn_output = self.dropout(attn_output)
        batch_size, hidden_dim, depth, height, width = attn_output.shape
        h_flat = h.permute(0, 2, 3, 4, 1).reshape(-1, hidden_dim)
        attn_output_flat = attn_output.permute(0, 2, 3, 4, 1).reshape(-1, hidden_dim)
        h_new_flat = self.layer_norm1(h_flat + attn_output_flat)
        h_new = h_new_flat.reshape(batch_size, depth, height, width, hidden_dim).permute(0, 4, 1, 2, 3)
        h_new_flat = h_new.permute(0, 2, 3, 4, 1).reshape(-1, hidden_dim)
        h_new_flat = self.layer_norm2(h_new_flat + self.ffn(h_new_flat))
        h_new = h_new_flat.reshape(batch_size, depth, height, width, hidden_dim).permute(0, 4, 1, 2, 3)
        return h_new

class LinearAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.Wq = nn.Conv3d(hidden_dim, hidden_dim, kernel_size=1)
        self.Wk = nn.Conv3d(hidden_dim, hidden_dim, kernel_size=1)
        self.Wv = nn.Conv3d(hidden_dim, hidden_dim, kernel_size=1)

    def forward(self, x, h):
        q = self.Wq(x).flatten(start_dim=2)
        k = self.Wk(h).flatten(start_dim=2)
        v = self.Wv(h).flatten(start_dim=2)
        attn_weights = torch.bmm(q.transpose(1, 2), k) / (k.size(1) ** 0.5)
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_output = torch.bmm(v, attn_weights.transpose(1, 2))
        return attn_output.view_as(h)


import os
import cv2
from PIL import Image
from collections import Counter
from random import shuffle
from torch.utils.data import Dataset

class ActionImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Dataset for loading images grouped by classes into subfolders.
        Each subfolder represents a class.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.data = []  # List to hold image paths
        self.labels = []  # List to hold corresponding class labels
        self.classes = sorted(os.listdir(root_dir))  # List of class names (subfolder names)

        for label, action in enumerate(self.classes):  # Iterate through subfolders
            action_dir = os.path.join(root_dir, action)
            if not os.path.isdir(action_dir):
                continue  # Skip non-folder files (if any)
            for img_file in os.listdir(action_dir):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):  # Check for image files
                    img_path = os.path.join(action_dir, img_file)
                    self.data.append(img_path)
                    self.labels.append(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        label = self.labels[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        if self.transform:
            img = self.transform(img)
        return img, label



class ActionVideoDatasetSingle(Dataset):
    def __init__(self, video_dir, num_frames=16, transform=None):
        self.video_dir = video_dir
        self.num_frames = num_frames
        self.transform = transform
        self.classes = os.listdir(video_dir)
        self.video_files = []
        for label, cls in enumerate(self.classes):
            action_folder = os.path.join(video_dir, cls)
            videos = [f for f in os.listdir(action_folder) if f.endswith('.mp4')]
            shuffle(videos)
            if videos:
                self.video_files.append((os.path.join(action_folder, videos[0]), label))

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        video_path, label = self.video_files[idx]
        cap = cv2.VideoCapture(video_path)
        frames = []
        while len(frames) < self.num_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            if self.transform:
                frame = self.transform(frame)
            frames.append(frame)
        cap.release()
        while len(frames) < self.num_frames:
            frames.append(torch.zeros_like(frames[0]))
        frames = torch.stack(frames)
        return frames, label


import os
import cv2
from PIL import Image
from collections import Counter
from random import shuffle
from torch.utils.data import Dataset

class ActionImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Dataset for loading images directly from a folder containing images (no subfolders).
        """
        self.root_dir = root_dir
        self.transform = transform
        self.data = []  # List to hold image paths
        self.labels = []  # List to hold corresponding action labels (optional, can be all 0)

        # Single class definition for compatibility
        self.classes = ['action_sp']  # Define a single class name

        # Collect all .jpg files in the directory
        for img_file in os.listdir(root_dir):
            if img_file.endswith('.jpg'):
                img_path = os.path.join(root_dir, img_file)  # Full path to the image
                self.data.append(img_path)
                self.labels.append(0)  # Assign a single class label (e.g., 0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        label = self.labels[idx]  # Will always be 0 in this case
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
        img = Image.fromarray(img)  # Convert NumPy array to PIL image
        if self.transform:
            img = self.transform(img)
        return img, label



class ActionVideoDatasetSingle(Dataset):
    def __init__(self, video_dir, num_frames=16, transform=None):
        self.video_dir = video_dir
        self.num_frames = num_frames
        self.transform = transform
        self.classes = os.listdir(video_dir)
        self.video_files = []
        for label, cls in enumerate(self.classes):
            action_folder = os.path.join(video_dir, cls)
            videos = [f for f in os.listdir(action_folder) if f.endswith('.mp4')]
            shuffle(videos)
            if videos:
                self.video_files.append((os.path.join(action_folder, videos[0]), label))

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        video_path, label = self.video_files[idx]
        cap = cv2.VideoCapture(video_path)
        frames = []
        while len(frames) < self.num_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            if self.transform:
                frame = self.transform(frame)
            frames.append(frame)
        cap.release()
        while len(frames) < self.num_frames:
            frames.append(torch.zeros_like(frames[0]))
        frames = torch.stack(frames)
        return frames, label


