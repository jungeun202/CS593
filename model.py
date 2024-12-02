import os
import csv
import subprocess
import random
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Paths
base_path = "/scratch/scholar/hwang227/CS593-main"
processed_frames_path = os.path.join(base_path, 'processed_frames')
train_csv = os.path.join(base_path, 'train_annotations.csv')
val_csv = os.path.join(base_path, 'val_annotations.csv')


def create_train_val_annotations(frames_folder, train_csv, val_csv, val_split=0.2):
    """
    Generate training and validation annotation files from the extracted frames directory
    for a structure where individual frames are stored directly under label folders.
    """
    all_annotations = []

    # Iterate through each label (category) folder
    for label_folder in os.listdir(frames_folder):
        label_path = os.path.join(frames_folder, label_folder)
        if not os.path.isdir(label_path) or label_folder.startswith('.'):
            # print(f"Skipping invalid or hidden label folder: {label_folder}")
            continue

        # Iterate through each frame file within the label folder
        valid_frame_count = 0
        for frame_file in os.listdir(label_path):
            frame_file_path = os.path.join(label_path, frame_file)
            # Remove the redundant check here
            # Check whether the frame is a valid file (optional based on your dataset structure)
            if not os.path.isfile(frame_file_path) or frame_file.startswith('.'):
                # print(f"Skipping invalid or hidden frame file: {frame_file}")
                continue  # Skip invalid frame files

            # Add frame file path and corresponding label
            valid_frame_count += 1
            # print(f"Adding frame file: {frame_file_path}, Label: {label_folder}")
            all_annotations.append((frame_file_path, label_folder))

        if valid_frame_count == 0:
            # print(f"No valid frame files found in label folder: {label_folder}")
            continue

    # Debug: Print collected annotations
    # print(f"Collected {len(all_annotations)} annotations.")

    # Shuffle and split into train and validation sets
    if all_annotations:
        random.shuffle(all_annotations)
        split_idx = int(len(all_annotations) * (1 - val_split))
        train_annotations = all_annotations[:split_idx]
        val_annotations = all_annotations[split_idx:]
    else:
        train_annotations = []
        val_annotations = []

    # Debug: Print train/val split info
    print(f"Train annotations: {len(train_annotations)}")
    print(f"Validation annotations: {len(val_annotations)}")

    # Write training annotations to CSV
    with open(train_csv, 'w', newline='') as train_file:
        writer = csv.writer(train_file)
        writer.writerow(['frame_folder', 'label'])  # Write header
        writer.writerows(train_annotations)

    # Write validation annotations to CSV
    with open(val_csv, 'w', newline='') as val_file:
        writer = csv.writer(val_file)
        writer.writerow(['frame_folder', 'label'])  # Write header
        writer.writerows(val_annotations)

    print("Annotation CSVs created successfully.")


create_train_val_annotations(processed_frames_path, train_csv, val_csv)


# Load annotations CSVs
train_annotations = pd.read_csv(train_csv)
val_annotations = pd.read_csv(val_csv)

print("Train Annotations:")
print(train_annotations.head())

print("Validation Annotations:")
print(val_annotations.head())

# Define the FrameDataset class
class FrameDataset(Dataset):
    def __init__(self, annotations_file, transform=None, num_frames=16, label_to_index=None, base_dir=None):
        """
        Dataset for loading video frames.

        Parameters:
        - annotations_file: Path to the CSV file with frame paths and labels.
        - transform: Transformations to apply to each frame.
        - num_frames: Number of frames per sequence.
        - label_to_index: Mapping of label names to indices.
        - base_dir: Base directory for frame paths (optional).
        """
        self.annotations = pd.read_csv(annotations_file)
        self.transform = transform
        self.num_frames = num_frames
        self.label_to_index = label_to_index
        self.base_dir = base_dir  # Base directory for resolving frame paths

    def __len__(self):
        return len(self.annotations) // self.num_frames  # Each sample is a sequence of frames

    def __getitem__(self, idx):
        # Get the starting index for the sequence
        start_idx = idx * self.num_frames
        sequence_frames = self.annotations.iloc[start_idx:start_idx + self.num_frames]

        frames = []

        for frame_file in sequence_frames['frame_folder']:
            # Resolve full path for the frame
            frame_path = frame_file
            if self.base_dir and not os.path.isabs(frame_file):
                frame_path = os.path.join(self.base_dir, frame_file)

            if not os.path.exists(frame_path):
                raise FileNotFoundError(f"Frame not found: {frame_path}")

            frame = Image.open(frame_path).convert("RGB")
            if self.transform:
                frame = self.transform(frame)
            frames.append(frame)

        # Handle cases with fewer frames than num_frames
        if len(frames) < self.num_frames:
            pad_frames = [torch.zeros_like(frames[0])] * (self.num_frames - len(frames))
            frames.extend(pad_frames)

        # Handle cases with more frames than num_frames
        if len(frames) > self.num_frames:
            frames = frames[:self.num_frames]

        # Stack frames into a single tensor
        frames_tensor = torch.stack(frames, dim=0)

        # Retrieve the label
        label = sequence_frames.iloc[0]['label']
        if self.label_to_index:
            label = self.label_to_index[label]

        return frames_tensor, torch.tensor(label, dtype=torch.long)


# Create a mapping from label names to indices
label_to_index = {label: idx for idx, label in enumerate(sorted(os.listdir(processed_frames_path)))}
print(f"Label to Index Mapping: {label_to_index}")


# Define transformation to match model requirements
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Match model's image size
    transforms.ToTensor(),          # Convert to tensor
])

# Create train and validation datasets
train_dataset = FrameDataset(
    annotations_file=train_csv,
    transform=transform,
    num_frames=16,  # Number of frames per sequence
    label_to_index=label_to_index
)

val_dataset = FrameDataset(
    annotations_file=val_csv,
    transform=transform,
    num_frames=16,
    label_to_index=label_to_index
)

# Print the lengths of your datasets to check for emptiness
print(f"Train Dataset Length: {len(train_dataset)}")
print(f"Validation Dataset Length: {len(val_dataset)}")

# Create DataLoaders

train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=8)
val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=8)

# Define the RViT model
class RViT(nn.Module):
    def __init__(self, num_classes, hidden_dim, num_layers, frame_dim):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Patch embedding with reduced stride and kernel size
        self.patch_embedding = nn.Conv3d(16, hidden_dim, kernel_size=(3, 8, 8), stride=(3, 4, 4), padding=(1, 2, 2))

        # Simplified learnable position encoding
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
        print("After patch embedding:", patches.shape)

        # Dynamically resize position encoding to match patches' spatial dimensions
        _, _, depth, height, width = patches.shape
        pos_encoding = F.interpolate(
            self.position_encoding, size=(depth, height, width), mode='trilinear', align_corners=False
        )  # Adjusted shape: [1, hidden_dim, depth, height, width]
        print("Position encoding shape after interpolation:", pos_encoding.shape)

        # Add position encoding
        patches += pos_encoding
        print("After adding position encoding:", patches.shape)

        # Initialize recurrent state
        h = torch.zeros_like(patches)
        print("Initialized recurrent state:", h.shape)

        # Pass through RViT units
        for i, unit in enumerate(self.rvit_units):
            h = unit(patches, h)  # Shape should be [batch_size, hidden_dim, depth, height, width]
            print(f"After RViT unit {i+1}:", h.shape)

        # Frame reconstruction
        reconstructed_frames = self.frame_reconstruction(h)
        reconstructed_frames = self.temporal_upsample(reconstructed_frames)

        # Classifier
        logits = self.classifier(h.mean(dim=(2, 3, 4)))  # Mean across spatial and temporal dimensions
        return logits, reconstructed_frames


# Define a simple RViTUnit for the model
class RViTUnit(nn.Module):
    def __init__(self, hidden_dim):
        super(RViTUnit, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

    def forward(self, x, h):
        # Attention layer
        attn_output, _ = self.attn(x.flatten(2).transpose(0, 1), x.flatten(2).transpose(0, 1), x.flatten(2).transpose(0, 1))
        attn_output = attn_output.transpose(0, 1).view_as(x)

        # Feedforward layer
        h = self.ffn(attn_output.mean(dim=(2, 3, 4)))
        return h

# Initialize model
model = RViT(num_classes=10, hidden_dim=512, num_layers=6, frame_dim=(3, 224, 224))
print("Model Initialized")
