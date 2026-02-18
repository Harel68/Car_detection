import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import os
from pathlib import Path

class DotDetectorDataset(Dataset):
    """
    Dataset for loading license plate images with dot annotations
    FULL SIZE 640x160 - NO RESIZE
    """
    def __init__(self, images_dir, labels_dir, augment=True):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.augment = augment
        
        # Get all image files
        self.image_files = list(self.images_dir.glob('*.jpg')) + \
                          list(self.images_dir.glob('*.png'))
        
        print(f"Found {len(self.image_files)} images in {images_dir}")
        if self.augment:
            print(f"Data augmentation enabled - effective dataset size: {len(self.image_files) * 3}")
        
    def __len__(self):
        if self.augment:
            return len(self.image_files) * 3
        return len(self.image_files)
    
    def apply_augmentation(self, image, boxes):
        """Apply random augmentations"""
        h, w = image.shape[:2]
        aug_type = np.random.randint(0, 3)
        
        if aug_type == 0:
            # Brightness
            factor = np.random.uniform(0.7, 1.3)
            image = np.clip(image * factor, 0, 255).astype(np.uint8)
        elif aug_type == 1:
            # Noise
            noise = np.random.normal(0, 10, image.shape).astype(np.uint8)
            image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        elif aug_type == 2:
            # Rotation
            angle = np.random.uniform(-3, 3)
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            image = cv2.warpAffine(image, M, (w, h))
            new_boxes = []
            for box in boxes:
                # box format: [cls, x_c, y_c, bw, bh]
                cls, x_c, y_c, bw, bh = box
                point = np.array([x_c, y_c, 1])
                rotated = M @ point
                new_boxes.append([cls, rotated[0], rotated[1], bw, bh])
            boxes = new_boxes
        
        return image, boxes
    
    def __getitem__(self, idx):
        actual_idx = idx % len(self.image_files)
        img_path = self.image_files[actual_idx]
        image = cv2.imread(str(img_path))
        
        if image is None:
            raise ValueError(f"Could not load image: {img_path}")
        
        h, w = image.shape[:2]
        label_path = self.labels_dir / (img_path.stem + '.txt')
        
        boxes = []
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cls, x_c, y_c, bw, bh = map(float, parts)
                        x_center = x_c * w
                        y_center = y_c * h
                        box_w = bw * w
                        box_h = bh * h
                        boxes.append([int(cls), x_center, y_center, box_w, box_h])
        
        if self.augment and idx >= len(self.image_files):
            image, boxes = self.apply_augmentation(image, boxes)
        
        if image.shape[:2] != (160, 640):
            image = cv2.resize(image, (640, 160))
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = gray.astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(gray).unsqueeze(0)
        
        # Create 2-Channel Heatmap
        heatmap = self.create_heatmap(boxes, w, h, 640, 160)
        
        return image_tensor, heatmap
    
    def create_heatmap(self, boxes, orig_w, orig_h, target_w, target_h):
        # Shape is (2, H, W) -> Channel 0 for Dots, Channel 1 for Limits
        heatmap = np.zeros((2, target_h, target_w), dtype=np.float32)
        
        scale_x = target_w / orig_w
        scale_y = target_h / orig_h
        
        for box in boxes:
            cls, x_c, y_c, _, _ = box
            
            if cls not in [0, 1]:
                continue
                
            x = int(x_c * scale_x)
            y = int(y_c * scale_y)
            
            sigma = 8
            # Draw on the specific channel for this class
            for i in range(max(0, y-sigma*3), min(target_h, y+sigma*3+1)):
                for j in range(max(0, x-sigma*3), min(target_w, x+sigma*3+1)):
                    dist = np.sqrt((i-y)**2 + (j-x)**2)
                    heatmap[int(cls), i, j] = max(heatmap[int(cls), i, j], 
                                                 np.exp(-(dist**2) / (2*sigma**2)))
        
        return torch.from_numpy(heatmap)


class DotDetectorCNN(nn.Module):
    """
    CNN for 640x160 plates
    Output: 2 Channels (0: Dots, 1: Limits)
    """
    def __init__(self):
        super(DotDetectorCNN, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            
            # Final output: 2 channels
            nn.Conv2d(16, 2, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def train_model(train_loader, val_loader, num_epochs=200, learning_rate=0.001, device='cpu'):
    
    
    model = DotDetectorCNN().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    
    best_val_loss = float('inf')
    
    print(f"\nTraining on {device} (2-Class Output)")
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device)
                targets = targets.to(device)
                outputs = model(images)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        
        print(f"Epoch [{epoch+1}/{num_epochs}] Train: {train_loss:.6f} | Val: {val_loss:.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
            }, 'best_dot_detector.pth')

    print("Training Complete.")
    return model

def main():
    IMAGES_DIR = 'dataset/images/train'
    LABELS_DIR = 'dataset/labels/train'
    
    if not os.path.exists(IMAGES_DIR):
        print("IMAGES_DIR in main() not set properly")
        return

    # Check for device availability inside main
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Detected device: {device}")

    full_dataset = DotDetectorDataset(IMAGES_DIR, LABELS_DIR, augment=True)
    
    train_size = int(0.85 * len(full_dataset.image_files))
    indices = list(range(len(full_dataset.image_files)))
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    train_indices_aug = train_indices + \
                       [i + len(full_dataset.image_files) for i in train_indices] + \
                       [i + 2*len(full_dataset.image_files) for i in train_indices]

    train_dataset = torch.utils.data.Subset(full_dataset, train_indices_aug)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    
    # FIXED: Explicitly passing the 'device' variable here
    train_model(train_loader, val_loader, num_epochs=100, device=device)

if __name__ == "__main__":
    main()