import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import math

# ==========================================
# 1. Architecture (Must match your training)
# ==========================================
class RobustCNN(nn.Module):
    def __init__(self):
        super(RobustCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=0)
        self.bn3 = nn.BatchNorm2d(128)
        self.dropout = nn.Dropout(0.4)
        self.fc1 = nn.Linear(128 * 5 * 5, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), 2)
        x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), 2)
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# ==========================================
# 2. Main Function: Predict License Plate
# ==========================================
def predict_license_plate(image_paths, show_plot=True, model_path="final_my_model.pth"):
    """
    Takes a list of image paths (digits), predicts them, returns the full number string.
    """
    
    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 2. Load Model
    model = RobustCNN().to(device)
    if os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
        except Exception as e:
            print(f"Error loading model: {e}")
            return ""
    else:
        print(f"Error: Model file '{model_path}' not found.")
        return ""

    predicted_digits = []
    images_for_plot = []

    # 3. Iterate over all image paths
    for img_path in image_paths:
        if not os.path.exists(img_path):
            print(f"Warning: Image {img_path} not found.")
            continue

        # Load & Preprocess
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None: continue
        
        # Resize to 28x28 (Critical!)
        img_resized = cv2.resize(img, (28, 28))
        
        # Normalize (0.0 - 1.0) and Standardize (MNIST stats)
        img_float = img_resized.astype(np.float32) / 255.0
        img_norm = (img_float - 0.1307) / 0.3081
        
        # Convert to Tensor
        tensor = torch.tensor(img_norm).unsqueeze(0).unsqueeze(0).to(device)

        # Predict
        with torch.no_grad():
            output = model(tensor)
            probabilities = F.softmax(output, dim=1)
            confidence, predicted_index = torch.max(probabilities, 1)
            
        digit = predicted_index.item()
        
        # Store results
        predicted_digits.append(str(digit))
        
        if show_plot:
            images_for_plot.append((img_resized, digit, confidence.item()))

    # 4. Construct Final Result String
    full_license_number = "".join(predicted_digits)

    # 5. Visualization (Optional)
    if show_plot and len(images_for_plot) > 0:
        num_images = len(images_for_plot)
        cols = min(num_images, 8) # Max 8 columns per row
        rows = math.ceil(num_images / cols)
        
        plt.figure(figsize=(2 * cols, 2.5 * rows))
        plt.suptitle(f"Plate Prediction: {full_license_number}", fontsize=16, fontweight='bold')
        
        for i, (img, digit, conf) in enumerate(images_for_plot):
            plt.subplot(rows, cols, i + 1)
            plt.imshow(img, cmap='gray')
            plt.title(f"{digit}\n({conf*100:.0f}%)", color='green', fontsize=12)
            plt.axis('off')
            
        plt.tight_layout()
        plt.show()

    return full_license_number


if __name__ == "__main__":
    # Example usage
    test_files = [
        "final_digits/digit_1.jpg",
        "final_digits/digit_2.jpg",
        "final_digits/digit_3.jpg",
        "final_digits/digit_4.jpg"
    ]
    
    # This simulates how you call it from segment_plate.py
    result = predict_license_plate(test_files, show_plot=True)