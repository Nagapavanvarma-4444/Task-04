import torch
import torch.nn as nn
import cv2
import numpy as np
from torchvision import transforms
from sklearn.preprocessing import LabelEncoder
import os

# Constants
IMG_SIZE = (64, 64)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_PATH = r"E:\PredictiveInventorySystem\gesture_images\05_thumb\00_frame_00_05_0001.png"
MODEL_PATH = "gesture_model.pth"
DATASET_DIR = 'gesture_images'  # Same as training path

# Transformation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load class names from folder structure and encode
class_names = [d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))]
encoder = LabelEncoder()
encoder.fit(class_names)

# Define model structure
class GestureCNN(nn.Module):
    def _init_(self, num_classes):
        super(GestureCNN, self)._init_()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.model(x)

# Load model
model = GestureCNN(num_classes=len(class_names)).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# Load and preprocess image
img = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, IMG_SIZE)
img_tensor = transform(img).unsqueeze(0).to(DEVICE)  # Add batch dimension

# Predict
with torch.no_grad():
    output = model(img_tensor)
    predicted = torch.argmax(output, dim=1).item()

# Get gesture name
gesture_name = encoder.inverse_transform([predicted])[0]
print(f"Predicted gesture: {gesture_name}")
