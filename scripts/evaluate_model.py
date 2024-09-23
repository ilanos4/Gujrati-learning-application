import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from train_model import SimpleCNN  # Import the model class
from PIL import Image

# Load the trained model
model = SimpleCNN()
model.load_state_dict(torch.load('trained_model.pth'))  # Ensure the correct path
model.eval()  # Set the model to evaluation mode

# Define transforms for evaluation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((32, 32)),  # Ensure the image size matches
])

# Custom dataset class for loading images without class subfolders
class CustomImageDataset(torch.utils.data.Dataset):
    def __init__(self, directory, transform=None):
        self.directory = directory
        self.transform = transform
        self.images = [f for f in os.listdir(directory) if f.endswith(('.jpg', '.png'))]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.directory, self.images[idx])
        image = Image.open(img_path).convert('L')  # Convert to grayscale
        if self.transform:
            image = self.transform(image)
        return image, 0  # Returning a dummy label

# Load your evaluation dataset
eval_dataset = CustomImageDataset('data/Gujrati-learning-application/gujarati_character_recognition/data/preprocessed', transform=transform)
eval_loader = DataLoader(eval_dataset, batch_size=32, shuffle=False)

# Evaluate the model
correct = 0
total = 0
with torch.no_grad():
    for images, _ in eval_loader:  # Ignoring dummy labels
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        # Here you need to implement logic to compare predicted with actual labels
        # For now, just counting the total images processed
        total += images.size(0)
        # correct += (predicted == actual_labels).sum().item()  # Implement actual labels as needed

print(f'Total Images Processed: {total}')
