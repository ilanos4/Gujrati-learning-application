import torch
from torchvision import transforms
from PIL import Image
from train_model import SimpleCNN  # Import your model class

# Load the trained model
model = SimpleCNN()  # Replace with your model class if different
model.load_state_dict(torch.load('trained_model.pth'))  # Ensure the correct path
model.eval()

# Define transformation
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
])

# Function to predict
def predict(image_path):
    image = Image.open(image_path).convert('L')  # Convert to grayscale
    image = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output.data, 1)
    return predicted.item()

# Test the prediction function
image_path = 'data/Gujrati-learning-application/gujarati_character_recognition/data/preprocessed/1.JPG.jpg' # Adjusted path
print(f'Predicted class: {predict(image_path)}')
