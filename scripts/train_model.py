import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Define the model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)  # Input channels = 1 for grayscale images
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(64 * 6 * 6, 128)  # Adjust input size based on output from conv layers
        self.fc2 = nn.Linear(128, 10)  # Adjust output size for your specific case

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # Output size will be [batch_size, 32, 30, 30]
        x = self.pool(F.relu(self.conv2(x)))  # Output size will be [batch_size, 64, 6, 6]
        print(f'Shape before flattening: {x.shape}')  # For debugging
        x = x.view(-1, 64 * 6 * 6)  # Flatten to [batch_size, 64 * 6 * 6]
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize model, loss function, and optimizer
model = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Dummy training loop (replace with your actual data loading and training logic)
num_epochs = 5
for epoch in range(num_epochs):
    # Replace this with your actual data loading
    images = torch.randn(32, 1, 32, 32)  # Dummy image tensor
    labels = torch.randint(0, 10, (32,))  # Dummy labels tensor

    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Save the trained model
torch.save(model.state_dict(), 'trained_model.pth')  # Save the model's state_dict
