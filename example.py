import torch
from torch import nn
import torch.autograd.profiler as profiler
import torch.nn.functional as F

# Define the model
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        # Define layers
        self.fc1 = nn.Linear(28 * 28, 128)  # Input layer (28x28 pixels flattened)
        self.fc2 = nn.Linear(128, 64)       # Hidden layer
        self.fc3 = nn.Linear(64, 10)        # Output layer (10 classes for MNIST)

    def forward(self, x):
        # Flatten the input image
        x = x.view(-1, 28 * 28)  # Flatten the 28x28 image into a 1D vector of size 784
        
        # Pass through the layers
        x = F.relu(self.fc1(x))  # Apply ReLU activation after the first layer
        x = F.relu(self.fc2(x))  # Apply ReLU activation after the second layer
        x = self.fc3(x)          # Output layer (no activation function)
        
        return x

# Create the model instance
model = SimpleNN()
input = torch.rand(1, 28, 28)

# warm-up
model(input)

with profiler.profile(profile_memory=True) as prof:
    with torch.no_grad():
        out = model(input)

x = max([data.cpu_memory_usage for data in prof.key_averages()])
print(x)