import torch.nn as nn
import torch.nn.functional as F

class ASLClassifier(nn.Module):
    def __init__(self, num_classes=30):
        super(ASLClassifier, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)  # Input: [3, 200, 200]
        self.pool = nn.MaxPool2d(2, 2)  # Halves H and W
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        # After 3 poolings: 200 → 100 → 50 → 25
        self.fc1 = nn.Linear(64 * 25 * 25, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # [16, 100, 100]
        x = self.pool(F.relu(self.conv2(x)))  # [32, 50, 50]
        x = self.pool(F.relu(self.conv3(x)))  # [64, 25, 25]
        
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # No softmax here; use CrossEntropyLoss

        return x
