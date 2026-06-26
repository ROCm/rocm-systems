#!/usr/bin/env python3
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
simple_model.py - Basic PyTorch model for rocjitsu debugging

Demonstrates:
- Simple neural network definition
- Training loop with GPU
- Loss monitoring
- Basic error checking
"""

import torch
import torch.nn as nn
import torch.optim as optim
import sys

class SimpleNet(nn.Module):
    """Simple feedforward neural network for MNIST-like data"""

    def __init__(self, input_size=784, hidden1=128, hidden2=64, num_classes=10):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Flatten input
        x = x.view(x.size(0), -1)

        # Forward pass
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train_model(epochs=5, batch_size=32, lr=0.01, device='cuda'):
    """Train a simple model and monitor progress"""

    print("PyTorch Simple Model Training")
    print(f"Device: {device}")

    # Check if CUDA is available
    if device == 'cuda' and not torch.cuda.is_available():
        print("WARNING: CUDA not available, using CPU")
        device = 'cpu'

    # Create model
    model = SimpleNet().to(device)
    print(f"Model: SimpleNet (784 -> 128 -> 64 -> 10)")
    print()

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    # Create dummy data (simulating MNIST)
    # In real usage, you would use a DataLoader with actual data
    x = torch.randn(batch_size, 1, 28, 28, device=device)
    y = torch.randint(0, 10, (batch_size,), device=device)

    # Training loop
    print(f"Training for {epochs} epochs...")
    print()

    for epoch in range(epochs):
        # Forward pass
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)

        # Backward pass
        loss.backward()

        # Check for NaN/Inf in gradients
        has_nan = False
        has_inf = False
        for name, param in model.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any():
                    has_nan = True
                    print(f"  WARNING: NaN gradient in {name}")
                if torch.isinf(param.grad).any():
                    has_inf = True
                    print(f"  WARNING: Inf gradient in {name}")

        # Update weights
        optimizer.step()

        # Calculate accuracy
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == y).sum().item()
        accuracy = 100.0 * correct / batch_size

        # Print progress
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"  Loss: {loss.item():.4f}")
        print(f"  Accuracy: {accuracy:.1f}%")

        if has_nan or has_inf:
            print("  ERROR: Numerical instability detected!")
            return False

        # Check if loss is NaN
        if torch.isnan(loss):
            print("  ERROR: Loss became NaN!")
            return False

        print()

    print("Training completed successfully!")

    # Test inference
    with torch.no_grad():
        test_x = torch.randn(10, 1, 28, 28, device=device)
        test_outputs = model(test_x)
        _, test_predicted = torch.max(test_outputs, 1)
        print(f"Inference test: Predicted classes: {test_predicted.cpu().numpy()}")

    return True

def main():
    """Main entry point"""

    # Parse command line arguments
    epochs = 5
    if len(sys.argv) > 1:
        epochs = int(sys.argv[1])

    # Run training
    success = train_model(epochs=epochs)

    # Exit with appropriate code
    return 0 if success else 1

if __name__ == '__main__':
    sys.exit(main())