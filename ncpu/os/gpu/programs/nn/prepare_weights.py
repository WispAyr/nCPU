#!/usr/bin/env python3
"""
Prepare MNIST Weights — Generate quantized weights for the GPU neural network.

This script either trains a simple MNIST model in PyTorch or downloads
pre-trained weights, quantizes them to Q8.8 fixed-point (int16), and
exports as C header constants.

Usage:
    python demos/nn/prepare_weights.py           # Generate with random init
    python demos/nn/prepare_weights.py --train    # Train on MNIST first
"""

import sys
import struct
import numpy as np
from pathlib import Path


def quantize_q88(weights: np.ndarray) -> np.ndarray:
    """Quantize float weights to Q8.8 fixed-point (int16)."""
    # Clip to [-127, 127] range, then scale by 256
    clipped = np.clip(weights, -127.0, 127.0)
    return (clipped * 256).astype(np.int16)


def generate_random_weights():
    """Generate Xavier-initialized random weights."""
    np.random.seed(42)

    # Layer 1: 784 → 128
    w1 = np.random.randn(128, 784).astype(np.float32) * np.sqrt(2.0 / 784)
    b1 = np.zeros(128, dtype=np.float32)

    # Layer 2: 128 → 10
    w2 = np.random.randn(10, 128).astype(np.float32) * np.sqrt(2.0 / 128)
    b2 = np.zeros(10, dtype=np.float32)

    return w1, b1, w2, b2


def try_train():
    """Try to train on MNIST with PyTorch, return weights."""
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torchvision import datasets, transforms

        print("Training MNIST model...")

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)

        model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        model.train()
        for epoch in range(3):
            total_loss = 0
            correct = 0
            total = 0
            for batch_idx, (data, target) in enumerate(train_loader):
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                pred = output.argmax(dim=1)
                correct += (pred == target).sum().item()
                total += target.size(0)
            acc = 100 * correct / total
            print(f"  Epoch {epoch+1}: loss={total_loss/len(train_loader):.4f}, acc={acc:.1f}%")

        # Extract weights
        w1 = model[1].weight.data.numpy()
        b1 = model[1].bias.data.numpy()
        w2 = model[3].weight.data.numpy()
        b2 = model[3].bias.data.numpy()

        print(f"  Final accuracy: {acc:.1f}%")
        return w1, b1, w2, b2

    except ImportError:
        print("PyTorch not available, using random weights")
        return None


def export_binary(w1, b1, w2, b2, output_path: str):
    """Export weights as raw binary file (int16 Q8.8)."""
    q_w1 = quantize_q88(w1)
    q_b1 = quantize_q88(b1)
    q_w2 = quantize_q88(w2)
    q_b2 = quantize_q88(b2)

    data = b""
    data += q_w1.tobytes()  # 128 * 784 * 2 = 200,704 bytes
    data += q_b1.tobytes()  # 128 * 2 = 256 bytes
    data += q_w2.tobytes()  # 10 * 128 * 2 = 2,560 bytes
    data += q_b2.tobytes()  # 10 * 2 = 20 bytes

    with open(output_path, "wb") as f:
        f.write(data)

    print(f"Exported {len(data):,} bytes to {output_path}")
    print(f"  w1: {q_w1.shape} ({q_w1.nbytes:,} bytes)")
    print(f"  b1: {q_b1.shape} ({q_b1.nbytes:,} bytes)")
    print(f"  w2: {q_w2.shape} ({q_w2.nbytes:,} bytes)")
    print(f"  b2: {q_b2.shape} ({q_b2.nbytes:,} bytes)")


def main():
    output_path = Path(__file__).parent / "mnist_weights.bin"

    if "--train" in sys.argv:
        result = try_train()
        if result:
            w1, b1, w2, b2 = result
        else:
            w1, b1, w2, b2 = generate_random_weights()
    else:
        print("Generating random Xavier-initialized weights...")
        print("Use --train to train on actual MNIST data")
        w1, b1, w2, b2 = generate_random_weights()

    export_binary(w1, b1, w2, b2, str(output_path))


if __name__ == "__main__":
    main()
