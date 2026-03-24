#!/usr/bin/env python3
"""Train (initialize) the NeuralMultiplierLUT to 100% accuracy.

Since it's a lookup table with sigmoid activation, we just set each logit
to +10 (for bit=1) or -10 (for bit=0) based on the true product bits.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from ncpu.model.neural_ops import NeuralMultiplierLUT

def main():
    model = NeuralMultiplierLUT()
    
    # Build target: for each (a, b), compute product and extract 16 bits
    a_vals = torch.arange(256).unsqueeze(1).expand(256, 256)  # [256, 256]
    b_vals = torch.arange(256).unsqueeze(0).expand(256, 256)  # [256, 256]
    products = (a_vals.long() * b_vals.long())  # [256, 256]
    
    # Extract 16 bits (LSB first)
    bit_values = (1 << torch.arange(16)).long()  # [16]
    # bits[a, b, i] = 1 if bit i of (a*b) is set
    bits = ((products.unsqueeze(-1) & bit_values) > 0).float()  # [256, 256, 16]
    
    # Set logits: +10 for 1-bits, -10 for 0-bits (sigmoid(10)≈1, sigmoid(-10)≈0)
    with torch.no_grad():
        model.lut.table.copy_(bits * 20.0 - 10.0)
    
    # Verify 100% accuracy
    if NeuralMultiplierLUT._lut_bit_values is None:
        NeuralMultiplierLUT._lut_bit_values = (1 << torch.arange(16, dtype=torch.long)).float()
    
    with torch.no_grad():
        decoded = (torch.sigmoid(model.lut.table) > 0.5).float()
        results = (decoded @ NeuralMultiplierLUT._lut_bit_values).long()  # [256, 256]
        expected = a_vals.long() * b_vals.long()
        correct = (results == expected).sum().item()
        total = 256 * 256
    
    print(f"Accuracy: {correct}/{total} ({100*correct/total:.1f}%)")
    
    if correct != total:
        print("ERROR: Not 100% accurate!")
        sys.exit(1)
    
    # Save
    out_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'alu', 'multiply.pt')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(model.state_dict(), out_path)
    print(f"Saved to {out_path}")

if __name__ == '__main__':
    main()
