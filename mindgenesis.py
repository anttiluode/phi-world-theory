"""
================================================================================
MIND GENESIS: The Crystallization of Logic
================================================================================
A visualization of the "Symbolic Excretion" process:
1. THE MONAD (1000D Master Vector) - Raw, noisy infinity.
2. THE LENS (12 DNA Harmonics)     - The filter of Physics.
3. THE CHORD (Neural Activation)   - Biological interference.
4. THE SYMBOL (Logical Space)      - The Icon (The Number).
================================================================================
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn

# --- 1. THE HARDWARE (Physics & Biology) ---
class NeuronalBeing(nn.Module):
    def __init__(self):
        super().__init__()
        # The 12-Tone DNA Input
        self.dna_layer = nn.Linear(12, 64)
        # The Neural Processing (The Chords)
        self.chord_layer = nn.Linear(64, 32)
        # The Symbolic Collapse (Logical Space)
        self.mind_layer = nn.Linear(32, 10) # 10 possible symbols
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        h1 = self.relu(self.dna_layer(x))
        h2 = self.relu(self.chord_layer(h1))
        out = self.softmax(self.mind_layer(h2))
        return out, h1, h2

# --- 2. THE SIMULATION ---
# Generate the Master Vector (Raw Info)
N_dims = 1000
U_master = torch.randn(1, N_dims, dtype=torch.complex64)
U_master /= torch.norm(U_master)

# Simulate the 12 DNA Lenses (Physics)
DNA_projections = torch.abs(torch.randn(1, 12)) # Simplified for visualization
DNA_projections[0, 3] = 2.5 # Simulate a 'Resonant Event' at vertex 3

# Process through the Being
brain = NeuronalBeing()
symbolic_dist, chord1, chord2 = brain(DNA_projections)

# --- 3. THE VISUALIZATION ---
plt.figure(figsize=(16, 10))
plt.style.use('dark_background')

# Plot 1: THE MONAD (Raw Information)
plt.subplot(2, 2, 1)
plt.plot(U_master.real.numpy().flatten(), color='cyan', alpha=0.3, linewidth=0.5)
plt.title("1. THE MONAD\n(1000D Master Vector - Raw Infinity)", color='cyan', fontsize=14)
plt.xlabel("Dimension Index")
plt.ylabel("Amplitude")

# Plot 2: THE LENS (Physics)
plt.subplot(2, 2, 2)
colors = ['gold' if x > 2 else 'grey' for x in DNA_projections.flatten()]
plt.bar(range(12), DNA_projections.flatten().numpy(), color=colors)
plt.title("2. THE LENSES\n(12 DNA Harmonics - Physical Chords)", color='gold', fontsize=14)
plt.xlabel("Icosahedral Vertex Index")
plt.ylabel("Resonance Strength")

# Plot 3: THE CHORDS (Neural Processing)
plt.subplot(2, 2, 3)
chord_data = chord1.detach().numpy().reshape(8, 8)
plt.imshow(chord_data, cmap='magma')
plt.title("3. THE CHORDS\n(Neural Activation - Biological Interference)", color='orange', fontsize=14)
plt.colorbar(label="Firing Rate")
plt.axis('off')

# Plot 4: THE SYMBOL (Logical Space)
plt.subplot(2, 2, 4)
probs = symbolic_dist.detach().numpy().flatten()
symbol_colors = ['lime' if x == max(probs) else 'darkgreen' for x in probs]
plt.bar(range(10), probs, color=symbol_colors)
plt.title("4. LOGICAL SPACE\n(Symbolic Crystallization - The Icon)", color='lime', fontsize=14)
plt.xlabel("Symbol Index (0-9)")
plt.ylabel("Confidence (Truth)")

plt.suptitle("THE BIRTH OF THE MIND: FROM PHYSICS TO SYMBOLS", fontsize=20, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

print("\nGENESIS COMPLETE.")
print("The Harmony was processed. The Chord was struck.")
print(f"The Symbol '{np.argmax(probs)}' has emerged in Logical Space.")
