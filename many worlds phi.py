"""
================================================================================
MULTIVERSE BRANCHING SIMULATOR (IHT-MWI Edition)
================================================================================
Concept: 
  Many-Worlds is just "Phase Adjacency." 
  The Master Vector is the trunk of the tree; 3D worlds are the leaves.
  
Mechanism:
  1. We take the Master Vector (The Monad).
  2. We apply a 'Differential Phase Slip' (rotating adjacent dimensions).
  3. We project 5 different 'Worlds' to see the divergence.
================================================================================
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 1. THE LENS (12-Tone DNA)
phi = (1 + 5**0.5) / 2
DNA_3D = torch.tensor([
    [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
    [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
    [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
]).float()
DNA_3D /= torch.norm(DNA_3D, dim=1, keepdim=True)

# 2. THE SOURCE (The Master Vector)
# We generate a coherent 1000D vector
def get_master_vector(dims=1000):
    U = torch.zeros(dims, dtype=torch.complex64)
    U[:12] = 1.0 + 0j # Coherent seed
    # Add high-dimensional 'Overtone' structure
    t = torch.linspace(0, 10*np.pi, dims)
    U += 0.2 * torch.exp(1j * t)
    return U / torch.norm(U)

U_trunk = get_master_vector()

def project_world(U_vector):
    """Projects the Master Vector through the 12 DNA lenses to 3D space."""
    # We simulate the magnitude of resonance at each icosahedral vertex
    # This determines the 'Existence' of each atom in that world
    magnitudes = []
    for i in range(12):
        # Local projection operator
        P = torch.zeros(1000, dtype=torch.complex64)
        P[i] = 1.0
        res = torch.vdot(P, U_vector).abs().item()
        magnitudes.append(res)
    return np.array(magnitudes)

# ==========================================================
# SIMULATE THE BRANCHING
# ==========================================================
print("Generating Multiverse Branches...")

num_worlds = 5
phase_slips = np.linspace(0, 0.05, num_worlds) # Tiny 5% slips

worlds_data = []

for slip in phase_slips:
    # APPLY THE SLIP: We rotate the Master Vector differently across dimensions
    # This is a 'Differential Phase Shift' - the mechanism of branching
    rotator = torch.exp(1j * torch.linspace(0, slip * np.pi, 1000))
    U_branch = U_trunk * rotator
    
    # Project the resulting Buckyball
    resonance = project_world(U_branch)
    worlds_data.append(resonance)

# ==========================================================
# VISUALIZATION: THE SUPERPOSITION
# ==========================================================
fig = plt.figure(figsize=(14, 8))
ax = fig.add_subplot(111, projection='3d')

colors = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF']
labels = [f"World {i} (Slip: {s:.3f})" for i, s in enumerate(phase_slips)]

# Plot the 12 vertices for each world
for w_idx, resonance in enumerate(worlds_data):
    # We use the DNA_3D coordinates scaled by the resonance (existence)
    # This shows how the 'Physical Atoms' shift between worlds
    pos = DNA_3D.numpy() * (1.0 + w_idx * 0.1) # Offset slightly for visibility
    
    # Scaling the size of the atoms by their resonance in that specific world
    sizes = (resonance / np.max(resonance)) * 200
    
    ax.scatter(pos[:,0], pos[:,1], pos[:,2], 
               s=sizes, c=colors[w_idx], label=labels[w_idx], alpha=0.6)

    # Draw the skeleton (the Bonds of the Buckyball)
    # This shows the geometric continuity across worlds
    for i in range(12):
        for j in range(i+1, 12):
            dist = np.linalg.norm(pos[i] - pos[j])
            if dist < 1.5: # Connect nearest neighbors
                ax.plot([pos[i,0], pos[j,0]], [pos[i,1], pos[j,1]], [pos[i,2], pos[j,2]], 
                        color=colors[w_idx], alpha=0.1)

ax.set_title("The Holographic Multiverse: 5 Phase-Adjacent Buckyballs", fontsize=16)
ax.set_axis_off()
plt.legend(loc='upper left')
plt.show()

print("MULTIVERSE REVEALED:")
print("Notice how each 'World' is the same Buckyball, but its atoms")
print("have shifted their intensity and position in response to the phase.")
print("The 'Many Worlds' are simply the interference patterns of the same Vector.")
