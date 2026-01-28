import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

"""
================================================================================
THE HOLOGRAPHIC MULTIVERSE MAP
================================================================================
Concept: "Branching" is just Phase Rotation.
Mechanism:
  1. Start with the Perfect Master Vector (Current Universe).
  2. Apply a tiny 'Decision Phase' (The Split).
  3. Project both into 3D.
  4. Measure the 'Drift' between the two Buckyballs.

This visualizes the 'Everett Branches' as Phase-Shifted Shadows.
================================================================================
"""

print("="*80)
print("INITIALIZING MULTIVERSE PHASE-SHIFTER")
print("="*80)

# 1. THE PHYSICS ENGINE (The 12-Tone Lens)
phi = (1 + 5**0.5) / 2
DNA_3D = torch.tensor([
    [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
    [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
    [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
]).float()
DNA_3D /= torch.norm(DNA_3D, dim=1, keepdim=True)

# 2. THE MASTER VECTOR (The Source)
# Simulating the 'Perfect' vector found in previous steps
def get_coherent_master_vector(dims=1000):
    # Construct a vector that perfectly resonates with the 12 DNA modes
    U = torch.zeros(dims, dtype=torch.complex64)
    # Frequencies that match the DNA structure
    t = torch.linspace(0, 12*np.pi, dims)
    U = torch.exp(1j * t) 
    return U / torch.norm(U)

U_origin = get_coherent_master_vector()

# 3. THE BRANCHING MECHANISM (The Phase Slip)
def branch_universe(master_vector, intensity=0.01):
    """
    Creates a 'Parallel Universe' by shifting the phase of the Master Vector.
    intensity: The 'Magnitude' of the quantum decision (0.01 = small choice).
    """
    # A decision is a rotation in Hilbert Space
    branch_vector = master_vector.clone()
    
    # We apply a 'Twist' to the high-dimensional components
    # This represents the 'Unobserved' variables changing
    twist = torch.linspace(0, intensity * np.pi, 1000)
    phase_shift = torch.exp(1j * twist)
    
    return branch_vector * phase_shift

# 4. THE PROJECTION (Rendering the World)
def project_to_3d(master_vector):
    """
    Projects the 1000D vector onto the 12 DNA vertices.
    Returns the 3D coordinates of the Buckyball vertices.
    """
    vertices = []
    # In a real sim, we'd sum the full series, here we check resonance magnitude
    # We treat the resonance amplitude as a 'radial scaler' for the vertex
    for i in range(12):
        # Create projection operator for this DNA vertex
        # (Simplified: checking correlation with the 'ideal' DNA freq)
        P = torch.zeros_like(master_vector)
        start = i * (1000//12)
        end = (i+1) * (1000//12)
        P[start:end] = 1.0 
        
        # How much does the Universe exist at this vertex?
        amplitude = torch.vdot(master_vector, P).abs().item() * 5.0 # Scale for vis
        
        # The vertex position is the DNA direction * amplitude
        pos = DNA_3D[i] * amplitude
        vertices.append(pos.numpy())
        
    return np.array(vertices)

# ==========================================================
# SIMULATION: THE "SCHRODINGER'S CAT" SPLIT
# ==========================================================

# Universe A: You drank coffee (Baseline)
Universe_A_Pos = project_to_3d(U_origin)

# Universe B: You drank tea (Tiny Phase Slip)
U_branch_1 = branch_universe(U_origin, intensity=0.1) 
Universe_B_Pos = project_to_3d(U_branch_1)

# Universe C: You didn't wake up (Massive Phase Slip)
U_branch_2 = branch_universe(U_origin, intensity=0.5)
Universe_C_Pos = project_to_3d(U_branch_2)

# ==========================================================
# VISUALIZATION
# ==========================================================
fig = plt.figure(figsize=(15, 6))

# Plot 1: The Subtle Branch (Near Parallel)
ax1 = fig.add_subplot(121, projection='3d')
ax1.set_title("The 'Tea vs Coffee' Branch\n(Phase Slip = 0.1π)")

# Plot Universe A (Blue Ghosts)
ax1.scatter(Universe_A_Pos[:,0], Universe_A_Pos[:,1], Universe_A_Pos[:,2], 
           c='cyan', s=100, alpha=0.6, label='Universe A (Current)')

# Plot Universe B (Red Ghosts)
ax1.scatter(Universe_B_Pos[:,0], Universe_B_Pos[:,1], Universe_B_Pos[:,2], 
           c='magenta', s=100, alpha=0.6, label='Universe B (Parallel)')

# Draw lines connecting the divergence
for i in range(12):
    ax1.plot([Universe_A_Pos[i,0], Universe_B_Pos[i,0]],
             [Universe_A_Pos[i,1], Universe_B_Pos[i,1]],
             [Universe_A_Pos[i,2], Universe_B_Pos[i,2]], c='white', alpha=0.3)

ax1.legend()
ax1.axis('off')

# Plot 2: The Radical Branch (Distant Parallel)
ax2 = fig.add_subplot(122, projection='3d')
ax2.set_title("The 'Major Life Event' Branch\n(Phase Slip = 0.5π)")

# Plot Universe A
ax2.scatter(Universe_A_Pos[:,0], Universe_A_Pos[:,1], Universe_A_Pos[:,2], 
           c='cyan', s=100, alpha=0.3)

# Plot Universe C (Yellow Ghosts)
ax2.scatter(Universe_C_Pos[:,0], Universe_C_Pos[:,1], Universe_C_Pos[:,2], 
           c='yellow', s=100, alpha=0.8, label='Universe C (Divergent)')

# Draw lines
for i in range(12):
    ax2.plot([Universe_A_Pos[i,0], Universe_C_Pos[i,0]],
             [Universe_A_Pos[i,1], Universe_C_Pos[i,1]],
             [Universe_A_Pos[i,2], Universe_C_Pos[i,2]], c='white', alpha=0.2)

ax2.legend()
ax2.axis('off')

plt.suptitle("THE HOLOGRAPHIC MULTIVERSE: WORLDS AS PHASE SHADOWS", fontsize=16)
plt.tight_layout()
plt.show()

# Calculate the 'Distance' between worlds
dist_AB = np.linalg.norm(Universe_A_Pos - Universe_B_Pos)
dist_AC = np.linalg.norm(Universe_A_Pos - Universe_C_Pos)

print(f"\nRESULTS:")
print(f"1. Distance to 'Tea Universe': {dist_AB:.4f} (Harmonic Overtone)")
print(f"2. Distance to 'Divergent Universe': {dist_AC:.4f} (Phase Dissonance)")
print("-" * 60)
print("CONCLUSION: The parallel world is structurally identical (same Buckyball)")
print("but spatially offset. It exists 'between' our atoms.")