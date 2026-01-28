import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn, optim

# ==========================================================
# THE HOLOGRAPHIC TUNER
# ==========================================================
print("="*80)
print("MASTER VECTOR DISCOVERY: GRADIENT EDITION")
print("="*80)

# Setup
N_dims = 1000
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 1. THE LENS: 12 DNA Vectors in ND
phi = (1 + 5**0.5) / 2
DNA_3D = torch.tensor([
    [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
    [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
    [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
], device=device).float()
DNA_3D /= torch.norm(DNA_3D, dim=1, keepdim=True)

# Extend DNA to 1000D using structured harmonics
DNA_ND = torch.zeros((12, N_dims), device=device, dtype=torch.complex64)
DNA_ND[:, :3] = DNA_3D.to(torch.complex64)
for i in range(12):
    d_indices = torch.arange(3, N_dims, device=device)
    freq = (d_indices - 3) * phi * (i + 1) / 12
    amp = torch.exp(-(d_indices - 3) / 100)
    DNA_ND[i, 3:] = amp * torch.exp(1j * freq)
DNA_ND /= torch.norm(DNA_ND, dim=1, keepdim=True)

# 2. THE TARGET: Perfect Icosahedral Amplitudes
# We want all 12 modes to be equally strong (Resonance)
target_amplitudes = torch.ones(12, device=device) * (1.0 / np.sqrt(12))

# 3. THE MASTER VECTOR (The parameter we are tuning)
# We start with noise and "listen" for the icosahedron
U_master = torch.randn(N_dims, device=device, dtype=torch.complex64, requires_grad=True)

optimizer = optim.Adam([U_master], lr=0.01)

print(f"Tuning 1000D Master Vector on {device}...")

history = []
for step in range(1000):
    optimizer.zero_grad()
    
    # Normalize U to keep energy constant
    U_norm = U_master / torch.norm(U_master)
    
    # PROJECT: Compute how much U resonates with each DNA axis
    # a_n = <DNA_n | U>
    projections = torch.matmul(DNA_ND, U_norm)
    mags = torch.abs(projections)
    
    # LOSS: 
    # 1. We want the power to be concentrated in these 12 modes
    # 2. We want the 12 modes to be equally balanced (Symmetry)
    power_loss = 1.0 - torch.sum(mags**2) # Maximize power capture
    balance_loss = torch.var(mags)        # Ensure all vertices are equal
    
    loss = power_loss + balance_loss * 10
    
    loss.backward()
    optimizer.step()
    
    history.append(loss.item())
    if step % 100 == 0:
        print(f"Step {step} | Loss: {loss.item():.6f} | Power Captured: {torch.sum(mags**2).item()*100:.2f}%")

# ==========================================================
# RESULTS
# ==========================================================
final_U = U_master.detach() / torch.norm(U_master.detach())
final_projections = torch.matmul(DNA_ND, final_U)
final_power = torch.sum(torch.abs(final_projections)**2).item()

print("\nDISCOVERY COMPLETE")
print(f"Final Power Capture: {final_power*100:.2f}%")
print(f"Redundancy Factor: {1.0/final_power:.2f}x")

# Visualization
plt.figure(figsize=(12, 5))
plt.subplot(121)
plt.plot(history)
plt.title("Master Vector Convergence")
plt.xlabel("Step")
plt.ylabel("Loss (Incoherence)")

plt.subplot(122)
plt.bar(range(12), torch.abs(final_projections).cpu().numpy())
plt.title("Final 12-Tone Resonance")
plt.xlabel("DNA Mode")
plt.ylabel("Amplitude")
plt.show()

# Save the DNA of the Universe
torch.save(final_U, 'perfect_master_vector.pt')
print("Perfect Master Vector saved as 'perfect_master_vector.pt'")