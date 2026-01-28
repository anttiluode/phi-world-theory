"""
================================================================================
THE REALITY GYROSCOPE: Phase-Lock Stabilization
================================================================================
Concept: 
  Accidental branching is 'Phase Drift'. 
  The Gyroscope is a background process that keeps the Observer 
  locked to a specific 'Gold' state of the Buckyball.
  
Mechanism:
  1. Define the 'Target Anchor' (Your stable 3D life).
  2. Simulate 'Entropy' (Random phase jitter in 1000D).
  3. The Gyroscope measures the Dissonance and applies a 
     RESTORATIVE FORCE (Phase Correction) to prevent the split.
================================================================================
"""
import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# 1. SETUP THE ANCHOR (The 12-Tone Lens)
phi = (1 + 5**0.5) / 2
DNA_3D = torch.tensor([
    [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
    [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
    [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
]).float()
DNA_3D /= torch.norm(DNA_3D, dim=1, keepdim=True)

# 2. THE MASTER VECTOR (The Substrate)
U_master = torch.zeros(1000, dtype=torch.complex64)
U_master[:12] = 1.0 + 0j
U_master /= torch.norm(U_master)

# 3. DEFINE THE 'GOLDEN TIMELINE' (The state we want to stay in)
# All 12 vertices equally resonant = Perfect Balance.
target_resonance = torch.ones(12) / np.sqrt(12)

def get_current_state(vector):
    """Measures the 3D projection of the vector."""
    res = []
    for i in range(12):
        P = torch.zeros(1000, dtype=torch.complex64)
        P[i] = 1.0
        res.append(torch.vdot(P, vector).abs())
    curr = torch.stack(res)
    return curr / (torch.norm(curr) + 1e-10)

# ==========================================================
# SIMULATION: 100 MOMENTS OF EXISTENCE
# ==========================================================
print("Activating Reality Gyroscope...")

current_vector = U_master.clone()
history_no_gyro = []
history_with_gyro = []

# Initialize correction phase (The Gyro's engine)
correction = torch.zeros(1000, requires_grad=True)
optimizer = optim.SGD([correction], lr=0.5) # High-speed correction

for t in range(100):
    # --- PHASE 1: ENTROPY ATTACKS ---
    # Random drift knocks the vector out of alignment
    entropy = torch.randn(1000) * 0.05
    current_vector = current_vector * torch.exp(1j * entropy)
    
    # --- PHASE 2: DRIFT (Without Gyro) ---
    res_drift = get_current_state(current_vector)
    loss_drift = torch.nn.functional.mse_loss(res_drift, target_resonance)
    history_no_gyro.append(loss_drift.item())
    
    # --- PHASE 3: CORRECTION (The Gyroscope) ---
    # The brain 'notices' the dissonance and runs a quick optimization
    for _ in range(5): # 5 micro-ticks of mental focus
        optimizer.zero_grad()
        # Predict the correction
        corrected_vec = current_vector * torch.exp(1j * correction)
        res_gyro = get_current_state(corrected_vec)
        loss_gyro = torch.nn.functional.mse_loss(res_gyro, target_resonance)
        loss_gyro.backward()
        optimizer.step()
    
    history_with_gyro.append(loss_gyro.item())
    
    if t % 20 == 0:
        print(f"Step {t} | Dissonance: {loss_gyro.item():.6f}")

# ==========================================================
# VISUALIZATION: STABILITY VS CHAOS
# ==========================================================
plt.figure(figsize=(12, 6))
plt.plot(history_no_gyro, color='red', alpha=0.4, label='Natural Drift (World Splits)')
plt.plot(history_with_gyro, color='cyan', linewidth=2, label='Gyroscope Active (Timeline Locked)')
plt.axhline(y=0.01, color='gold', linestyle='--', label='Coherence Threshold')

plt.title("THE REALITY GYROSCOPE\nStabilizing the 3D Projection against 1000D Entropy", fontsize=16)
plt.xlabel("Moments (Time)")
plt.ylabel("Dissonance (Instability)")
plt.legend()
plt.grid(alpha=0.2)
plt.show()

print("\nSYSTEM ANALYSIS:")
print(f"Final Chaos (No Gyro): {history_no_gyro[-1]:.4f}")
print(f"Final Stability (With Gyro): {history_with_gyro[-1]:.4f}")
print("-" * 60)
print("The Red line shows how you would naturally 'fade' into parallel branches.")
print("The Cyan line shows the Observer's 'Will' keeping the Buckyball solid.")