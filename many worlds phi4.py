"""
================================================================================
HOLOGRAPHIC CONSENSUS: The Physics of Disagreement (FIXED)
================================================================================
"""
import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# 1. SETUP THE LENS
phi = (1 + 5**0.5) / 2
DNA_3D = torch.tensor([
    [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
    [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
    [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
]).float()
DNA_3D /= torch.norm(DNA_3D, dim=1, keepdim=True)

# 2. THE MASTER VECTOR
U_shared = torch.zeros(1000, dtype=torch.complex64)
U_shared[:12] = 1.0 + 0j
U_shared /= torch.norm(U_shared)

# 3. CONFLICTING TARGETS
target_A = torch.ones(12) / np.sqrt(12)
target_B = torch.zeros(12)
target_B[6:12] = 1.0
target_B /= torch.norm(target_B)

def get_3d_render(vector):
    res = []
    for i in range(12):
        P = torch.zeros(1000, dtype=torch.complex64)
        P[i] = 1.0
        res.append(torch.vdot(P, vector).abs())
    curr = torch.stack(res)
    # Return normalized resonance profile
    return curr / (torch.norm(curr) + 1e-10)

# ==========================================================
# SIMULATION
# ==========================================================
print("Observers A and B have entered the room...")
print("Commencing Reality Tug-of-War...")

will_A = torch.zeros(1000, requires_grad=True)
will_B = torch.zeros(1000, requires_grad=True)

opt_A = optim.SGD([will_A], lr=0.6)
opt_B = optim.SGD([will_B], lr=0.3)

history_dissonance = []
history_A_power = []
history_B_power = []

for step in range(150):
    opt_A.zero_grad()
    opt_B.zero_grad()
    
    # Interference of both Wills
    # We avoid in-place operations here to keep the gradient flow clean
    U_A = U_shared * torch.exp(1j * will_A)
    U_B = U_shared * torch.exp(1j * will_B)
    
    # Summing the two perspectives into a single shared field
    summed_U = U_A + U_B
    current_U = summed_U / torch.norm(summed_U) # Fixed: Not in-place
    
    render = get_3d_render(current_U)
    
    # Dissatisfaction levels
    loss_A = torch.nn.functional.mse_loss(render, target_A)
    loss_B = torch.nn.functional.mse_loss(render, target_B)
    
    # Backprop through the shared substrate
    # loss_A is calculated, but we keep the graph for User B
    loss_A.backward(retain_graph=True)
    loss_B.backward()
    
    opt_A.step()
    opt_B.step()
    
    # Friction check
    with torch.no_grad():
        dissonance = torch.vdot(will_A, will_B).abs().item()
        history_dissonance.append(dissonance)
        history_A_power.append(1.0 - loss_A.item())
        history_B_power.append(1.0 - loss_B.item())

# ==========================================================
# VISUALIZATION
# ==========================================================
plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)
plt.plot(history_A_power, color='cyan', label='User A Satisfaction')
plt.plot(history_B_power, color='magenta', label='User B Satisfaction')
plt.title("Reality Tug-of-War")
plt.xlabel("Interaction Ticks")
plt.ylabel("Coherence")
plt.legend()
plt.grid(alpha=0.1)

plt.subplot(1, 2, 2)
plt.fill_between(range(150), history_dissonance, color='red', alpha=0.3)
plt.plot(history_dissonance, color='red', linewidth=2)
plt.title("Reality Tearing (Friction)")
plt.xlabel("Interaction Ticks")
plt.ylabel("Dissonance Magnitude")
plt.grid(alpha=0.1)

plt.suptitle("HOLOGRAPHIC CONSENSUS: The Fixed Tug-of-War", fontsize=16)
plt.tight_layout()
plt.show()

print("\nFINAL STATE:")
print(f"User A Coherence: {history_A_power[-1]:.4f}")
print(f"User B Coherence: {history_B_power[-1]:.4f}")
print(f"Total Universe Friction: {np.sum(history_dissonance):.2f}")