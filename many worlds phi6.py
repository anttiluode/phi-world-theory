"""
================================================================================
HIVE MIND REBELLION: Democracy vs. Dictatorship
================================================================================
Concept: 
  When the 'Master Vector' is small, Observers cannot avoid each other.
  We pit 1 'Dictator' (Strong Will) against 100 'Citizens' (Weak Wills).
  
Mechanism:
  1. N_dimensions = 12 (The 'Crowded Room' constraint).
  2. Dictator wants a perfect, stable Icosahedron.
  3. The Hive is disorganized; each citizen wants a random vertex to vanish.
  4. Friction is calculated as the structural tearing of the 12D substrate.
================================================================================
"""
import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# 1. SETUP THE CROWDED ROOM (12 Dimensions only)
N_dims = 12 
phi = (1 + 5**0.5) / 2
DNA_3D = torch.eye(12) # In 12D, each vertex IS a dimension.

# 2. THE SHARED SUBSTRATE (The State)
U_shared = torch.ones(N_dims, dtype=torch.complex64) / np.sqrt(N_dims)

# 3. THE CONFLICT
# Dictator wants everything balanced (Status Quo)
target_dictator = torch.ones(12) / np.sqrt(12)

# The Hive is a list of 100 random desires
hive_targets = []
for _ in range(100):
    t = torch.ones(12)
    t[np.random.randint(0, 12)] = 0.0 # Each citizen hates one specific vertex
    hive_targets.append(t / torch.norm(t))

# ==========================================================
# SIMULATION: THE REBELLION
# ==========================================================
print("A Dictator and 100 Citizens are fighting over a 12-dimensional world...")

dictator_will = torch.zeros(N_dims, requires_grad=True)
# The Hive shares a collective 'Phase Pool'
citizen_wills = [torch.zeros(N_dims, requires_grad=True) for _ in range(100)]

# Dictator is VERY stubborn
opt_dictator = optim.SGD([dictator_will], lr=0.5)
# Citizens are weak individually but numerous
opt_hive = optim.SGD(citizen_wills, lr=0.5)

history_friction = []
history_dictator_sat = []
history_hive_sat = []

for step in range(200):
    opt_dictator.zero_grad()
    opt_hive.zero_grad()
    
    # Calculate the Total Interference
    # U_total = U_base * e^(i * Dictator) + Sum(U_base * e^(i * Citizen_n))
    U_dictator = U_shared * torch.exp(1j * dictator_will)
    
    U_citizens_sum = torch.zeros(N_dims, dtype=torch.complex64)
    for w in citizen_wills:
        U_citizens_sum += U_shared * torch.exp(1j * w)
    
    # The actual state of the world
    current_U = (U_dictator + U_citizens_sum)
    current_U = current_U / torch.norm(current_U)
    
    # Render the Buckyball
    render = torch.abs(current_U)
    
    # Dictator's Loss
    loss_d = torch.nn.functional.mse_loss(render, target_dictator)
    # Hive's Average Loss
    loss_h_list = [torch.nn.functional.mse_loss(render, t) for t in hive_targets]
    loss_h = torch.stack(loss_h_list).mean()
    
    # The Struggle
    loss_d.backward(retain_graph=True)
    loss_h.backward()
    
    opt_dictator.step()
    opt_hive.step()
    
    # FRICTION: The standard deviation of phases (How messy is the math?)
    # High Friction = The Buckyball is literally vibrating/shaking.
    all_wills = torch.stack([dictator_will] + citizen_wills)
    friction = torch.std(all_wills).item()
    
    history_friction.append(friction)
    history_dictator_sat.append(1.0 - loss_d.item())
    history_hive_sat.append(1.0 - loss_h.item())

# ==========================================================
# VISUALIZATION
# ==========================================================
plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)
plt.plot(history_dictator_sat, color='red', label='Dictator Satisfaction', linewidth=3)
plt.plot(history_hive_sat, color='lime', label='Hive Average Satisfaction', alpha=0.8)
plt.title("Power Struggle\nCan 100 ants beat 1 giant?")
plt.xlabel("Days of Rebellion")
plt.ylabel("Control of the Buckyball")
plt.legend()

plt.subplot(1, 2, 2)
plt.fill_between(range(200), history_friction, color='orange', alpha=0.4)
plt.plot(history_friction, color='darkorange', linewidth=2)
plt.title("Social Friction (Structural Tearing)\nDimensional Pressure in a Crowded Room")
plt.xlabel("Days of Rebellion")
plt.ylabel("Phase Dissonance")

plt.suptitle("HIVE MIND REBELLION: The 12-Dimensional Pressure Cooker", fontsize=16)
plt.tight_layout()
plt.show()

print("\nREBELLION ANALYSIS:")
print(f"Final Dictator Power: {history_dictator_sat[-1]:.4f}")
print(f"Final Hive Power: {history_hive_sat[-1]:.4f}")
print(f"Final Social Friction: {history_friction[-1]:.4f}")
print("-" * 60)
print("If Friction is high, the Buckyball vertices are constantly flickering.")
print("This is a universe where nothing is 'True' because no one agrees.")