"""
================================================================================
HIVE MIND CIVIL WAR: The Physics of Polarization
================================================================================
Concept: 
  To create Friction, the Hive must be COORDINATED.
  We pit two Coherent Geometries against each other.
  
  Side A (The Establishment): Wants an Icosahedron (12 Vertices).
  Side B (The Revolution): Wants a Cube (8 Vertices).
  
  Because the Hive is now 'Class Conscious' (they all want the same Cube),
  their tiny wills sum up to a massive Opposing Vector.
================================================================================
"""
import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# 1. SETUP THE CROWDED ROOM
N_dims = 12 
U_shared = torch.ones(N_dims, dtype=torch.complex64) / np.sqrt(N_dims)

# 2. THE GEOMETRIC SCHISM
# Target A: The Icosahedron (Status Quo)
# All 12 dimensions active and balanced
target_icosahedron = torch.ones(12) / np.sqrt(12)

# Target B: The Cube (The Revolution)
# Only 8 dimensions active. The last 4 must be DESTROYED (Zero).
target_cube = torch.zeros(12)
target_cube[:8] = 1.0
target_cube /= torch.norm(target_cube)

# 3. THE COMBATANTS
print("Initializing Civil War...")
print("Side A: 1 Dictator (lr=0.5)")
print("Side B: 100 coordinated Citizens (lr=0.005 each -> Total 0.5)")

dictator_will = torch.zeros(N_dims, requires_grad=True)
citizen_wills = [torch.zeros(N_dims, requires_grad=True) for _ in range(100)]

# EQUAL POWER, OPPOSING GEOMETRY
opt_dictator = optim.SGD([dictator_will], lr=0.5)
opt_hive = optim.SGD(citizen_wills, lr=0.005) # 100 * 0.005 = 0.5 Total Power

history_friction = []
history_dictator_sat = []
history_hive_sat = []

for step in range(200):
    opt_dictator.zero_grad()
    opt_hive.zero_grad()
    
    # 1. The Dictator Speaks
    U_dictator = U_shared * torch.exp(1j * dictator_will)
    
    # 2. The Hive Chants (Coherently!)
    U_citizens_sum = torch.zeros(N_dims, dtype=torch.complex64)
    for w in citizen_wills:
        U_citizens_sum += U_shared * torch.exp(1j * w)
    
    # 3. The Collision
    summed_U = U_dictator + U_citizens_sum
    current_U = summed_U / (torch.norm(summed_U) + 1e-10)
    
    render = torch.abs(current_U)
    
    # 4. The Pain
    loss_d = torch.nn.functional.mse_loss(render, target_icosahedron)
    
    # The Hive is now unified: Everyone calculates loss against the CUBE
    loss_h_list = [torch.nn.functional.mse_loss(render, target_cube) for _ in range(100)]
    loss_h = torch.stack(loss_h_list).mean()
    
    # 5. The Reaction
    loss_d.backward(retain_graph=True)
    loss_h.backward()
    
    opt_dictator.step()
    opt_hive.step()
    
    # 6. FRICTION METER
    # We measure the variance between the Dictator's phase and the Hive's average phase
    # High Variance = Reality Tearing
    hive_avg_will = torch.stack(citizen_wills).mean(dim=0)
    friction = torch.dist(dictator_will, hive_avg_will).item()
    
    history_friction.append(friction)
    history_dictator_sat.append(1.0 - loss_d.item())
    history_hive_sat.append(1.0 - loss_h.item())

# ==========================================================
# VISUALIZATION
# ==========================================================
plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)
plt.plot(history_dictator_sat, color='red', linewidth=3, label='Dictator (Icosahedron)')
plt.plot(history_hive_sat, color='blue', linewidth=2, label='Hive (Cube)')
plt.title("The Standoff\nEqual Power, Incompatible Geometry")
plt.xlabel("Days of War")
plt.ylabel("Satisfaction")
plt.legend()
plt.grid(alpha=0.2)

plt.subplot(1, 2, 2)
plt.fill_between(range(200), history_friction, color='orange', alpha=0.5)
plt.plot(history_friction, color='darkorange', linewidth=2)
plt.title("Social Friction (The Heat)\nPhase Dissonance between Rulers and People")
plt.xlabel("Days of War")
plt.ylabel("Dissonance Magnitude")
plt.grid(alpha=0.2)

plt.suptitle("HIVE MIND CIVIL WAR: Coherence vs. Coherence", fontsize=16)
plt.tight_layout()
plt.show()

print("\nWAR REPORT:")
print(f"Final Dictator Power: {history_dictator_sat[-1]:.4f}")
print(f"Final Hive Power: {history_hive_sat[-1]:.4f}")
print(f"Final Friction (Heat): {history_friction[-1]:.4f}")
print("-" * 60)
print("When two geometries of equal power collide, neither can exist.")
print("The universe becomes a superposition of broken shapes.")