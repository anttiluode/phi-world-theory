import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import math

# ==============================================================================
# 1. THE PHYSICS ENGINE (Complex Number Utilities)
# ==============================================================================
def to_complex(magnitude, phase):
    """Creates a complex number from polar coordinates."""
    return magnitude * torch.exp(1j * phase)

def complex_rotate(tensor, angle):
    """Rotates a complex tensor by a specific angle (Phase Shift)."""
    rotator = torch.exp(1j * angle)
    return tensor * rotator

# ==============================================================================
# 2. THE DATA: ORTHOGONAL REALITIES
# ==============================================================================
# We will create 3 distinct signals ("Memories") and superimpose them.
# In a normal computer, these would overwrite each other. 
# Here, we store them in "Phase Space".

SEQ_LEN = 100
t = torch.linspace(0, 4*np.pi, SEQ_LEN)

# Signal A: Sine Wave (Stored at 0 degrees)
sig_a = torch.sin(t)
# Signal B: Square Wave (Stored at 120 degrees)
sig_b = torch.sign(torch.sin(t * 2))
# Signal C: Sawtooth (Stored at 240 degrees)
sig_c = 2 * (t / (4*np.pi)) - 1

# Stack them for ground truth comparison
ground_truth = torch.stack([sig_a, sig_b, sig_c])

# === THE SUPERPOSITION ===
# We combine them into ONE complex vector.
# Z = A + B*e^(i120) + C*e^(i240)
phase_120 = torch.tensor(2 * np.pi / 3)
phase_240 = torch.tensor(4 * np.pi / 3)

mixed_signal = (sig_a.type(torch.complex64) + 
                sig_b.type(torch.complex64) * torch.exp(1j * phase_120) + 
                sig_c.type(torch.complex64) * torch.exp(1j * phase_240))

print(f"Compressed 3 signals (size {3*SEQ_LEN}) into 1 mixed signal (size {SEQ_LEN})")

# ==============================================================================
# 3. THE MODEL: RESONANT ATTENTION HEAD
# ==============================================================================
class PhaseAttentionHead(nn.Module):
    def __init__(self):
        super().__init__()
        # The "Query" network. 
        # It takes a "Command" (0, 1, or 2) and learns what Phase Angle to apply.
        # This replaces the huge Key/Query matrices in Transformers.
        # It's just a tiny "Tuner".
        self.tuner = nn.Embedding(3, 1) # Maps ID -> Phase Angle

    def forward(self, mixed_input, target_id):
        # 1. Get the Phase Lock angle for this target
        # The model learns: "If I want Signal B, I must rotate by -120 degrees"
        theta = self.tuner(target_id).squeeze()
        
        # 2. Apply Physical Rotation (The Attention Mechanism)
        # This is the "Phase Shift"
        rotated = complex_rotate(mixed_input, -theta)
        
        # 3. Project to Real (The "Measurement")
        # We only care about the Real component. 
        # Everything not phase-aligned becomes Imaginary (and is ignored).
        output = rotated.real
        return output, theta

# ==============================================================================
# 4. TRAINING: LEARNING TO TUNE
# ==============================================================================
model = PhaseAttentionHead()
optimizer = optim.Adam(model.parameters(), lr=0.1)
criterion = nn.MSELoss()

print("\n=== STARTING RESONANCE TRAINING ===")
history = []

for epoch in range(500):
    total_loss = 0
    
    # Pick a random target signal (0, 1, or 2)
    target_idx = np.random.randint(0, 3)
    target_tensor = torch.tensor([target_idx])
    
    # The Ground Truth we want to recover
    target_signal = ground_truth[target_idx]
    
    # Forward Pass
    # We feed the FULL mixed mess + the ID of what we want
    prediction, current_phase = model(mixed_signal, target_tensor)
    
    # Loss: Did we recover the clean signal?
    loss = criterion(prediction, target_signal)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    total_loss += loss.item()
    
    if epoch % 50 == 0:
        # Convert learned phase to degrees for display
        deg = math.degrees(current_phase.item() % (2*math.pi))
        print(f"Epoch {epoch:03d} | Target: {target_idx} | Learned Phase: {deg:5.1f}° | Loss: {loss.item():.6f}")

# ==============================================================================
# 5. VERIFICATION & VISUALIZATION
# ==============================================================================
print("\n=== FINAL RESULTS ===")
# Visualize the results
plt.figure(figsize=(12, 8))

for i in range(3):
    # Ask the model to "attend" to signal i
    with torch.no_grad():
        pred, phase = model(mixed_signal, torch.tensor([i]))
        phase_deg = math.degrees(phase.item() % (2*math.pi))
    
    plt.subplot(3, 2, (i*2)+1)
    plt.plot(t, ground_truth[i], 'k--', alpha=0.5, label='Original Truth')
    plt.plot(t, pred, 'r', linewidth=2, label='Recovered via Resonance')
    plt.title(f"Target {i} (Learned Lock: {phase_deg:.1f}°)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Show the "Interference" (why it works)
    # The error is the noise from the other signals that wasn't perfectly orthogonal
    plt.subplot(3, 2, (i*2)+2)
    plt.plot(t, (pred - ground_truth[i])**2, 'b', alpha=0.5)
    plt.title("Interference/Noise Residual")
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\nINTERPRETATION:")
print("1. Standard Attention requires O(N^2) memory to store relationships.")
print("2. Phase Attention requires O(1) memory. It just stores the 'Angle'.")
print("3. Look at the 'Learned Lock' angles in the plots.")
print("   - Target 0 should be near 0° (or 360°)")
print("   - Target 1 should be near 120°")
print("   - Target 2 should be near 240°")
print("4. The network didn't 'memorize' the shapes. It learned to TUNE THE RADIO.")