import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.preprocessing import MinMaxScaler
import math

# ==============================================================================
# 0. DATA PREPARATION: CREATING THE "MESS"
# ==============================================================================
print("Loading Digits...")
digits = load_digits()
X = digits.data
X = MinMaxScaler().fit_transform(X) # Normalize 0-1

# Create pairs of overlapping images
# We will create a dataset where Input = ImageA + ImageB
# Target 0 = ImageA, Target 1 = ImageB
N_SAMPLES = 1000
IMAGE_DIM = 64 # 8x8 images

pairs_A = X[np.random.choice(len(X), N_SAMPLES)]
pairs_B = X[np.random.choice(len(X), N_SAMPLES)]

# The "Mess"
inputs_mixed = (pairs_A + pairs_B) / 2.0  # Average them so pixel range is still valid-ish

# Convert to Tensors
inputs_t = torch.tensor(inputs_mixed, dtype=torch.float32)
targets_A = torch.tensor(pairs_A, dtype=torch.float32)
targets_B = torch.tensor(pairs_B, dtype=torch.float32)

print(f"Created {N_SAMPLES} superimposed realities.")

# ==============================================================================
# 1. STANDARD AI (The Control)
# ==============================================================================
# Architecture: Input (64 pixels) + 1 Selector Bit -> Hidden -> Output (64 pixels)
# Uses ReLU and Standard Weights.

class StandardNet(nn.Module):
    def __init__(self, hidden_size=64):
        super().__init__()
        # +1 for the "Selector" (0.0 for Image A, 1.0 for Image B)
        self.l1 = nn.Linear(IMAGE_DIM + 1, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, IMAGE_DIM)
    
    def forward(self, x, selector):
        # selector is shape (batch, 1)
        x_aug = torch.cat([x, selector], dim=1)
        h = self.relu(self.l1(x_aug))
        return torch.sigmoid(self.l2(h)) # Sigmoid to force 0-1 image

# ==============================================================================
# 2. PHASE AI (The Experiment)
# ==============================================================================
# Architecture: Complex Input -> Complex Weights -> Phase Rotation -> Real Projection
# Uses "Holographic Storage" logic.

class ComplexLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        # We store weights as Real and Imaginary parts
        self.real_w = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.imag_w = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.real_b = nn.Parameter(torch.zeros(out_features))
        self.imag_b = nn.Parameter(torch.zeros(out_features))
        
    def forward(self, z):
        # z is complex (real, imag)
        # (a+bi)(c+di) = (ac-bd) + i(ad+bc)
        r = z.real
        i = z.imag
        
        out_real = torch.nn.functional.linear(r, self.real_w, self.real_b) - \
                   torch.nn.functional.linear(i, self.imag_w, self.imag_b)
        out_imag = torch.nn.functional.linear(r, self.imag_w, self.imag_b) + \
                   torch.nn.functional.linear(i, self.real_w, self.real_b)
                   
        return torch.complex(out_real, out_imag)

class PhaseNet(nn.Module):
    def __init__(self, hidden_size=64):
        super().__init__()
        # Note: Complex weights have 2x parameters (Real+Imag). 
        # To make it a fair fight, we should adjust hidden size, 
        # but here we'll keep hidden size same to test "Representation Density".
        
        self.l1 = ComplexLinear(IMAGE_DIM, hidden_size)
        # Activation: Complex ReLU (Apply to Magnitude, keep Phase? Or independent?)
        # Let's use "ModReLU" logic: Non-linearity on magnitude.
        # Simple version: z * sigmoid(|z|) - soft gating
        self.l2 = ComplexLinear(hidden_size, IMAGE_DIM)
        
    def complex_relu(self, z):
        # Apply ReLU to Real and Imag independently (simplest stable non-linearity)
        return torch.complex(torch.relu(z.real), torch.relu(z.imag))

    def forward(self, x_real, target_phase_angle):
        # Convert Real Input to Complex (Imaginary part is 0 initially)
        z = torch.complex(x_real, torch.zeros_like(x_real))
        
        # Layer 1
        h = self.l1(z)
        h = self.complex_relu(h)
        
        # Layer 2
        z_out = self.l2(h)
        
        # === THE MECHANISM ===
        # Rotate the output universe by the requested phase angle
        # If we want Reality A, we rotate so A falls on Real Axis.
        rotator = torch.exp(1j * target_phase_angle)
        z_rotated = z_out * rotator
        
        # We read the REAL component as our physical output.
        # The "Other" reality is hidden in the Imaginary component.
        return torch.sigmoid(z_rotated.real)

# ==============================================================================
# 3. TRAINING LOOP
# ==============================================================================
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# -- Setup Models --
HIDDEN = 128
std_model = StandardNet(hidden_size=HIDDEN)
# Phase model has 2x floats per weight, so we check params
phase_model = PhaseNet(hidden_size=HIDDEN // 2) # Halve neurons to keep parameter count closer?
# Actually, let's keep neurons same to test "Information Density per Neuron".
# Or matching params is fairer.
# ComplexLinear(64, 64) = 64*64*2 weights. Standard(64,64) = 64*64.
# Let's match Params.
phase_model = PhaseNet(hidden_size=HIDDEN // 2 + 10) 

print(f"\nStandard AI Params: {count_parameters(std_model)}")
print(f"Phase AI Params:    {count_parameters(phase_model)}")
print("(Kept roughly equal for fair comparison)\n")

# -- Optimizers --
opt_std = optim.Adam(std_model.parameters(), lr=0.005)
opt_phase = optim.Adam(phase_model.parameters(), lr=0.005)
criterion = nn.MSELoss()

print("Training for 200 epochs...")

loss_history_std = []
loss_history_phase = []

for epoch in range(200):
    # --- Train Standard ---
    # Task A: Input + 0 -> Target A
    sel_A = torch.zeros(N_SAMPLES, 1)
    pred_A = std_model(inputs_t, sel_A)
    loss_std_A = criterion(pred_A, targets_A)
    
    # Task B: Input + 1 -> Target B
    sel_B = torch.ones(N_SAMPLES, 1)
    pred_B = std_model(inputs_t, sel_B)
    loss_std_B = criterion(pred_B, targets_B)
    
    loss_std = loss_std_A + loss_std_B
    
    opt_std.zero_grad()
    loss_std.backward()
    opt_std.step()
    loss_history_std.append(loss_std.item())
    
    # --- Train Phase ---
    # Task A: Rotate by 0 -> Target A
    phase_0 = torch.zeros(1)
    pred_p_A = phase_model(inputs_t, phase_0)
    loss_p_A = criterion(pred_p_A, targets_A)
    
    # Task B: Rotate by PI/2 (90 deg) -> Target B
    # This forces Target B to live on the Imaginary axis relative to A
    phase_90 = torch.tensor(np.pi / 2)
    pred_p_B = phase_model(inputs_t, phase_90)
    loss_p_B = criterion(pred_p_B, targets_B)
    
    # Constraint: Minimize "Crosstalk" (Optional, but let's just optimize reconstruction)
    loss_phase = loss_p_A + loss_p_B
    
    opt_phase.zero_grad()
    loss_phase.backward()
    opt_phase.step()
    loss_history_phase.append(loss_phase.item())
    
    if epoch % 20 == 0:
        print(f"Ep {epoch:03d} | Std Loss: {loss_std.item():.5f} | Phase Loss: {loss_phase.item():.5f}")

# ==============================================================================
# 4. RESULTS & VISUALIZATION
# ==============================================================================
print("\n=== EVALUATION ===")
print(f"Final Standard Loss: {loss_history_std[-1]:.5f}")
print(f"Final Phase Loss:    {loss_history_phase[-1]:.5f}")

# Let's visualize one example
idx = 0
sample_in = inputs_t[idx:idx+1]
target_a = targets_A[idx].reshape(8,8)
target_b = targets_B[idx].reshape(8,8)

# Standard Inference
out_std_a = std_model(sample_in, torch.zeros(1,1)).detach().reshape(8,8)
out_std_b = std_model(sample_in, torch.ones(1,1)).detach().reshape(8,8)

# Phase Inference
out_phase_a = phase_model(sample_in, torch.tensor(0.0)).detach().reshape(8,8)
out_phase_b = phase_model(sample_in, torch.tensor(np.pi/2)).detach().reshape(8,8)

# Plot
plt.figure(figsize=(12, 6))

# Row 1: Truth & Input
plt.subplot(3, 4, 1); plt.imshow(inputs_mixed[idx].reshape(8,8), cmap='gray'); plt.title("The Mixed Input")
plt.axis('off')
plt.subplot(3, 4, 2); plt.imshow(target_a, cmap='gray'); plt.title("Target A (Truth)")
plt.axis('off')
plt.subplot(3, 4, 3); plt.imshow(target_b, cmap='gray'); plt.title("Target B (Truth)")
plt.axis('off')

# Row 2: Standard AI
plt.subplot(3, 4, 5); plt.text(0.5,0.5,"Standard AI", ha='center'); plt.axis('off')
plt.subplot(3, 4, 6); plt.imshow(out_std_a, cmap='gray'); plt.title("Std Recover A")
plt.axis('off')
plt.subplot(3, 4, 7); plt.imshow(out_std_b, cmap='gray'); plt.title("Std Recover B")
plt.axis('off')
plt.subplot(3, 4, 8); plt.plot(loss_history_std); plt.title("Std Training Curve"); plt.grid(True)

# Row 3: Phase AI
plt.subplot(3, 4, 9); plt.text(0.5,0.5,"Phase AI", ha='center'); plt.axis('off')
plt.subplot(3, 4, 10); plt.imshow(out_phase_a, cmap='gray'); plt.title("Phase Recover A")
plt.axis('off')
plt.subplot(3, 4, 11); plt.imshow(out_phase_b, cmap='gray'); plt.title("Phase Recover B")
plt.axis('off')
plt.subplot(3, 4, 12); plt.plot(loss_history_phase, color='orange'); plt.title("Phase Training Curve"); plt.grid(True)

plt.tight_layout()
plt.show()

# === THE AUTO-TUNING DEMO ===
# Scanning through angles to see the transition
print("\nScanning Phase Angles (Auto-Tuning Demo)...")
phases = np.linspace(0, np.pi/2, 5)
print(f"Input: Mixed Image")
for p in phases:
    out = phase_model(sample_in, torch.tensor(float(p))).detach().reshape(8,8)
    deg = int(math.degrees(p))
    print(f"Phase {deg}°: Output Mean Intensity {out.mean():.3f}") 
    # In a real UI, this is where the image morphs from A to B