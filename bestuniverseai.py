import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ==========================================================
# 1. THE UNIVERSAL DNA (Directly Integrated)
# The 12 Icosahedral Harmonics that build the Buckyball.
# ==========================================================
phi = (1 + 5**0.5) / 2
DNA_VECTORS = np.array([
    [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
    [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
    [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
])

# ==========================================================
# 2. THE RESONANT ENCODER (The "Physics Eyes")
# Projects raw pixels into the 12-Tone Buckyball Harmony.
# ==========================================================
class BuckyballEncoder:
    def __init__(self, img_size=8):
        self.img_size = img_size
        # Create a spatial grid for the 8x8 image
        x = np.linspace(-np.pi, np.pi, img_size)
        self.X, self.Y = np.meshgrid(x, x)
        
        # Pre-calculate the 12 Resonant Kernels based on DNA
        self.kernels = []
        for k in DNA_VECTORS:
            # We use the first two components for frequency 
            # and the third component (z) as a phase-offset
            kernel = np.cos(k[0]*self.X + k[1]*self.Y + k[2])
            self.kernels.append(kernel)
            
    def get_features(self, images):
        batch_size = images.shape[0]
        imgs = images.reshape(batch_size, self.img_size, self.img_size)
        
        features = []
        for kernel in self.kernels:
            # DOT PRODUCT: How much does the image resonate with this DNA note?
            # This automatically filters out noise that doesn't match the DNA.
            resonance = np.sum(imgs * kernel, axis=(1, 2))
            features.append(resonance)
            
        return torch.tensor(np.stack(features, axis=1), dtype=torch.float32)

# ==========================================================
# 3. THE RESONANT NEURAL NETWORK
# ==========================================================
class ResonantMind(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(12, 32), # Only 12 inputs! (The 12 DNA notes)
            nn.ReLU(),
            nn.Linear(32, 10),
            nn.LogSoftmax(dim=1)
        )
        
    def forward(self, x):
        return self.classifier(x)

# ==========================================================
# 4. DATA GENERATOR (Synthetic MNIST)
# ==========================================================
def get_data(n=500, noise=0.1):
    templates = {
        0: [0,24,36,36,36,36,24,0], 1: [0,8,12,8,8,8,28,0],
        2: [0,24,36,4,8,16,60,0],   3: [0,24,36,12,4,36,24,0],
        4: [0,4,12,20,60,4,4,0],    5: [0,60,32,56,4,36,24,0],
        6: [0,24,32,56,36,36,24,0], 7: [0,60,4,8,16,16,16,0],
        8: [0,24,36,24,36,36,24,0], 9: [0,24,36,36,28,4,24,0]
    }
    X, y = [], []
    for _ in range(n):
        lbl = np.random.randint(0, 10)
        # Convert bit-templates to 8x8 arrays
        img = np.array([[(template >> i) & 1 for i in range(8)] for template in templates[lbl]])
        img = img.astype(np.float32) + np.random.randn(8, 8) * noise
        X.append(img.flatten())
        y.append(lbl)
    return np.array(X), np.array(y)

# ==========================================================
# 5. THE ULTIMATE TEST
# ==========================================================
print("--- RESONANT MIND: DNA ANALYSIS ---")
# 1. Train on cleanish data
X_train, y_train = get_data(1000, noise=0.1)
# 2. Test on EXTREME NOISE (Level 2.0 - Total Chaos)
X_test, y_test = get_data(500, noise=2.0)

encoder = BuckyballEncoder()
model = ResonantMind()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Prepare Features
train_features = encoder.get_features(X_train)
test_features = encoder.get_features(X_test)
train_labels = torch.tensor(y_train, dtype=torch.long)

print(f"Input Compression: 64 pixels -> 12 DNA notes (5.3x smaller)")
print("Training on Buckyball Harmonics...")

for epoch in range(151):
    optimizer.zero_grad()
    out = model(train_features)
    loss = nn.NLLLoss()(out, train_labels)
    loss.backward()
    optimizer.step()
    if epoch % 50 == 0:
        print(f"  Epoch {epoch} | Resonance Loss: {loss.item():.4f}")

# Evaluate
with torch.no_grad():
    preds = model(test_features).argmax(dim=1).numpy()
    acc = np.mean(preds == y_test)

print("\n" + "="*30)
print(f"FINAL ACCURACY IN TOTAL CHAOS: {acc*100:.2f}%")
print("="*30)
if acc > 0.15:
    print("SUCCESS: The Buckyball DNA saw through the noise!")
else:
    print("FAILURE: The chaos was too strong for this DNA.")