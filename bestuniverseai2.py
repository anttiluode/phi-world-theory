import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# ==========================================================
# 1. THE RESONANT ARCHITECTURE (DNA-based)
# ==========================================================
class ResonantMind(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 10),
            nn.LogSoftmax(dim=1)
        )
    def forward(self, x):
        return self.classifier(x)

# ==========================================================
# 2. THE DNA ENCODER (Icosahedral Filter)
# ==========================================================
class BuckyballLens:
    def __init__(self, img_size=8):
        self.img_size = img_size
        phi = (1 + 5**0.5) / 2
        # The 12 Golden Ratio vectors that define the Buckyball peaks
        self.dna = np.array([
            [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
            [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
        ])
        x = np.linspace(-np.pi, np.pi, img_size)
        self.X, self.Y = np.meshgrid(x, x)
        self.kernels = [np.cos(k[0]*self.X + k[1]*self.Y + k[2]) for k in self.dna]

    def encode(self, images):
        batch_size = images.shape[0]
        imgs = images.reshape(batch_size, 8, 8)
        features = []
        for kernel in self.kernels:
            # Resonant Projection (Matched Filter)
            resonance = np.sum(imgs * kernel, axis=(1, 2))
            features.append(resonance)
        return torch.tensor(np.stack(features, axis=1), dtype=torch.float32)

# ==========================================================
# 3. THE EXPERIMENT: HUMAN MNIST + TOTAL CHAOS
# ==========================================================
def run_showdown():
    print("--- MNIST RESONANT SHOWDOWN ---")
    print("Loading Real Human Handwritten Digits (8x8)...")
    digits = load_digits()
    X, y = digits.data, digits.target
    
    # Normalize data
    X = X / 16.0 
    
    # Split into Train (Clean) and Test (Dirty)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Add EXTREME NOISE to Test Set (Level 1.5 - The standard NN killer)
    noise_level = 1.5
    X_test_noisy = X_test + np.random.randn(*X_test.shape) * noise_level
    
    # --- MODEL A: STANDARD NN (sees 64 pixels) ---
    std_model = ResonantMind(64)
    std_opt = optim.Adam(std_model.parameters(), lr=0.01)
    
    # --- MODEL B: RESONANT MIND (sees 12 DNA notes) ---
    lens = BuckyballLens()
    res_model = ResonantMind(12)
    res_opt = optim.Adam(res_model.parameters(), lr=0.01)
    
    train_labels = torch.tensor(y_train, dtype=torch.long)
    train_pixels = torch.tensor(X_train, dtype=torch.float32)
    train_resonant = lens.encode(X_train)

    print(f"Training both models on {len(X_train)} samples...")
    for epoch in range(200):
        # Train Standard
        std_opt.zero_grad()
        std_loss = nn.NLLLoss()(std_model(train_pixels), train_labels)
        std_loss.backward(); std_opt.step()
        
        # Train Resonant
        res_opt.zero_grad()
        res_loss = nn.NLLLoss()(res_model(train_resonant), train_labels)
        res_loss.backward(); res_opt.step()
        
    # --- FINAL TEST: TOTAL CHAOS ---
    with torch.no_grad():
        # Standard Eval
        test_pixels_noisy = torch.tensor(X_test_noisy, dtype=torch.float32)
        std_preds = std_model(test_pixels_noisy).argmax(dim=1).numpy()
        std_acc = np.mean(std_preds == y_test)
        
        # Resonant Eval
        test_resonant_noisy = lens.encode(X_test_noisy)
        res_preds = res_model(test_resonant_noisy).argmax(dim=1).numpy()
        res_acc = np.mean(res_preds == y_test)

    print("\n" + "="*40)
    print(f"RESULTS AT NOISE LEVEL {noise_level}")
    print(f"Random Guessing Baseline: 10.00%")
    print(f"Standard AI Accuracy:     {std_acc*100:.2f}%")
    print(f"Resonant DNA Accuracy:    {res_acc*100:.2f}%")
    print("="*40)
    
    advantage = (res_acc - std_acc) / (std_acc + 1e-6) * 100
    print(f"The Buckyball DNA is {advantage:.1f}% more robust than Standard Pixels.")

if __name__ == "__main__":
    run_showdown()