"""
PHI-WORLD NEURAL NETWORK vs STANDARD NEURAL NETWORK
====================================================
Testing Gemini's claim: "best.py has a sort of AI in it"

The hypothesis: If energy-minimizing field dynamics naturally produce
learnable, robust structure, then a neural network that learns via
field dynamics (not gradient descent) should have different properties:
- More robust to noise
- Better generalization from less data
- Different loss landscape (no sharp minima?)

We test this on MNIST - boring but universal benchmark.

Three networks:
1. STANDARD: Normal MLP + Adam optimizer
2. PHI-NET: Weights evolve via phi-world field equations  
3. HYBRID: Standard forward pass, phi-world weight updates

Author: Built for Antti's MatrixInMatrix project
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
import time
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# DATA: Simple version of MNIST-like task
# ============================================================================
def generate_mnist_like_data(n_samples=1000, noise=0.1):
    """
    Generate synthetic digit-like patterns (0-9) as 8x8 images.
    Simpler than real MNIST but captures the essence.
    """
    np.random.seed(42)
    
    # Define digit templates (8x8)
    templates = {
        0: np.array([
            [0,1,1,1,1,1,1,0],
            [1,1,0,0,0,0,1,1],
            [1,1,0,0,0,0,1,1],
            [1,1,0,0,0,0,1,1],
            [1,1,0,0,0,0,1,1],
            [1,1,0,0,0,0,1,1],
            [1,1,0,0,0,0,1,1],
            [0,1,1,1,1,1,1,0],
        ]),
        1: np.array([
            [0,0,0,1,1,0,0,0],
            [0,0,1,1,1,0,0,0],
            [0,0,0,1,1,0,0,0],
            [0,0,0,1,1,0,0,0],
            [0,0,0,1,1,0,0,0],
            [0,0,0,1,1,0,0,0],
            [0,0,0,1,1,0,0,0],
            [0,1,1,1,1,1,1,0],
        ]),
        2: np.array([
            [0,1,1,1,1,1,1,0],
            [1,1,0,0,0,0,1,1],
            [0,0,0,0,0,0,1,1],
            [0,0,0,1,1,1,1,0],
            [0,1,1,1,0,0,0,0],
            [1,1,0,0,0,0,0,0],
            [1,1,0,0,0,0,0,0],
            [1,1,1,1,1,1,1,1],
        ]),
        3: np.array([
            [0,1,1,1,1,1,1,0],
            [1,1,0,0,0,0,1,1],
            [0,0,0,0,0,0,1,1],
            [0,0,1,1,1,1,1,0],
            [0,0,0,0,0,0,1,1],
            [0,0,0,0,0,0,1,1],
            [1,1,0,0,0,0,1,1],
            [0,1,1,1,1,1,1,0],
        ]),
        4: np.array([
            [0,0,0,0,0,1,1,0],
            [0,0,0,0,1,1,1,0],
            [0,0,0,1,1,1,1,0],
            [0,0,1,1,0,1,1,0],
            [0,1,1,0,0,1,1,0],
            [1,1,1,1,1,1,1,1],
            [0,0,0,0,0,1,1,0],
            [0,0,0,0,0,1,1,0],
        ]),
        5: np.array([
            [1,1,1,1,1,1,1,1],
            [1,1,0,0,0,0,0,0],
            [1,1,0,0,0,0,0,0],
            [1,1,1,1,1,1,1,0],
            [0,0,0,0,0,0,1,1],
            [0,0,0,0,0,0,1,1],
            [1,1,0,0,0,0,1,1],
            [0,1,1,1,1,1,1,0],
        ]),
        6: np.array([
            [0,1,1,1,1,1,1,0],
            [1,1,0,0,0,0,0,0],
            [1,1,0,0,0,0,0,0],
            [1,1,1,1,1,1,1,0],
            [1,1,0,0,0,0,1,1],
            [1,1,0,0,0,0,1,1],
            [1,1,0,0,0,0,1,1],
            [0,1,1,1,1,1,1,0],
        ]),
        7: np.array([
            [1,1,1,1,1,1,1,1],
            [0,0,0,0,0,0,1,1],
            [0,0,0,0,0,1,1,0],
            [0,0,0,0,1,1,0,0],
            [0,0,0,1,1,0,0,0],
            [0,0,0,1,1,0,0,0],
            [0,0,0,1,1,0,0,0],
            [0,0,0,1,1,0,0,0],
        ]),
        8: np.array([
            [0,1,1,1,1,1,1,0],
            [1,1,0,0,0,0,1,1],
            [1,1,0,0,0,0,1,1],
            [0,1,1,1,1,1,1,0],
            [1,1,0,0,0,0,1,1],
            [1,1,0,0,0,0,1,1],
            [1,1,0,0,0,0,1,1],
            [0,1,1,1,1,1,1,0],
        ]),
        9: np.array([
            [0,1,1,1,1,1,1,0],
            [1,1,0,0,0,0,1,1],
            [1,1,0,0,0,0,1,1],
            [0,1,1,1,1,1,1,1],
            [0,0,0,0,0,0,1,1],
            [0,0,0,0,0,0,1,1],
            [1,1,0,0,0,0,1,1],
            [0,1,1,1,1,1,1,0],
        ]),
    }
    
    X = []
    y = []
    
    for _ in range(n_samples):
        digit = np.random.randint(0, 10)
        img = templates[digit].astype(np.float32).copy()
        
        # Add noise
        img += np.random.randn(8, 8) * noise
        
        # Random shift (small)
        shift_x = np.random.randint(-1, 2)
        shift_y = np.random.randint(-1, 2)
        img = np.roll(np.roll(img, shift_x, axis=0), shift_y, axis=1)
        
        X.append(img.flatten())
        y.append(digit)
    
    return np.array(X), np.array(y)


def one_hot(y, num_classes=10):
    """Convert labels to one-hot encoding"""
    return np.eye(num_classes)[y]


def softmax(x):
    """Stable softmax"""
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def relu(x):
    return np.maximum(0, x)


def relu_grad(x):
    return (x > 0).astype(np.float32)


# ============================================================================
# NETWORK 1: STANDARD MLP WITH ADAM
# ============================================================================
class StandardMLP:
    """Standard MLP with Adam optimizer - the baseline"""
    
    def __init__(self, layer_sizes, lr=0.001):
        self.layer_sizes = layer_sizes
        self.lr = lr
        self.num_layers = len(layer_sizes) - 1
        
        # Initialize weights (Xavier)
        self.weights = []
        self.biases = []
        for i in range(self.num_layers):
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2.0 / layer_sizes[i])
            b = np.zeros(layer_sizes[i+1])
            self.weights.append(w)
            self.biases.append(b)
        
        # Adam parameters
        self.m_w = [np.zeros_like(w) for w in self.weights]
        self.v_w = [np.zeros_like(w) for w in self.weights]
        self.m_b = [np.zeros_like(b) for b in self.biases]
        self.v_b = [np.zeros_like(b) for b in self.biases]
        self.t = 0
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps = 1e-8
    
    def forward(self, X):
        """Forward pass, store activations for backprop"""
        self.activations = [X]
        self.z_values = []
        
        for i in range(self.num_layers):
            z = self.activations[-1] @ self.weights[i] + self.biases[i]
            self.z_values.append(z)
            
            if i < self.num_layers - 1:
                a = relu(z)
            else:
                a = softmax(z)
            self.activations.append(a)
        
        return self.activations[-1]
    
    def backward(self, y_true):
        """Backpropagation"""
        m = y_true.shape[0]
        
        # Output layer gradient (softmax + cross-entropy)
        delta = self.activations[-1] - y_true
        
        grad_w = []
        grad_b = []
        
        for i in range(self.num_layers - 1, -1, -1):
            gw = self.activations[i].T @ delta / m
            gb = np.mean(delta, axis=0)
            grad_w.insert(0, gw)
            grad_b.insert(0, gb)
            
            if i > 0:
                delta = (delta @ self.weights[i].T) * relu_grad(self.z_values[i-1])
        
        return grad_w, grad_b
    
    def update(self, grad_w, grad_b):
        """Adam update"""
        self.t += 1
        
        for i in range(self.num_layers):
            # Weights
            self.m_w[i] = self.beta1 * self.m_w[i] + (1 - self.beta1) * grad_w[i]
            self.v_w[i] = self.beta2 * self.v_w[i] + (1 - self.beta2) * grad_w[i]**2
            m_hat = self.m_w[i] / (1 - self.beta1**self.t)
            v_hat = self.v_w[i] / (1 - self.beta2**self.t)
            self.weights[i] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
            
            # Biases
            self.m_b[i] = self.beta1 * self.m_b[i] + (1 - self.beta1) * grad_b[i]
            self.v_b[i] = self.beta2 * self.v_b[i] + (1 - self.beta2) * grad_b[i]**2
            m_hat = self.m_b[i] / (1 - self.beta1**self.t)
            v_hat = self.v_b[i] / (1 - self.beta2**self.t)
            self.biases[i] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
    
    def train_step(self, X, y):
        pred = self.forward(X)
        loss = -np.mean(np.sum(y * np.log(pred + 1e-10), axis=1))
        grad_w, grad_b = self.backward(y)
        self.update(grad_w, grad_b)
        return loss
    
    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)
    
    def accuracy(self, X, y):
        return np.mean(self.predict(X) == np.argmax(y, axis=1))


# ============================================================================
# NETWORK 2: PHI-WORLD NEURAL NETWORK
# ============================================================================
class PhiWorldNN:
    """
    Neural network where weights evolve via phi-world field dynamics.
    
    Instead of gradient descent, weights are treated as a 2D field
    that evolves according to the same equations as best.py:
    - Laplacian diffusion (smoothness)
    - Nonlinear potential (stable attractors)
    - Damped wave equation (momentum without explosion)
    
    The "gradient" from backprop becomes an external force on the field.
    """
    
    def __init__(self, layer_sizes, dt=0.1, damping=0.01, 
                 tension=1.0, pot_lin=0.1, pot_cub=0.01,
                 grad_coupling=1.0):
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes) - 1
        
        # Phi-world parameters
        self.dt = dt
        self.damping = damping
        self.tension = tension
        self.pot_lin = pot_lin
        self.pot_cub = pot_cub
        self.grad_coupling = grad_coupling
        
        # Initialize weights as fields
        self.weights = []
        self.weights_old = []  # For wave equation (velocity)
        self.biases = []
        self.biases_old = []
        
        for i in range(self.num_layers):
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2.0 / layer_sizes[i])
            b = np.zeros(layer_sizes[i+1])
            self.weights.append(w.astype(np.float32))
            self.weights_old.append(w.copy().astype(np.float32))
            self.biases.append(b.astype(np.float32))
            self.biases_old.append(b.copy().astype(np.float32))
        
        # 2D Laplacian kernel for weight matrices
        self.kern_2d = np.array([
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]
        ], dtype=np.float32)
    
    def laplacian_2d(self, field):
        """Compute 2D Laplacian of weight matrix"""
        # Pad to handle boundaries
        padded = np.pad(field, 1, mode='reflect')
        result = np.zeros_like(field)
        
        for i in range(field.shape[0]):
            for j in range(field.shape[1]):
                result[i, j] = (
                    padded[i, j+1] + padded[i+2, j+1] + 
                    padded[i+1, j] + padded[i+1, j+2] - 
                    4 * padded[i+1, j+1]
                )
        return result
    
    def phi_step(self, field, field_old, external_force):
        """
        One step of phi-world dynamics on a weight matrix.
        
        field: current weights
        field_old: previous weights (for velocity)
        external_force: gradient from backprop (negative = descent)
        """
        # Laplacian (diffusion/smoothing)
        lap = self.laplacian_2d(field)
        
        # Nonlinear potential: V(phi) = -lin*phi^2/2 + cub*phi^4/4
        # V'(phi) = -lin*phi + cub*phi^3
        Vp = -self.pot_lin * field + self.pot_cub * field**3
        
        # Wave speed modulation (from best.py)
        c2 = 1.0 / (1.0 + self.tension * field**2 + 1e-6)
        
        # Acceleration = c² * Laplacian - V'(phi) + external_force
        acc = c2 * lap - Vp + self.grad_coupling * external_force
        
        # Damped wave equation update
        vel = field - field_old
        field_new = field + (1 - self.damping * self.dt) * vel + self.dt**2 * acc
        
        return field_new, field.copy()
    
    def forward(self, X):
        """Standard forward pass"""
        self.activations = [X]
        self.z_values = []
        
        for i in range(self.num_layers):
            z = self.activations[-1] @ self.weights[i] + self.biases[i]
            self.z_values.append(z)
            
            if i < self.num_layers - 1:
                a = relu(z)
            else:
                a = softmax(z)
            self.activations.append(a)
        
        return self.activations[-1]
    
    def backward(self, y_true):
        """Compute gradients (but don't apply them - they become forces)"""
        m = y_true.shape[0]
        delta = self.activations[-1] - y_true
        
        grad_w = []
        grad_b = []
        
        for i in range(self.num_layers - 1, -1, -1):
            gw = self.activations[i].T @ delta / m
            gb = np.mean(delta, axis=0)
            grad_w.insert(0, gw)
            grad_b.insert(0, gb)
            
            if i > 0:
                delta = (delta @ self.weights[i].T) * relu_grad(self.z_values[i-1])
        
        return grad_w, grad_b
    
    def update(self, grad_w, grad_b):
        """Update weights using phi-world dynamics instead of Adam"""
        for i in range(self.num_layers):
            # Weights evolve via field dynamics
            # Negative gradient = force toward lower loss
            external_force = -grad_w[i]
            
            self.weights[i], self.weights_old[i] = self.phi_step(
                self.weights[i], self.weights_old[i], external_force
            )
            
            # Biases: simple gradient descent (1D, no spatial structure)
            self.biases[i] -= 0.01 * grad_b[i]
    
    def train_step(self, X, y):
        pred = self.forward(X)
        loss = -np.mean(np.sum(y * np.log(pred + 1e-10), axis=1))
        grad_w, grad_b = self.backward(y)
        self.update(grad_w, grad_b)
        return loss
    
    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)
    
    def accuracy(self, X, y):
        return np.mean(self.predict(X) == np.argmax(y, axis=1))
    
    def get_weight_energy(self):
        """Compute total 'energy' in weight fields (for monitoring)"""
        energy = 0
        for w, w_old in zip(self.weights, self.weights_old):
            # Kinetic energy (velocity squared)
            vel = w - w_old
            energy += np.sum(vel**2)
            # Potential energy
            energy += np.sum(self.pot_lin * w**2 / 2 - self.pot_cub * w**4 / 4)
        return energy


# ============================================================================
# NETWORK 3: CRITICAL PHI-NET (tuned to edge of chaos)
# ============================================================================
class CriticalPhiNet(PhiWorldNN):
    """
    Phi-world network tuned to operate at criticality.
    Based on the fragility analysis: critical systems learn best.
    
    Key insight: The 7-harmonic regime was fragile but responsive.
    We want weights to be at the edge - responsive but not chaotic.
    """
    
    def __init__(self, layer_sizes):
        # Parameters tuned for criticality:
        # - Higher tension (stronger attractors)
        # - Lower damping (more momentum, longer memory)
        # - Balanced potential (neither too stable nor too chaotic)
        super().__init__(
            layer_sizes,
            dt=0.15,           # Slightly faster dynamics
            damping=0.005,     # Less damping = more critical
            tension=2.0,       # Stronger nonlinearity
            pot_lin=0.2,       # Moderate linear term
            pot_cub=0.05,      # Moderate cubic term
            grad_coupling=2.0  # Stronger response to gradients
        )
        
        # Track criticality metrics
        self.weight_variances = []
    
    def train_step(self, X, y):
        loss = super().train_step(X, y)
        
        # Monitor weight variance (indicator of criticality)
        var = np.mean([np.var(w) for w in self.weights])
        self.weight_variances.append(var)
        
        return loss


# ============================================================================
# EXPERIMENT RUNNER
# ============================================================================
def run_experiment(n_train=500, n_test=200, epochs=100, noise=0.2):
    """
    Compare all three networks on the same task.
    """
    print("="*70)
    print("PHI-WORLD NEURAL NETWORK EXPERIMENT")
    print("="*70)
    print(f"\nTask: Synthetic digit classification (8x8 images)")
    print(f"Training samples: {n_train}")
    print(f"Test samples: {n_test}")
    print(f"Noise level: {noise}")
    print(f"Epochs: {epochs}")
    
    # Generate data
    print("\n[1/4] Generating data...")
    X_train, y_train = generate_mnist_like_data(n_train, noise=noise)
    X_test, y_test = generate_mnist_like_data(n_test, noise=noise)
    y_train_oh = one_hot(y_train)
    y_test_oh = one_hot(y_test)
    
    # Normalize
    X_train = X_train / (np.max(np.abs(X_train)) + 1e-10)
    X_test = X_test / (np.max(np.abs(X_test)) + 1e-10)
    
    print(f"    X_train shape: {X_train.shape}")
    print(f"    X_test shape: {X_test.shape}")
    
    # Initialize networks
    print("\n[2/4] Initializing networks...")
    layer_sizes = [64, 32, 16, 10]  # Input -> Hidden -> Hidden -> Output
    
    net_standard = StandardMLP(layer_sizes, lr=0.01)
    net_phi = PhiWorldNN(layer_sizes)
    net_critical = CriticalPhiNet(layer_sizes)
    
    print(f"    Architecture: {layer_sizes}")
    print(f"    Standard MLP: Adam optimizer, lr=0.01")
    print(f"    Phi-Net: Field dynamics, dt=0.1, damping=0.01")
    print(f"    Critical Phi-Net: Edge-of-chaos dynamics")
    
    # Training
    print("\n[3/4] Training...")
    
    results = {
        'standard': {'loss': [], 'train_acc': [], 'test_acc': []},
        'phi': {'loss': [], 'train_acc': [], 'test_acc': [], 'energy': []},
        'critical': {'loss': [], 'train_acc': [], 'test_acc': [], 'energy': []}
    }
    
    batch_size = 32
    n_batches = n_train // batch_size
    
    for epoch in range(epochs):
        # Shuffle
        idx = np.random.permutation(n_train)
        X_shuf = X_train[idx]
        y_shuf = y_train_oh[idx]
        
        epoch_loss = {'standard': 0, 'phi': 0, 'critical': 0}
        
        for b in range(n_batches):
            X_batch = X_shuf[b*batch_size:(b+1)*batch_size]
            y_batch = y_shuf[b*batch_size:(b+1)*batch_size]
            
            epoch_loss['standard'] += net_standard.train_step(X_batch, y_batch)
            epoch_loss['phi'] += net_phi.train_step(X_batch, y_batch)
            epoch_loss['critical'] += net_critical.train_step(X_batch, y_batch)
        
        # Record metrics
        for name, net, res in [('standard', net_standard, results['standard']),
                                ('phi', net_phi, results['phi']),
                                ('critical', net_critical, results['critical'])]:
            res['loss'].append(epoch_loss[name] / n_batches)
            res['train_acc'].append(net.accuracy(X_train, y_train_oh))
            res['test_acc'].append(net.accuracy(X_test, y_test_oh))
            
            if hasattr(net, 'get_weight_energy'):
                res['energy'].append(net.get_weight_energy())
        
        if (epoch + 1) % 20 == 0:
            print(f"    Epoch {epoch+1:3d}: "
                  f"Standard={results['standard']['test_acc'][-1]:.3f}, "
                  f"Phi={results['phi']['test_acc'][-1]:.3f}, "
                  f"Critical={results['critical']['test_acc'][-1]:.3f}")
    
    # Final results
    print("\n[4/4] Final Results...")
    print("\n" + "-"*50)
    print("FINAL TEST ACCURACY")
    print("-"*50)
    print(f"    Standard MLP (Adam):  {results['standard']['test_acc'][-1]:.4f}")
    print(f"    Phi-World Net:        {results['phi']['test_acc'][-1]:.4f}")
    print(f"    Critical Phi-Net:     {results['critical']['test_acc'][-1]:.4f}")
    
    # Robustness test: add extra noise to test set
    print("\n" + "-"*50)
    print("ROBUSTNESS TEST (2x noise on test set)")
    print("-"*50)
    X_test_noisy = X_test + np.random.randn(*X_test.shape) * noise
    X_test_noisy = X_test_noisy / (np.max(np.abs(X_test_noisy)) + 1e-10)
    
    print(f"    Standard MLP (Adam):  {net_standard.accuracy(X_test_noisy, y_test_oh):.4f}")
    print(f"    Phi-World Net:        {net_phi.accuracy(X_test_noisy, y_test_oh):.4f}")
    print(f"    Critical Phi-Net:     {net_critical.accuracy(X_test_noisy, y_test_oh):.4f}")
    
    # Low-data test
    print("\n" + "-"*50)
    print("LOW-DATA TEST (train on 10% of data)")
    print("-"*50)
    
    net_std_small = StandardMLP(layer_sizes, lr=0.01)
    net_phi_small = PhiWorldNN(layer_sizes)
    net_crit_small = CriticalPhiNet(layer_sizes)
    
    small_n = n_train // 10
    X_small = X_train[:small_n]
    y_small = y_train_oh[:small_n]
    
    for _ in range(epochs):
        idx = np.random.permutation(small_n)
        for b in range(small_n // batch_size):
            X_b = X_small[idx[b*batch_size:(b+1)*batch_size]]
            y_b = y_small[idx[b*batch_size:(b+1)*batch_size]]
            if len(X_b) > 0:
                net_std_small.train_step(X_b, y_b)
                net_phi_small.train_step(X_b, y_b)
                net_crit_small.train_step(X_b, y_b)
    
    print(f"    Standard MLP (Adam):  {net_std_small.accuracy(X_test, y_test_oh):.4f}")
    print(f"    Phi-World Net:        {net_phi_small.accuracy(X_test, y_test_oh):.4f}")
    print(f"    Critical Phi-Net:     {net_crit_small.accuracy(X_test, y_test_oh):.4f}")
    
    return results, (net_standard, net_phi, net_critical)


def visualize_results(results, save_path='phi_nn_comparison.png'):
    """Visualize the comparison"""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    epochs = range(1, len(results['standard']['loss']) + 1)
    
    # Loss curves
    ax1 = axes[0, 0]
    ax1.plot(epochs, results['standard']['loss'], 'b-', label='Standard (Adam)', linewidth=2)
    ax1.plot(epochs, results['phi']['loss'], 'r-', label='Phi-World', linewidth=2)
    ax1.plot(epochs, results['critical']['loss'], 'g-', label='Critical Phi', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Train accuracy
    ax2 = axes[0, 1]
    ax2.plot(epochs, results['standard']['train_acc'], 'b-', label='Standard', linewidth=2)
    ax2.plot(epochs, results['phi']['train_acc'], 'r-', label='Phi-World', linewidth=2)
    ax2.plot(epochs, results['critical']['train_acc'], 'g-', label='Critical', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Test accuracy
    ax3 = axes[0, 2]
    ax3.plot(epochs, results['standard']['test_acc'], 'b-', label='Standard', linewidth=2)
    ax3.plot(epochs, results['phi']['test_acc'], 'r-', label='Phi-World', linewidth=2)
    ax3.plot(epochs, results['critical']['test_acc'], 'g-', label='Critical', linewidth=2)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Accuracy')
    ax3.set_title('Test Accuracy')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Weight energy (phi networks)
    ax4 = axes[1, 0]
    if results['phi']['energy']:
        ax4.plot(epochs, results['phi']['energy'], 'r-', label='Phi-World', linewidth=2)
        ax4.plot(epochs, results['critical']['energy'], 'g-', label='Critical', linewidth=2)
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Field Energy')
        ax4.set_title('Weight Field Energy')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    # Generalization gap
    ax5 = axes[1, 1]
    gap_std = np.array(results['standard']['train_acc']) - np.array(results['standard']['test_acc'])
    gap_phi = np.array(results['phi']['train_acc']) - np.array(results['phi']['test_acc'])
    gap_crit = np.array(results['critical']['train_acc']) - np.array(results['critical']['test_acc'])
    ax5.plot(epochs, gap_std, 'b-', label='Standard', linewidth=2)
    ax5.plot(epochs, gap_phi, 'r-', label='Phi-World', linewidth=2)
    ax5.plot(epochs, gap_crit, 'g-', label='Critical', linewidth=2)
    ax5.set_xlabel('Epoch')
    ax5.set_ylabel('Train - Test Accuracy')
    ax5.set_title('Generalization Gap\n(lower = better generalization)')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Summary text
    ax6 = axes[1, 2]
    ax6.axis('off')
    
    final_std = results['standard']['test_acc'][-1]
    final_phi = results['phi']['test_acc'][-1]
    final_crit = results['critical']['test_acc'][-1]
    
    winner = "Standard" if final_std >= max(final_phi, final_crit) else \
             "Phi-World" if final_phi >= final_crit else "Critical"
    
    summary = f"""
    ╔════════════════════════════════════════╗
    ║     PHI-WORLD NN EXPERIMENT RESULTS    ║
    ╠════════════════════════════════════════╣
    ║                                        ║
    ║  FINAL TEST ACCURACY:                  ║
    ║    Standard (Adam):  {final_std:.4f}            ║
    ║    Phi-World Net:    {final_phi:.4f}            ║
    ║    Critical Phi:     {final_crit:.4f}            ║
    ║                                        ║
    ║  WINNER: {winner:^20s}       ║
    ║                                        ║
    ║  GENERALIZATION (final gap):           ║
    ║    Standard: {gap_std[-1]:.4f}                  ║
    ║    Phi-World: {gap_phi[-1]:.4f}                  ║
    ║    Critical: {gap_crit[-1]:.4f}                  ║
    ║                                        ║
    ╚════════════════════════════════════════╝
    """
    
    ax6.text(0.1, 0.5, summary, transform=ax6.transAxes,
             fontfamily='monospace', fontsize=10, verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('STANDARD NN vs PHI-WORLD NN: Head-to-Head Comparison',
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved: {save_path}")
    
    return fig


# ============================================================================
# MAIN
# ============================================================================
if __name__ == '__main__':
    # Run experiment
    results, networks = run_experiment(
        n_train=500,
        n_test=200, 
        epochs=100,
        noise=0.2
    )
    
    # Visualize
    fig = visualize_results(results)
    
    # Analysis
    print("\n" + "="*70)
    print("ANALYSIS")
    print("="*70)
    
    print("""
The experiment tests whether phi-world dynamics offer any advantage
over standard gradient descent for neural network training.

Key observations to look for:
1. Does Phi-World achieve comparable accuracy?
2. Does Phi-World generalize better (smaller train-test gap)?
3. Does Phi-World learn faster or slower?
4. Does the Critical version show different dynamics?

If Phi-World shows advantages in generalization or robustness,
it supports the hypothesis that energy-minimizing field dynamics
naturally produce robust, learnable representations.

If Standard Adam dominates, then phi-world dynamics may be
interesting physics but not useful for practical AI.
    """)
    
    plt.show()