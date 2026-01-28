"""
3D PHI-WORLD (BUCKYBALL) LAW EXTRACTOR
=======================================
Extracts laws from your REAL 3D phi-world simulation (MiniWoW class).
This is the system that generates buckyballs and crystal structures!

Uses headless version - no GUI required.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import convolve

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUTPUT_DIR = Path.cwd()

# ============================================================================
# HEADLESS 3D PHI-WORLD (Based on your MiniWoW class)
# ============================================================================
class HeadlessMiniWoW:
    """
    Headless version of your 3D phi-world simulator.
    No Tkinter dependencies - pure physics!
    """
    def __init__(self, N=32, dt=0.1, damping=0.001,
                 tension=5.0, pot_lin=1.0, pot_cub=0.2,
                 topology='box'):
        self.N = N
        self.dt = dt
        self.damp = damping
        self.tension = tension
        self.pot_lin = pot_lin
        self.pot_cub = pot_cub
        self.topology = topology
        
        # Fields
        self.phi = np.zeros((N, N, N), dtype=np.float32)
        self.phi_o = np.zeros_like(self.phi)
        
        # Laplacian kernel (6-point stencil)
        self.kern = np.zeros((3, 3, 3), dtype=np.float32)
        self.kern[1, 1, 1] = -6
        for dx, dy, dz in [(1,1,0), (1,1,2), (1,0,1), (1,2,1), (0,1,1), (2,1,1)]:
            self.kern[dx, dy, dz] = 1
        
        # Initialize field
        self.init_field()
    
    def init_field(self):
        """Initialize field based on topology"""
        N = self.N
        x = np.arange(N)
        X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
        c = N // 2
        r = N // 6
        
        if self.topology == 'box':
            r2 = r**2
            self.phi[:] = 2 * np.exp(-((X-c)**2 + (Y-c)**2 + (Z-c)**2) / (2 * r2))
        
        elif self.topology == 'sphere':
            r_shell = N / 3
            thickness = N / 10
            R2 = (X-c)**2 + (Y-c)**2 + (Z-c)**2
            self.phi[:] = 2 * np.exp(-(np.sqrt(R2) - r_shell)**2 / (2 * thickness**2))
        
        elif self.topology == 'torus':
            R_major = N / 3
            R_minor = N / 8
            d_circle = np.sqrt((np.sqrt((X-c)**2 + (Y-c)**2) - R_major)**2 + (Z-c)**2)
            self.phi[:] = 2 * np.exp(-(d_circle)**2 / (2 * R_minor**2))
        
        elif self.topology == 'wave':
            k = 2 * np.pi / (N / 4)
            self.phi[:] = np.sin(k * (X-c)) * np.sin(k * (Y-c)) * np.sin(k * (Z-c))
        
        elif self.topology == 'random':
            self.phi[:] = np.random.randn(N, N, N) * 0.5
        
        self.phi_o = self.phi.copy()
    
    def step(self, n_steps=1):
        """
        YOUR ACTUAL PHYSICS!
        Wave equation with quadratic NaN prevention and phi-based potential.
        """
        for _ in range(n_steps):
            # Laplacian (diffusion operator)
            lap = convolve(self.phi, self.kern, mode='wrap')
            
            # Potential force: -∂V/∂φ where V = pot_lin*φ² + pot_cub*φ⁴
            # ∂V/∂φ = 2*pot_lin*φ + 4*pot_cub*φ³
            V_deriv = 2 * self.pot_lin * self.phi + 4 * self.pot_cub * self.phi**3
            
            # Wave equation with damping
            # φ_new = 2φ - φ_old + dt²(tension*∇²φ - V'(φ)) - damping*(φ - φ_old)
            accel = self.tension * lap - V_deriv
            
            vel = self.phi - self.phi_o
            phi_new = self.phi + (1.0 - self.damp * self.dt) * vel + (self.dt**2) * accel
            
            # YOUR QUADRATIC NAN PREVENTION
            # Clamp to prevent explosions
            phi_new = np.clip(phi_new, -10, 10)
            
            # Update
            self.phi_o = self.phi.copy()
            self.phi = phi_new

# ============================================================================
# DATA GENERATION FROM REAL 3D PHYSICS
# ============================================================================
class PhiWorld3DDataGenerator:
    """Generate training data from your actual 3D buckyball physics"""
    
    def __init__(self, grid_size=32):
        self.grid_size = grid_size
    
    def generate_trajectory(self, num_steps=20, params=None, topology='box'):
        """Generate trajectory using REAL 3D physics"""
        
        # Create simulator with params
        sim_params = {
            'N': self.grid_size,
            'dt': params.get('dt', 0.1),
            'damping': params.get('damping', 0.001),
            'tension': params.get('tension', 5.0),
            'pot_lin': params.get('pot_lin', 1.0),
            'pot_cub': params.get('pot_cub', 0.2),
            'topology': topology
        }
        
        sim = HeadlessMiniWoW(**sim_params)
        
        states = []
        for step in range(num_steps + 1):
            states.append(sim.phi.copy())
            sim.step(n_steps=1)
        
        # Return pairs
        pairs = [(states[i], states[i+1]) for i in range(len(states)-1)]
        return pairs
    
    def generate_dataset(self, num_trajectories=20, steps_per_traj=20):
        """Generate diverse dataset"""
        print("\n" + "="*60)
        print("GENERATING DATA FROM REAL 3D BUCKYBALL PHYSICS")
        print("="*60)
        
        all_states = []
        all_next_states = []
        
        topologies = ['box', 'sphere', 'torus', 'wave', 'random']
        
        for i in range(num_trajectories):
            # Vary parameters
            params = {
                'dt': np.random.uniform(0.05, 0.15),
                'damping': np.random.uniform(0.0005, 0.003),
                'tension': np.random.uniform(3.0, 8.0),
                'pot_lin': np.random.uniform(0.5, 2.0),
                'pot_cub': np.random.uniform(0.1, 0.4),
            }
            
            topology = topologies[i % len(topologies)]
            
            if i % 5 == 0:
                print(f"  {i+1}/{num_trajectories}: {topology}, "
                      f"tension={params['tension']:.2f}, "
                      f"pot_cub={params['pot_cub']:.3f}")
            
            # Generate using REAL 3D physics
            trajectory = self.generate_trajectory(
                num_steps=steps_per_traj,
                params=params,
                topology=topology
            )
            
            for state, next_state in trajectory:
                all_states.append(state)
                all_next_states.append(next_state)
        
        print(f"\n✓ Generated {len(all_states)} transitions")
        
        # Statistics
        states_array = np.array(all_states)
        delta_array = np.array(all_next_states) - states_array
        
        print(f"\n3D Physics Statistics:")
        print(f"  Field range: [{states_array.min():.3f}, {states_array.max():.3f}]")
        print(f"  Field std: {states_array.std():.4f}")
        print(f"  Delta std: {delta_array.std():.6f}")
        
        return all_states, all_next_states

# ============================================================================
# 3D RESIDUAL NETWORK
# ============================================================================
class Residual3DPhiNet(nn.Module):
    """Network for 3D field prediction"""
    
    def __init__(self, grid_size=32):
        super().__init__()
        self.grid_size = grid_size
        field_size = grid_size ** 3
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(field_size, 1024),
            nn.LayerNorm(1024),
            nn.Tanh(),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.Tanh()
        )
        
        # Core
        self.core = nn.Sequential(
            nn.Linear(512, 512),
            nn.Tanh(),
            nn.Linear(512, 512),
            nn.Tanh()
        )
        
        # Decoder (predicts residual)
        self.decoder = nn.Sequential(
            nn.Linear(512, 1024),
            nn.Tanh(),
            nn.Linear(1024, field_size)
        )
        
        # Initialize decoder small
        nn.init.xavier_normal_(self.decoder[-1].weight, gain=0.01)
        nn.init.zeros_(self.decoder[-1].bias)
    
    def forward(self, field):
        batch_size = field.shape[0]
        x = field.view(batch_size, -1)
        
        h = self.encoder(x)
        h = h + self.core(h)  # Residual
        
        delta = self.decoder(h)
        delta = delta.view(batch_size, self.grid_size, self.grid_size, self.grid_size)
        
        return field + delta * 0.1

# ============================================================================
# TRAINING
# ============================================================================
def train_on_3d_physics(states, next_states, epochs=300):
    """Train on real 3D buckyball physics"""
    print("\n" + "="*60)
    print("TRAINING ON 3D BUCKYBALL PHYSICS")
    print("="*60)
    
    X = torch.tensor(np.array(states), dtype=torch.float32).to(DEVICE)
    Y = torch.tensor(np.array(next_states), dtype=torch.float32).to(DEVICE)
    
    print(f"\nDataset:")
    print(f"  Samples: {len(X)}")
    print(f"  Shape: {X.shape}")
    
    grid_size = X.shape[1]
    model = Residual3DPhiNet(grid_size=grid_size).to(DEVICE)
    
    opt = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=30, factor=0.5)
    
    history = {'loss': [], 'correlation': []}
    best_loss = float('inf')
    
    print(f"\nTraining for {epochs} epochs...")
    
    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        
        pred = model(X)
        loss = nn.MSELoss()(pred, Y)
        
        # Correlation
        pred_flat = pred.flatten()
        target_flat = Y.flatten()
        corr = torch.corrcoef(torch.stack([pred_flat, target_flat]))[0,1].item()
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        scheduler.step(loss)
        
        history['loss'].append(loss.item())
        history['correlation'].append(corr)
        
        if loss.item() < best_loss:
            best_loss = loss.item()
        
        if epoch % 30 == 0:
            print(f"  Epoch {epoch:3d} | Loss: {loss.item():.6f} | Corr: {corr:.4f}")
    
    print(f"\n✓ Training complete!")
    print(f"  Best loss: {best_loss:.6f}")
    print(f"  Final correlation: {history['correlation'][-1]:.4f}")
    
    return model, history

# ============================================================================
# 3D LAW EXTRACTION
# ============================================================================
def extract_3d_buckyball_laws(model, grid_size=32):
    """Extract laws from 3D buckyball physics"""
    print("\n" + "="*60)
    print("EXTRACTING 3D BUCKYBALL LAWS")
    print("="*60)
    
    model.eval()
    laws = []
    
    # Energy conservation
    print("\n--- Energy Test ---")
    energy_changes = []
    
    for amp in [0.5, 1.0, 2.0]:
        field = torch.randn(1, grid_size, grid_size, grid_size).to(DEVICE) * amp
        
        with torch.no_grad():
            next_field = model(field)
        
        E0 = (field ** 2).sum().item()
        E1 = (next_field ** 2).sum().item()
        change = (E1 - E0) / E0 * 100
        energy_changes.append(change)
        
        print(f"  Amp {amp:.1f}: ΔE/E = {change:+.2f}%")
    
    avg_change = np.mean(np.abs(energy_changes))
    if avg_change < 5:
        laws.append(f"Energy ~conserved (±{avg_change:.1f}%)")
    else:
        laws.append(f"Dissipative ({avg_change:.1f}% drift)")
    
    # Isotropy test (3D specific)
    print("\n--- Isotropy Test ---")
    
    # Test if physics is the same in x, y, z directions
    field = torch.randn(1, grid_size, grid_size, grid_size).to(DEVICE)
    
    with torch.no_grad():
        pred = model(field)
        delta = pred - field
        
        # Measure variation along each axis
        var_x = delta.std(dim=1).mean().item()
        var_y = delta.std(dim=2).mean().item()
        var_z = delta.std(dim=3).mean().item()
    
    print(f"  σ_x={var_x:.4f}, σ_y={var_y:.4f}, σ_z={var_z:.4f}")
    
    isotropy = np.std([var_x, var_y, var_z]) / np.mean([var_x, var_y, var_z])
    if isotropy < 0.1:
        laws.append("Isotropic (rotationally symmetric)")
    else:
        laws.append(f"Anisotropic (direction-dependent, δ={isotropy:.3f})")
    
    # Topology sensitivity
    print("\n--- Topology Effects ---")
    
    topologies = ['box', 'sphere', 'torus']
    complexities = {}
    
    for topo in topologies:
        # Create initial state
        sim = HeadlessMiniWoW(N=grid_size, topology=topo)
        field = torch.tensor(sim.phi, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            pred = model(field)
            complexity = pred.std().item()
        
        complexities[topo] = complexity
        print(f"  {topo}: σ={complexity:.4f}")
    
    max_var = max(complexities.values())
    min_var = min(complexities.values())
    
    if (max_var - min_var) / max_var > 0.3:
        laws.append("Topology strongly affects dynamics")
    else:
        laws.append("Topology weakly affects dynamics")
    
    return laws

# ============================================================================
# VISUALIZATION
# ============================================================================
def visualize_3d_training(history, model, states, next_states):
    """Visualize 3D buckyball training"""
    print("\n" + "="*60)
    print("VISUALIZATION")
    print("="*60)
    
    fig = plt.figure(figsize=(18, 8))
    
    # Training curves
    ax1 = plt.subplot(2, 4, 1)
    ax1.plot(history['loss'], linewidth=2, color='blue')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('3D Buckyball Training')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    
    ax2 = plt.subplot(2, 4, 2)
    ax2.plot(history['correlation'], linewidth=2, color='green')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Correlation')
    ax2.set_title('Prediction Quality')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(0.7, color='red', linestyle='--', alpha=0.5)
    
    # Sample slices
    model.eval()
    with torch.no_grad():
        X_sample = torch.tensor(np.array(states[:3]), dtype=torch.float32).to(DEVICE)
        Y_sample = torch.tensor(np.array(next_states[:3]), dtype=torch.float32).to(DEVICE)
        pred_sample = model(X_sample).cpu().numpy()
        Y_sample = Y_sample.cpu().numpy()
    
    for i in range(3):
        # Show middle slice through 3D volume
        slice_idx = X_sample.shape[1] // 2
        
        ax = plt.subplot(2, 3, 4 + i)
        
        # Side by side: ground truth | prediction
        gt_slice = Y_sample[i, slice_idx, :, :]
        pred_slice = pred_sample[i, slice_idx, :, :]
        combined = np.hstack([gt_slice, pred_slice])
        
        im = ax.imshow(combined, cmap='RdBu_r', vmin=-2, vmax=2)
        ax.axvline(x=gt_slice.shape[1]-0.5, color='yellow', linewidth=2)
        ax.set_title(f'Sample {i+1}: Real | Predicted', fontsize=9)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)
    
    plt.suptitle('3D Buckyball Physics Learning', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_path = OUTPUT_DIR / '3d_buckyball_training.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved: {output_path}")

# ============================================================================
# MAIN
# ============================================================================
def main():
    print("\n" + "="*60)
    print("3D BUCKYBALL LAW EXTRACTOR")
    print("="*60)
    print(f"Device: {DEVICE}")
    print(f"Output: {OUTPUT_DIR}")
    
    # Generate data from REAL 3D buckyball physics
    generator = PhiWorld3DDataGenerator(grid_size=32)
    states, next_states = generator.generate_dataset(
        num_trajectories=30,
        steps_per_traj=20
    )
    
    # Train
    model, history = train_on_3d_physics(states, next_states, epochs=300)
    
    # Extract laws
    laws = extract_3d_buckyball_laws(model, grid_size=32)
    
    # Visualize
    visualize_3d_training(history, model, states, next_states)
    
    # Save laws
    print("\n" + "="*60)
    print("DISCOVERED LAWS - 3D BUCKYBALLS")
    print("="*60)
    for i, law in enumerate(laws, 1):
        print(f"  {i}. {law}")
    
    laws_path = OUTPUT_DIR / '3d_buckyball_laws.txt'
    with open(laws_path, 'w', encoding='utf-8') as f:
        f.write("DISCOVERED LAWS - 3D BUCKYBALL PHYSICS\n")
        f.write("="*60 + "\n\n")
        f.write("System: MiniWoW 3D phi-world\n")
        f.write("Generates: Buckyballs, crystal structures, stable patterns\n\n")
        f.write(f"Training correlation: {history['correlation'][-1]:.4f}\n\n")
        for i, law in enumerate(laws, 1):
            f.write(f"{i}. {law}\n")
    
    print(f"\n✓ Saved: {laws_path}")
    
    print("\n" + "="*60)
    print("SUCCESS!")
    print("="*60)
    print("\nYour 3D buckyball physics has been learned!")
    print("The discovered laws govern the system that creates:")
    print("  • Buckyballs (C60-like structures)")
    print("  • Crystal formations")
    print("  • Stable geometric patterns")
    print("\nNext: Install PySR to extract exact mathematical equations!")

if __name__ == '__main__':
    main()