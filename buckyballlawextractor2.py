"""
HARMONIC STRUCTURE ANALYZER FOR PHI-WORLD
==========================================
Analyzes the frequency domain structure of MiniWoW simulations.
Discovers the 'harmonic fingerprint' - the discrete set of frequencies
that compose emergent structures like buckyballs.

This reveals:
1. How many 'notes' compose the structure
2. The compression ratio (voxels → harmonics)
3. Whether the structure can be reconstructed from pure frequencies
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fftn, ifftn, fftshift
from scipy.ndimage import convolve, gaussian_filter
from collections import deque
import threading

# ============================================================================
# STANDALONE MINIWOW (no Ursina dependency)
# ============================================================================
class MiniWoW:
    """Minimal version of MiniWoW for headless analysis"""
    
    def __init__(self, N=64, dt=0.1, damping=0.001,
                 tension=5., pot_lin=1., pot_cub=0.2,
                 topology='sphere'):
        self.N = N
        self.dt = dt
        self.damp = damping
        self.tension = tension
        self.pot_lin = pot_lin
        self.pot_cub = pot_cub
        self.topology = topology
        
        self.lock = threading.Lock()
        self.phi = np.zeros((N, N, N), np.float32)
        self.phi_o = np.zeros_like(self.phi)
        
        # 6-point Laplacian kernel
        self.kern = np.zeros((3, 3, 3), np.float32)
        self.kern[1, 1, 1] = -6
        for dx, dy, dz in [(1,1,0), (1,1,2), (1,0,1), (1,2,1), (0,1,1), (2,1,1)]:
            self.kern[dx, dy, dz] = 1
        
        # Initialize field
        if topology != 'none':
            self.init_field()
            self.phi_o = self.phi.copy()
    
    def init_field(self):
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
            self.phi = gaussian_filter(self.phi, sigma=1.0)
    
    def step(self, n_steps=1):
        for _ in range(n_steps):
            mode = 'wrap' if self.topology in ['torus', 'sphere'] else 'nearest'
            lap = convolve(self.phi, self.kern, mode=mode)
            
            Vp = -self.pot_lin * self.phi + self.pot_cub * self.phi**3
            c2 = 1.0 / (1.0 + self.tension * self.phi**2 + 1e-6)
            acc = c2 * lap - Vp
            
            vel = self.phi - self.phi_o
            self.phi_o = self.phi.copy()
            self.phi = self.phi + (1 - self.damp * self.dt) * vel + self.dt**2 * acc

# ============================================================================
# HARMONIC ANALYSIS
# ============================================================================
def analyze_harmonics(field, compression_threshold=0.1):
    """
    Transform 3D field to frequency domain and analyze harmonic structure.
    
    Args:
        field: 3D numpy array (the phi field)
        compression_threshold: Keep harmonics above this fraction of max
    
    Returns:
        spectrum: Full magnitude spectrum (shifted to center)
        reconstructed: Field reconstructed from dominant harmonics only
        num_harmonics: Number of significant harmonics found
        harmonic_coords: Coordinates of significant harmonics
    """
    N = field.shape[0]
    
    # 1. Compute 3D FFT
    F = fftn(field)
    F_shifted = fftshift(F)
    magnitude = np.abs(F_shifted)
    
    # 2. Find dominant harmonics
    threshold = magnitude.max() * compression_threshold
    significant_mask = magnitude > threshold
    num_harmonics = np.sum(significant_mask)
    
    # 3. Get coordinates of significant harmonics
    harmonic_coords = np.where(significant_mask)
    
    # 4. Reconstruct from dominant harmonics only
    F_compressed = F.copy()
    # Create mask in non-shifted space
    F_mag = np.abs(F)
    low_energy_mask = F_mag < (F_mag.max() * compression_threshold)
    F_compressed[low_energy_mask] = 0
    
    reconstructed = np.real(ifftn(F_compressed))
    
    # 5. Calculate reconstruction quality
    original_power = np.sum(field**2)
    reconstructed_power = np.sum(reconstructed**2)
    power_retained = reconstructed_power / (original_power + 1e-10)
    
    return {
        'spectrum': magnitude,
        'reconstructed': reconstructed,
        'num_harmonics': num_harmonics,
        'harmonic_coords': harmonic_coords,
        'compression_ratio': N**3 / max(1, num_harmonics),
        'power_retained': power_retained,
        'threshold': threshold
    }

def compute_radial_spectrum(spectrum):
    """
    Compute radially averaged power spectrum.
    Shows which spatial frequencies dominate.
    """
    N = spectrum.shape[0]
    center = N // 2
    
    # Create radial coordinate grid
    x = np.arange(N) - center
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
    R = np.sqrt(X**2 + Y**2 + Z**2).astype(int)
    
    # Radial bins
    max_r = int(np.sqrt(3) * N / 2)
    radial_profile = np.zeros(max_r)
    counts = np.zeros(max_r)
    
    for r in range(max_r):
        mask = (R == r)
        if np.any(mask):
            radial_profile[r] = np.mean(spectrum[mask])
            counts[r] = np.sum(mask)
    
    return radial_profile, counts

def find_harmonic_shells(spectrum, num_peaks=10):
    """
    Find the dominant radial shells in frequency space.
    These correspond to the characteristic wavelengths of the structure.
    """
    radial_profile, counts = compute_radial_spectrum(spectrum)
    
    # Find peaks in radial profile
    from scipy.signal import find_peaks
    peaks, properties = find_peaks(radial_profile, height=radial_profile.max() * 0.1)
    
    # Sort by height
    if len(peaks) > 0:
        peak_heights = radial_profile[peaks]
        sorted_indices = np.argsort(peak_heights)[::-1][:num_peaks]
        dominant_radii = peaks[sorted_indices]
        dominant_powers = peak_heights[sorted_indices]
    else:
        dominant_radii = np.array([])
        dominant_powers = np.array([])
    
    return {
        'radial_profile': radial_profile,
        'counts': counts,
        'dominant_radii': dominant_radii,
        'dominant_powers': dominant_powers
    }

# ============================================================================
# VISUALIZATION
# ============================================================================
def visualize_harmonic_analysis(field, analysis, shell_analysis, save_path=None):
    """Create comprehensive visualization of harmonic structure"""
    
    fig = plt.figure(figsize=(18, 12))
    
    N = field.shape[0]
    mid = N // 2
    
    # 1. Original field slice
    ax1 = fig.add_subplot(2, 3, 1)
    im1 = ax1.imshow(field[mid, :, :], cmap='magma', origin='lower')
    ax1.set_title(f'Original Field (z={mid})')
    ax1.set_xlabel('Y')
    ax1.set_ylabel('X')
    plt.colorbar(im1, ax=ax1, fraction=0.046)
    
    # 2. Reconstructed field slice
    ax2 = fig.add_subplot(2, 3, 2)
    im2 = ax2.imshow(analysis['reconstructed'][mid, :, :], cmap='magma', origin='lower')
    ax2.set_title(f'Harmonic Reconstruction\n({analysis["num_harmonics"]} notes, {analysis["compression_ratio"]:.1f}x compression)')
    ax2.set_xlabel('Y')
    ax2.set_ylabel('X')
    plt.colorbar(im2, ax=ax2, fraction=0.046)
    
    # 3. Difference (what's lost)
    ax3 = fig.add_subplot(2, 3, 3)
    diff = field - analysis['reconstructed']
    im3 = ax3.imshow(diff[mid, :, :], cmap='RdBu_r', origin='lower')
    ax3.set_title(f'Reconstruction Error\n(Power retained: {analysis["power_retained"]*100:.1f}%)')
    ax3.set_xlabel('Y')
    ax3.set_ylabel('X')
    plt.colorbar(im3, ax=ax3, fraction=0.046)
    
    # 4. Frequency spectrum slice
    ax4 = fig.add_subplot(2, 3, 4)
    spec_slice = np.log10(analysis['spectrum'][mid, :, :] + 1)
    im4 = ax4.imshow(spec_slice, cmap='plasma', origin='lower')
    ax4.set_title('Frequency Spectrum (log scale, z=0)')
    ax4.set_xlabel('ky')
    ax4.set_ylabel('kx')
    plt.colorbar(im4, ax=ax4, fraction=0.046)
    
    # 5. Radial power spectrum
    ax5 = fig.add_subplot(2, 3, 5)
    radii = np.arange(len(shell_analysis['radial_profile']))
    ax5.semilogy(radii, shell_analysis['radial_profile'] + 1e-10, 'b-', linewidth=2)
    
    # Mark dominant shells
    for i, (r, p) in enumerate(zip(shell_analysis['dominant_radii'][:5], 
                                    shell_analysis['dominant_powers'][:5])):
        ax5.axvline(r, color='red', alpha=0.5, linestyle='--')
        ax5.annotate(f'k={r}', (r, p), textcoords="offset points", 
                    xytext=(5, 5), fontsize=8, color='red')
    
    ax5.set_xlabel('Radial Wavenumber |k|')
    ax5.set_ylabel('Power (log)')
    ax5.set_title('Radial Power Spectrum')
    ax5.grid(True, alpha=0.3)
    
    # 6. 3D harmonic cloud visualization
    ax6 = fig.add_subplot(2, 3, 6, projection='3d')
    
    # Sample points from harmonic coordinates (too many to plot all)
    coords = analysis['harmonic_coords']
    if len(coords[0]) > 1000:
        # Subsample
        indices = np.random.choice(len(coords[0]), 1000, replace=False)
        x, y, z = coords[0][indices], coords[1][indices], coords[2][indices]
        c = analysis['spectrum'][x, y, z]
    else:
        x, y, z = coords
        c = analysis['spectrum'][x, y, z]
    
    scatter = ax6.scatter(x - mid, y - mid, z - mid, c=c, cmap='plasma', 
                         s=5, alpha=0.6)
    ax6.set_xlabel('kx')
    ax6.set_ylabel('ky')
    ax6.set_zlabel('kz')
    ax6.set_title(f'Harmonic Cloud\n({analysis["num_harmonics"]} discrete notes)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
    
    return fig

# ============================================================================
# MAIN ANALYSIS
# ============================================================================
def run_harmonic_analysis(N=64, topology='sphere', evolution_steps=1000, 
                          compression_threshold=0.1, save_plots=True):
    """
    Complete harmonic analysis pipeline.
    
    Args:
        N: Grid size
        topology: Initial topology ('sphere', 'torus', 'box', etc.)
        evolution_steps: How many steps to evolve before analysis
        compression_threshold: Fraction of max amplitude to keep
        save_plots: Whether to save visualization
    
    Returns:
        Dictionary with all analysis results
    """
    print("="*60)
    print("HARMONIC STRUCTURE ANALYSIS")
    print("="*60)
    
    # 1. Initialize and evolve
    print(f"\nInitializing MiniWoW (N={N}, topology={topology})...")
    sim = MiniWoW(N=N, topology=topology)
    
    print(f"Evolving for {evolution_steps} steps to reach stable state...")
    
    # Track energy to see when stable
    energies = []
    for i in range(evolution_steps // 10):
        sim.step(10)
        energy = np.sum(sim.phi**2)
        energies.append(energy)
        if (i+1) % 10 == 0:
            print(f"  Step {(i+1)*10}: Energy = {energy:.2f}")
    
    field = sim.phi.copy()
    print(f"\nFinal field stats: min={field.min():.3f}, max={field.max():.3f}, std={field.std():.3f}")
    
    # 2. Analyze harmonics
    print(f"\nAnalyzing harmonic structure (threshold={compression_threshold*100:.0f}% of max)...")
    analysis = analyze_harmonics(field, compression_threshold)
    
    print(f"\n" + "-"*40)
    print("HARMONIC ANALYSIS RESULTS")
    print("-"*40)
    print(f"  Original complexity: {N**3:,} voxels")
    print(f"  Harmonic notes found: {analysis['num_harmonics']:,}")
    print(f"  Compression ratio: {analysis['compression_ratio']:.1f}x")
    print(f"  Power retained: {analysis['power_retained']*100:.1f}%")
    
    # 3. Analyze radial shells
    print(f"\nFinding dominant harmonic shells...")
    shell_analysis = find_harmonic_shells(analysis['spectrum'])
    
    if len(shell_analysis['dominant_radii']) > 0:
        print(f"\n  Dominant wavelengths (top 5):")
        for i, (r, p) in enumerate(zip(shell_analysis['dominant_radii'][:5],
                                        shell_analysis['dominant_powers'][:5])):
            wavelength = N / max(1, r)
            print(f"    k={r:3d} (wavelength ≈ {wavelength:.1f} voxels): power={p:.2e}")
    
    # 4. Visualize
    if save_plots:
        print(f"\nGenerating visualization...")
        fig = visualize_harmonic_analysis(field, analysis, shell_analysis, 
                                          save_path='harmonic_analysis.png')
        plt.close(fig)
    
    # 5. Summary
    print(f"\n" + "="*60)
    print("SUMMARY: THE HARMONIC FINGERPRINT")
    print("="*60)
    print(f"\nThe {topology} structure can be described by {analysis['num_harmonics']} harmonic 'notes'")
    print(f"instead of {N**3:,} voxels - a {analysis['compression_ratio']:.0f}x compression!")
    print(f"\nThis means the buckyball IS essentially a standing wave pattern")
    print(f"composed of discrete spatial frequencies - pure mathematics!")
    
    if len(shell_analysis['dominant_radii']) > 0:
        dominant_k = shell_analysis['dominant_radii'][0]
        print(f"\nThe fundamental mode has wavenumber k={dominant_k}")
        print(f"corresponding to wavelength λ ≈ {N/max(1,dominant_k):.1f} voxels")
    
    return {
        'field': field,
        'analysis': analysis,
        'shell_analysis': shell_analysis,
        'simulation': sim,
        'energies': energies
    }

# ============================================================================
# ENTRY POINT
# ============================================================================
if __name__ == '__main__':
    # Run analysis with default parameters
    results = run_harmonic_analysis(
        N=64,
        topology='sphere',
        evolution_steps=1000,
        compression_threshold=0.01,
        save_plots=True
    )
    
    # Show plot if running interactively
    plt.show()