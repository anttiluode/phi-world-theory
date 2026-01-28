"""
HARMONIC FRAGILITY TEST
========================
Tests the robustness/fragility of extracted physics laws.

The Question: If you perturb a single harmonic (change its phase or amplitude),
does the universe:
  A) Heal gracefully (robust physics - small changes = small effects)
  B) Warp dramatically (fragile physics - small changes = large effects)
  C) Collapse entirely (critical physics - some harmonics are load-bearing)

This tells us whether the "laws" we extracted are:
  - Fundamental (perturbing them breaks everything)
  - Emergent (perturbing them causes local changes)
  - Redundant (perturbing them does nothing)

Author: Built for Antti's MatrixInMatrix project
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fftn, ifftn, fftshift, ifftshift
from scipy.ndimage import convolve, gaussian_filter
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# MINIWOW SIMULATION (standalone, no Ursina)
# ============================================================================
class MiniWoW:
    """Minimal MiniWoW for headless analysis"""
    
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
        
        self.phi = np.zeros((N, N, N), np.float32)
        self.phi_o = np.zeros_like(self.phi)
        
        self.kern = np.zeros((3, 3, 3), np.float32)
        self.kern[1, 1, 1] = -6
        for dx, dy, dz in [(1,1,0), (1,1,2), (1,0,1), (1,2,1), (0,1,1), (2,1,1)]:
            self.kern[dx, dy, dz] = 1
        
        if topology != 'none':
            self.init_field()
            self.phi_o = self.phi.copy()
    
    def init_field(self):
        N = self.N
        x = np.arange(N)
        X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
        c = N // 2
        
        if self.topology == 'sphere':
            r_shell = N / 3
            thickness = N / 10
            R2 = (X-c)**2 + (Y-c)**2 + (Z-c)**2
            self.phi[:] = 2 * np.exp(-(np.sqrt(R2) - r_shell)**2 / (2 * thickness**2))
        elif self.topology == 'box':
            r = N // 6
            r2 = r**2
            self.phi[:] = 2 * np.exp(-((X-c)**2 + (Y-c)**2 + (Z-c)**2) / (2 * r2))
        elif self.topology == 'torus':
            R_major = N / 3
            R_minor = N / 8
            d_circle = np.sqrt((np.sqrt((X-c)**2 + (Y-c)**2) - R_major)**2 + (Z-c)**2)
            self.phi[:] = 2 * np.exp(-(d_circle)**2 / (2 * R_minor**2))
    
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
# HARMONIC EXTRACTION
# ============================================================================
def extract_harmonics(field, threshold=0.01):
    """Extract significant harmonics from the field"""
    F = fftn(field)
    magnitude = np.abs(F)
    phase = np.angle(F)
    
    # Find significant harmonics
    max_mag = magnitude.max()
    significant = magnitude > (max_mag * threshold)
    
    # Get coordinates and values of significant harmonics
    coords = np.where(significant)
    harmonics = []
    for i in range(len(coords[0])):
        x, y, z = coords[0][i], coords[1][i], coords[2][i]
        harmonics.append({
            'coord': (x, y, z),
            'magnitude': magnitude[x, y, z],
            'phase': phase[x, y, z],
            'complex': F[x, y, z]
        })
    
    # Sort by magnitude (most important first)
    harmonics.sort(key=lambda h: h['magnitude'], reverse=True)
    
    return harmonics, F


def reconstruct_from_harmonics(F_original, harmonics_to_keep):
    """Reconstruct field from a subset of harmonics"""
    F_new = np.zeros_like(F_original)
    for h in harmonics_to_keep:
        x, y, z = h['coord']
        F_new[x, y, z] = h['complex']
    return np.real(ifftn(F_new))


# ============================================================================
# FRAGILITY METRICS
# ============================================================================
def compute_structural_similarity(field1, field2):
    """Compute structural similarity between two fields (0 to 1)"""
    # Normalize fields
    f1 = (field1 - field1.mean()) / (field1.std() + 1e-10)
    f2 = (field2 - field2.mean()) / (field2.std() + 1e-10)
    
    # Correlation coefficient
    correlation = np.mean(f1 * f2)
    
    # Power similarity
    p1 = np.sum(field1**2)
    p2 = np.sum(field2**2)
    power_ratio = min(p1, p2) / (max(p1, p2) + 1e-10)
    
    # Combined metric
    similarity = (correlation + power_ratio) / 2
    return max(0, min(1, similarity))


def compute_topology_preserved(original, perturbed, iso_level=None):
    """Check if the topological structure is preserved"""
    if iso_level is None:
        iso_level = (original.max() + original.min()) / 2
    
    # Binary masks
    mask_orig = original > iso_level
    mask_pert = perturbed > iso_level
    
    # Intersection over union
    intersection = np.sum(mask_orig & mask_pert)
    union = np.sum(mask_orig | mask_pert)
    
    if union == 0:
        return 0.0
    return intersection / union


# ============================================================================
# MAIN FRAGILITY TEST
# ============================================================================
def run_fragility_test(N=64, topology='sphere', evolution_steps=1000, 
                       compression_threshold=0.01, 
                       perturbation_strengths=[0.01, 0.05, 0.1, 0.25, 0.5],
                       num_harmonics_to_test=20):
    """
    Run the complete fragility analysis.
    
    Tests what happens when you perturb individual harmonics by various amounts.
    """
    
    print("="*70)
    print("HARMONIC FRAGILITY TEST")
    print("="*70)
    print("\nQuestion: How robust or fragile is the extracted physics?")
    print("Method: Perturb individual harmonics and measure universe stability\n")
    
    # 1. Generate stable universe
    print(f"[1/5] Generating stable {topology} universe (N={N})...")
    sim = MiniWoW(N=N, topology=topology)
    for i in range(evolution_steps // 100):
        sim.step(100)
        if (i+1) % 5 == 0:
            energy = np.sum(sim.phi**2)
            print(f"      Step {(i+1)*100}: Energy = {energy:.2f}")
    
    original_field = sim.phi.copy()
    print(f"      Final field: min={original_field.min():.3f}, max={original_field.max():.3f}")
    
    # 2. Extract harmonics
    print(f"\n[2/5] Extracting harmonics (threshold={compression_threshold*100:.1f}%)...")
    harmonics, F_full = extract_harmonics(original_field, compression_threshold)
    print(f"      Found {len(harmonics)} significant harmonics")
    
    # Reconstruct baseline
    baseline_reconstruction = reconstruct_from_harmonics(F_full, harmonics)
    baseline_similarity = compute_structural_similarity(original_field, baseline_reconstruction)
    baseline_topology = compute_topology_preserved(original_field, baseline_reconstruction)
    print(f"      Baseline reconstruction: similarity={baseline_similarity:.4f}, topology={baseline_topology:.4f}")
    
    # 3. Test perturbations
    print(f"\n[3/5] Testing perturbations on top {num_harmonics_to_test} harmonics...")
    
    results = {
        'harmonics': [],
        'fragility_scores': [],
        'phase_sensitivity': [],
        'amplitude_sensitivity': [],
        'is_load_bearing': []
    }
    
    test_harmonics = harmonics[:num_harmonics_to_test]
    
    for idx, harmonic in enumerate(test_harmonics):
        coord = harmonic['coord']
        orig_mag = harmonic['magnitude']
        orig_phase = harmonic['phase']
        
        phase_effects = []
        amp_effects = []
        
        for strength in perturbation_strengths:
            # Test PHASE perturbation
            perturbed_harmonics = []
            for h in harmonics:
                h_copy = h.copy()
                if h['coord'] == coord:
                    # Shift phase
                    new_phase = h['phase'] + strength * np.pi
                    h_copy['complex'] = h['magnitude'] * np.exp(1j * new_phase)
                perturbed_harmonics.append(h_copy)
            
            phase_recon = reconstruct_from_harmonics(F_full, perturbed_harmonics)
            phase_sim = compute_structural_similarity(baseline_reconstruction, phase_recon)
            phase_topo = compute_topology_preserved(baseline_reconstruction, phase_recon)
            phase_effects.append((1 - phase_sim, 1 - phase_topo))
            
            # Test AMPLITUDE perturbation
            perturbed_harmonics = []
            for h in harmonics:
                h_copy = h.copy()
                if h['coord'] == coord:
                    # Scale amplitude
                    new_mag = h['magnitude'] * (1 + strength)
                    h_copy['complex'] = new_mag * np.exp(1j * h['phase'])
                perturbed_harmonics.append(h_copy)
            
            amp_recon = reconstruct_from_harmonics(F_full, perturbed_harmonics)
            amp_sim = compute_structural_similarity(baseline_reconstruction, amp_recon)
            amp_topo = compute_topology_preserved(baseline_reconstruction, amp_recon)
            amp_effects.append((1 - amp_sim, 1 - amp_topo))
        
        # Compute sensitivity (slope of effect vs perturbation strength)
        phase_sensitivity = np.mean([e[0] for e in phase_effects]) / np.mean(perturbation_strengths)
        amp_sensitivity = np.mean([e[0] for e in amp_effects]) / np.mean(perturbation_strengths)
        
        # Overall fragility score for this harmonic
        fragility = (phase_sensitivity + amp_sensitivity) / 2
        
        # Is it load-bearing? (topology breaks with small perturbation)
        load_bearing = phase_effects[0][1] > 0.1 or amp_effects[0][1] > 0.1
        
        results['harmonics'].append({
            'rank': idx + 1,
            'coord': coord,
            'magnitude': orig_mag,
            'relative_power': orig_mag / harmonics[0]['magnitude']
        })
        results['fragility_scores'].append(fragility)
        results['phase_sensitivity'].append(phase_sensitivity)
        results['amplitude_sensitivity'].append(amp_sensitivity)
        results['is_load_bearing'].append(load_bearing)
        
        status = "LOAD-BEARING" if load_bearing else "stable"
        print(f"      Harmonic #{idx+1}: fragility={fragility:.4f}, {status}")
    
    # 4. Analyze results
    print(f"\n[4/5] Analyzing fragility distribution...")
    
    fragility_scores = np.array(results['fragility_scores'])
    load_bearing_count = sum(results['is_load_bearing'])
    
    # Classify universe type
    mean_fragility = np.mean(fragility_scores)
    max_fragility = np.max(fragility_scores)
    fragility_variance = np.var(fragility_scores)
    
    if mean_fragility < 0.1 and load_bearing_count == 0:
        universe_type = "ROBUST"
        description = "Small perturbations cause small effects. Physics is forgiving."
    elif load_bearing_count > len(test_harmonics) * 0.3:
        universe_type = "CRITICAL"
        description = "Many load-bearing harmonics. Physics is finely tuned."
    elif max_fragility > 0.5:
        universe_type = "FRAGILE"
        description = "Some harmonics are extremely sensitive. Butterfly effects possible."
    else:
        universe_type = "MIXED"
        description = "Some robust, some fragile harmonics. Hierarchical physics."
    
    print(f"\n      Mean fragility: {mean_fragility:.4f}")
    print(f"      Max fragility:  {max_fragility:.4f}")
    print(f"      Fragility variance: {fragility_variance:.6f}")
    print(f"      Load-bearing harmonics: {load_bearing_count}/{num_harmonics_to_test}")
    print(f"\n      Universe classification: {universe_type}")
    print(f"      {description}")
    
    # 5. Generate visualization
    print(f"\n[5/5] Generating visualization...")
    
    fig = plt.figure(figsize=(20, 16))
    
    # Plot 1: Fragility by harmonic rank
    ax1 = fig.add_subplot(2, 3, 1)
    ranks = range(1, len(fragility_scores) + 1)
    colors = ['red' if lb else 'blue' for lb in results['is_load_bearing']]
    ax1.bar(ranks, fragility_scores, color=colors, alpha=0.7, edgecolor='black')
    ax1.axhline(y=mean_fragility, color='green', linestyle='--', label=f'Mean: {mean_fragility:.3f}')
    ax1.set_xlabel('Harmonic Rank (by magnitude)')
    ax1.set_ylabel('Fragility Score')
    ax1.set_title('Fragility by Harmonic Importance\n(Red = Load-bearing)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Phase vs Amplitude sensitivity
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.scatter(results['phase_sensitivity'], results['amplitude_sensitivity'], 
                c=fragility_scores, cmap='hot', s=100, edgecolors='black')
    ax2.plot([0, max(results['phase_sensitivity'])], [0, max(results['phase_sensitivity'])], 
             'k--', alpha=0.3, label='Equal sensitivity')
    ax2.set_xlabel('Phase Sensitivity')
    ax2.set_ylabel('Amplitude Sensitivity')
    ax2.set_title('Phase vs Amplitude Sensitivity')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.colorbar(ax2.collections[0], ax=ax2, label='Fragility')
    
    # Plot 3: Fragility vs Harmonic Power
    ax3 = fig.add_subplot(2, 3, 3)
    powers = [h['relative_power'] for h in results['harmonics']]
    ax3.scatter(powers, fragility_scores, c=colors, s=100, edgecolors='black')
    ax3.set_xlabel('Relative Harmonic Power')
    ax3.set_ylabel('Fragility Score')
    ax3.set_title('Fragility vs Harmonic Importance')
    ax3.set_xscale('log')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Original field slice
    ax4 = fig.add_subplot(2, 3, 4)
    mid = N // 2
    im4 = ax4.imshow(original_field[mid], cmap='magma', origin='lower')
    ax4.set_title(f'Original Universe (z={mid})')
    plt.colorbar(im4, ax=ax4, fraction=0.046)
    
    # Plot 5: Most fragile harmonic's effect
    most_fragile_idx = np.argmax(fragility_scores)
    most_fragile_harmonic = harmonics[most_fragile_idx]
    
    # Create maximally perturbed version
    perturbed_harmonics = []
    for h in harmonics:
        h_copy = h.copy()
        if h['coord'] == most_fragile_harmonic['coord']:
            new_phase = h['phase'] + 0.5 * np.pi  # 50% phase shift
            h_copy['complex'] = h['magnitude'] * np.exp(1j * new_phase)
        perturbed_harmonics.append(h_copy)
    
    max_perturbed = reconstruct_from_harmonics(F_full, perturbed_harmonics)
    
    ax5 = fig.add_subplot(2, 3, 5)
    im5 = ax5.imshow(max_perturbed[mid], cmap='magma', origin='lower')
    ax5.set_title(f'Most Fragile Harmonic Perturbed (#{most_fragile_idx+1})\n50% phase shift')
    plt.colorbar(im5, ax=ax5, fraction=0.046)
    
    # Plot 6: Difference
    ax6 = fig.add_subplot(2, 3, 6)
    diff = original_field - max_perturbed
    im6 = ax6.imshow(diff[mid], cmap='RdBu_r', origin='lower')
    ax6.set_title('Difference (Original - Perturbed)')
    plt.colorbar(im6, ax=ax6, fraction=0.046)
    
    # Add summary text
    fig.suptitle(f'HARMONIC FRAGILITY ANALYSIS\n'
                 f'Universe Type: {universe_type}\n'
                 f'{len(harmonics)} harmonics, {load_bearing_count} load-bearing, '
                 f'mean fragility = {mean_fragility:.4f}',
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    
    # Save
    plt.savefig('fragility_analysis.png', dpi=150, bbox_inches='tight')
    print(f"      Saved: fragility_analysis.png")
    
    # Summary report
    print("\n" + "="*70)
    print("FRAGILITY ANALYSIS COMPLETE")
    print("="*70)
    print(f"\nUNIVERSE TYPE: {universe_type}")
    print(f"{description}")
    print(f"\nKey findings:")
    print(f"  - Total harmonics in 'physics': {len(harmonics)}")
    print(f"  - Load-bearing (critical) harmonics: {load_bearing_count}")
    print(f"  - Mean fragility: {mean_fragility:.4f}")
    print(f"  - Most fragile: Harmonic #{most_fragile_idx+1} (fragility={max_fragility:.4f})")
    
    if universe_type == "ROBUST":
        print("\n  → This universe's physics is FORGIVING.")
        print("    Small errors in the 'laws' don't break the universe.")
        print("    Minds here could have imperfect models and still survive.")
    elif universe_type == "CRITICAL":
        print("\n  → This universe's physics is FINELY TUNED.")
        print("    Many harmonics are load-bearing - remove one and structure collapses.")
        print("    The 'laws' here are tightly interdependent.")
    elif universe_type == "FRAGILE":
        print("\n  → This universe has BUTTERFLY EFFECTS.")
        print("    Some harmonics, when slightly changed, cause massive deviations.")
        print("    Prediction here is fundamentally limited.")
    else:
        print("\n  → This universe has HIERARCHICAL physics.")
        print("    Core harmonics are robust, fine details are fragile.")
        print("    Like our universe: gravity is robust, quantum details are sensitive.")
    
    return {
        'universe_type': universe_type,
        'mean_fragility': mean_fragility,
        'max_fragility': max_fragility,
        'load_bearing_count': load_bearing_count,
        'results': results,
        'original_field': original_field,
        'harmonics': harmonics
    }


# ============================================================================
# ENTRY POINT
# ============================================================================
if __name__ == '__main__':
    # Run the fragility test
    results = run_fragility_test(
        N=64,
        topology='sphere',
        evolution_steps=1000,
        compression_threshold=0.1,
        perturbation_strengths=[0.01, 0.05, 0.1, 0.25, 0.5],
        num_harmonics_to_test=20
    )
    
    plt.show()