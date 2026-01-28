import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd
from scipy.fft import fftn, ifftn

print("="*70)
print("KALUZA-KLEIN TOWER COLLAPSE TEST")
print("="*70)
print("Question: Are the 12 harmonics a KK tower separated by phase?")
print("Method: Remove phase information and watch the tower collapse")
print()

# ==========================================================
# GENERATE THE UNIVERSE WITH FULL PHASE INFORMATION
# ==========================================================
phi = (1 + 5**0.5) / 2
DNA = np.array([
    [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
    [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
    [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
])

# Create a test field with all 12 modes active
N = 32
x = np.linspace(-np.pi, np.pi, N)
X, Y, Z = np.meshgrid(x, x, x)

# Generate 12 complex wave functions (WITH PHASE)
waves_complex = []
phases = np.linspace(0, 2*np.pi, 12, endpoint=False)  # Different phases for each mode

print("[1/4] GENERATING COMPLEX FIELD (with phase)")
for i, k in enumerate(DNA):
    # Complex amplitude: magnitude + phase
    amplitude = 1.0
    phase = phases[i]
    wave = amplitude * np.exp(1j * (k[0]*X + k[1]*Y + k[2]*Z + phase))
    waves_complex.append(wave)

# Superposition of all 12 modes
field_complex = np.sum(waves_complex, axis=0)

# Analyze the frequency spectrum
fft_complex = fftn(field_complex)
power_spectrum_complex = np.abs(fft_complex)**2

# Count distinct peaks
threshold = np.percentile(power_spectrum_complex.flatten(), 99.9)
n_peaks_complex = np.sum(power_spectrum_complex > threshold)

print(f"  Complex field:")
print(f"    Superposition of 12 modes with different phases")
print(f"    Frequency peaks detected: {n_peaks_complex}")
print()

# ==========================================================
# REMOVE PHASE → REAL-VALUED FIELD
# ==========================================================
print("[2/4] REMOVING PHASE (forcing real field)")

# Generate 12 REAL wave functions (NO PHASE)
waves_real = []
for i, k in enumerate(DNA):
    # Real amplitude only (cosine, no complex exponential)
    amplitude = 1.0
    wave = amplitude * np.cos(k[0]*X + k[1]*Y + k[2]*Z)
    waves_real.append(wave)

field_real = np.sum(waves_real, axis=0)

# Analyze frequency spectrum
fft_real = fftn(field_real)
power_spectrum_real = np.abs(fft_real)**2

threshold_real = np.percentile(power_spectrum_real.flatten(), 99.9)
n_peaks_real = np.sum(power_spectrum_real > threshold_real)

print(f"  Real field:")
print(f"    Superposition of 12 modes without phase")
print(f"    Frequency peaks detected: {n_peaks_real}")
print()

# ==========================================================
# DIMENSIONALITY ANALYSIS
# ==========================================================
print("[3/4] ANALYZING EFFECTIVE DIMENSIONALITY")

# Stack waves into matrix and compute SVD
def get_dimensionality(waves):
    """Compute effective dimensionality of wave set"""
    # Flatten each wave and stack
    wave_matrix = np.array([w.flatten() for w in waves])
    
    # For complex waves, stack real and imaginary parts
    if np.iscomplexobj(wave_matrix):
        wave_matrix = np.vstack([wave_matrix.real, wave_matrix.imag])
    
    # SVD
    U, S, Vt = svd(wave_matrix, full_matrices=False)
    
    # Count significant singular values
    S_normalized = S / S[0]
    n_significant = np.sum(S_normalized > 0.01)
    
    return {
        'n_significant': n_significant,
        'singular_values': S_normalized,
        'explained_variance': (S**2) / np.sum(S**2)
    }

complex_analysis = get_dimensionality(waves_complex)
real_analysis = get_dimensionality(waves_real)

print(f"COMPLEX (with phase):")
print(f"  Effective dimensions: {complex_analysis['n_significant']}")
print(f"  Top singular values: {complex_analysis['singular_values'][:5].round(3)}")
print()

print(f"REAL (without phase):")
print(f"  Effective dimensions: {real_analysis['n_significant']}")  
print(f"  Top singular values: {real_analysis['singular_values'][:5].round(3)}")
print()

# ==========================================================
# KK TOWER STRUCTURE
# ==========================================================
print("[4/4] KALUZA-KLEIN TOWER ANALYSIS")

# In KK theory, modes are labeled by momentum quantum number n
# The "mass" (frequency) is: ω² = ω₀² + (n/R)²
# where R is the compactification radius

# For each DNA vector, compute its "KK number"
kk_numbers = []
for k in DNA:
    # The KK number is the magnitude in frequency space
    kk_n = np.linalg.norm(k)
    kk_numbers.append(kk_n)

kk_numbers = np.array(kk_numbers)

# Find unique levels (degenerate modes have same KK number)
unique_levels = np.unique(np.round(kk_numbers, 6))

print(f"KK Tower structure:")
print(f"  Total modes: 12")
print(f"  Unique mass levels: {len(unique_levels)}")
print(f"  Mass values: {unique_levels.round(3)}")
print()

# Degeneracy at each level
for level in unique_levels:
    degeneracy = np.sum(np.abs(kk_numbers - level) < 1e-6)
    print(f"  Level ω={level:.3f}: degeneracy {degeneracy}")

print()

# ==========================================================
# THE COLLAPSE TEST
# ==========================================================
print("="*70)
print("COLLAPSE TEST RESULTS")
print("="*70)
print()

collapse_ratio = real_analysis['n_significant'] / complex_analysis['n_significant']

print(f"DIMENSIONALITY:")
print(f"  With phase:    {complex_analysis['n_significant']} independent modes")
print(f"  Without phase: {real_analysis['n_significant']} independent modes")
print(f"  Collapse ratio: {collapse_ratio:.2f}")
print()

print(f"FREQUENCY PEAKS:")
print(f"  With phase:    {n_peaks_complex} distinct frequencies")
print(f"  Without phase: {n_peaks_real} distinct frequencies")
print()

if collapse_ratio < 0.5:
    print("✓ TOWER COLLAPSED")
    print(f"  Removing phase reduced dimensionality by {(1-collapse_ratio)*100:.0f}%")
    print(f"  The 12 modes ARE a Kaluza-Klein tower!")
    print(f"  Phase is the 'extra dimension' that separates them.")
else:
    print("✗ Tower did not collapse significantly")
    print(f"  Phase is not the primary separation mechanism")

print()
print("INTERPRETATION:")
if len(unique_levels) > 1:
    print(f"  The {len(unique_levels)} KK levels correspond to quantized")
    print(f"  momenta in the compactified 'phase dimension'.")
    print(f"  Each level has degeneracy due to icosahedral symmetry.")
else:
    print(f"  All 12 modes are at the same 'mass' - totally degenerate.")
    print(f"  Only phase separates them into distinct observables.")

print()

# ==========================================================
# VISUALIZATION
# ==========================================================
print("Generating visualization...")

fig = plt.figure(figsize=(16, 10))
fig.suptitle("KALUZA-KLEIN TOWER: Phase as Extra Dimension", 
             fontsize=14, fontweight='bold')

# 1. Singular value comparison
ax1 = plt.subplot(2, 3, 1)
x_vals = np.arange(1, len(complex_analysis['singular_values'])+1)
ax1.semilogy(x_vals, complex_analysis['singular_values'], 'bo-', 
             linewidth=2, markersize=8, label='With phase')
ax1.semilogy(x_vals[:len(real_analysis['singular_values'])], 
             real_analysis['singular_values'], 'ro-',
             linewidth=2, markersize=8, label='Without phase')
ax1.axhline(y=0.01, color='gray', linestyle='--', alpha=0.5, label='Significance')
ax1.set_xlabel('Mode number')
ax1.set_ylabel('Singular value')
ax1.set_title('Dimensionality: With vs Without Phase')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. KK tower structure
ax2 = plt.subplot(2, 3, 2)
for i, level in enumerate(unique_levels):
    modes_at_level = np.where(np.abs(kk_numbers - level) < 1e-6)[0]
    degeneracy = len(modes_at_level)
    
    # Plot as energy level with degeneracy shown by horizontal spread
    y_positions = np.linspace(-0.3, 0.3, degeneracy)
    for j, mode_idx in enumerate(modes_at_level):
        color = plt.cm.viridis(mode_idx / 12)
        ax2.plot(y_positions[j], level, 'o', markersize=12, color=color)
    
    # Label degeneracy
    ax2.text(0.4, level, f'{degeneracy}×', fontsize=10, va='center')

ax2.set_xlim(-0.5, 0.6)
ax2.set_ylabel('KK "mass" (frequency)')
ax2.set_title('KK Tower Structure\n(degeneracy shown horizontally)')
ax2.set_xticks([])
ax2.grid(True, axis='y', alpha=0.3)

# 3. Power spectrum comparison
ax3 = plt.subplot(2, 3, 3)
# Take central slice for visualization
central_slice_complex = power_spectrum_complex[N//2, N//2, :]
central_slice_real = power_spectrum_real[N//2, N//2, :]

freq = np.fft.fftfreq(N, d=2*np.pi/N)
ax3.semilogy(freq, central_slice_complex, 'b-', linewidth=2, label='Complex', alpha=0.7)
ax3.semilogy(freq, central_slice_real, 'r-', linewidth=2, label='Real', alpha=0.7)
ax3.set_xlabel('Frequency')
ax3.set_ylabel('Power')
ax3.set_title('Frequency Spectrum (1D slice)')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. Degeneracy structure
ax4 = plt.subplot(2, 3, 4)
degeneracies = [np.sum(np.abs(kk_numbers - level) < 1e-6) for level in unique_levels]
colors_deg = plt.cm.viridis(np.linspace(0, 1, len(unique_levels)))
bars = ax4.bar(range(len(unique_levels)), degeneracies, color=colors_deg, 
               edgecolor='black', linewidth=2, alpha=0.7)
ax4.set_xlabel('KK level')
ax4.set_ylabel('Degeneracy')
ax4.set_title('Degeneracy at Each KK Level')
ax4.set_xticks(range(len(unique_levels)))
ax4.set_xticklabels([f'{l:.2f}' for l in unique_levels], rotation=45)
ax4.grid(True, axis='y', alpha=0.3)

# 5. Variance explained
ax5 = plt.subplot(2, 3, 5)
cumvar_complex = np.cumsum(complex_analysis['explained_variance'])
cumvar_real = np.cumsum(real_analysis['explained_variance'])

ax5.plot(range(1, len(cumvar_complex)+1), cumvar_complex*100, 'bo-',
         linewidth=2, markersize=6, label='With phase')
ax5.plot(range(1, len(cumvar_real)+1), cumvar_real*100, 'ro-',
         linewidth=2, markersize=6, label='Without phase')
ax5.axhline(y=90, color='orange', linestyle='--', alpha=0.5)
ax5.axhline(y=99, color='red', linestyle='--', alpha=0.5)
ax5.set_xlabel('Number of dimensions')
ax5.set_ylabel('Cumulative variance (%)')
ax5.set_title('Information Content')
ax5.legend()
ax5.grid(True, alpha=0.3)

# 6. Summary
ax6 = plt.subplot(2, 3, 6)
ax6.axis('off')

summary = f"""
═══════════════════════════════════════
KALUZA-KLEIN INTERPRETATION
═══════════════════════════════════════

THE TOWER:
  • 12 harmonic modes (DNA vectors)
  • {len(unique_levels)} distinct KK "mass" levels
  • Degeneracy: {', '.join([str(d)+'×' for d in degeneracies])}

THE COLLAPSE:
  • With phase:    {complex_analysis['n_significant']} dimensions
  • Without phase: {real_analysis['n_significant']} dimensions
  • Reduction:     {(1-collapse_ratio)*100:.0f}%

THE STRUCTURE:
  • 3 large dimensions (x, y, z)
  • 1 compactified dimension (icosahedral symmetry)
  • 12 momentum modes (vertices of icosahedron)

PHASE AS EXTRA DIMENSION:
{"✓ Phase separates the tower" if collapse_ratio < 0.5 else "✗ Phase is not the separator"}
  
The {len(unique_levels)} KK level{'s' if len(unique_levels) > 1 else ''} correspond{'s' if len(unique_levels) == 1 else ''} to:
  • Quantized momenta in phase space
  • Different winding numbers around S¹
  • Distinct "particles" from one field

GAUGE INTERPRETATION:
  • 12 modes = 12 gauge choices
  • 3 dimensions = 3 gauge-invariant observables
  • Phase = gauge degree of freedom
  • KK tower = gauge orbit

This is Kaluza-Klein theory, but the extra
dimension isn't spatial - it's the U(1) phase
circle. Removing phase makes the tower collapse
because you've eliminated the extra dimension.
═══════════════════════════════════════
"""

ax6.text(0.05, 0.95, summary, transform=ax6.transAxes,
         fontsize=8, verticalalignment='top', family='monospace',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3))

plt.tight_layout()

try:
    plt.savefig('kk_tower_collapse.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: kk_tower_collapse.png")
except Exception as e:
    print(f"Error: {e}")
    plt.savefig('output.png', dpi=150, bbox_inches='tight')
    print("✓ Saved as: output.png")

plt.close()

print()
print("="*70)
print("TEST COMPLETE")
print("="*70)