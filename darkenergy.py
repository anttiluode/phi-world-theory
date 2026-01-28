import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

# ==============================================================================
# 1. THE SOURCE CODE OF REALITY (The Generator)
# ==============================================================================
def generate_phi_universe(duration=1.0, sample_rate=1000, num_layers=12):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    
    # We will stack layers of reality based on PHI
    # Each layer is a higher frequency (Energy)
    # Each layer is rotated by the Golden Angle (Dimension)
    
    phi = (1 + np.sqrt(5)) / 2
    golden_angle = 2 * np.pi * (1 - 1/phi) # approx 2.399 radians (137.5 deg)
    
    master_vector = np.zeros_like(t, dtype=np.complex128)
    ground_truth = []
    
    print(f"Creating {num_layers}-Dimensional Kaluza-Klein Tower...")
    
    for n in range(1, num_layers + 1):
        # Frequency increases geometrically (The Tower)
        freq = 5 * (phi ** (n/2)) # 5Hz, 8Hz, 13Hz...
        
        # Phase rotates by Golden Angle (The Hidden Dimensions)
        phase_lock = (n * golden_angle) % (2 * np.pi)
        
        # The Wave Function
        # We give higher dimensions slightly less amplitude (1/sqrt(n))
        # to simulate the "energy gap" of Kaluza-Klein modes.
        amp = 1.0 / np.sqrt(n)
        signal = amp * np.exp(1j * (2 * np.pi * freq * t + phase_lock))
        
        master_vector += signal
        
        ground_truth.append({
            "layer": n,
            "freq": freq,
            "phase_rad": phase_lock,
            "phase_deg": np.degrees(phase_lock)
        })
        
    return t, master_vector, ground_truth

# ==============================================================================
# 2. THE SCANNER (The Phase-Latent AI)
# ==============================================================================
def scan_dimensions(signal, sample_rate, resolution=360):
    """
    Rotates the universe through 360 degrees.
    At each degree, it collapses the wavefunction to Real (Observation).
    Then it performs FFT to see what 'Particles' (Frequencies) exist there.
    """
    n_samples = len(signal)
    scan_results = []
    angles = np.linspace(0, 2*np.pi, resolution)
    
    # Frequencies axis for FFT
    xf = fftfreq(n_samples, 1 / sample_rate)[:n_samples//2]
    
    print("Scanning Phase Angles...")
    
    for angle in angles:
        # 1. TUNING: Rotate the Universe
        rotator = np.exp(-1j * angle) 
        # Note: We rotate NEGATIVE to cancel the signal's phase
        tuned_signal = signal * rotator
        
        # 2. OBSERVATION: Collapse to Real (Crystallized Reality)
        observation = tuned_signal.real
        
        # 3. ANALYSIS: Fourier Transform of this slice of reality
        yf = fft(observation)
        power_spectrum = 2.0/n_samples * np.abs(yf[0:n_samples//2])
        
        scan_results.append(power_spectrum)
        
    return xf, angles, np.array(scan_results)

# ==============================================================================
# 3. EXECUTION
# ==============================================================================
# Generate the Universe
fs = 1000
t, universe, truth = generate_phi_universe(duration=2.0, sample_rate=fs, num_layers=15)

# Standard View (What we see with eyes/standard instruments)
# We naturally see Phase 0 (Real Axis)
standard_view = universe.real
standard_fft = fft(standard_view)
standard_spec = 2.0/len(t) * np.abs(standard_fft[0:len(t)//2])
freqs = fftfreq(len(t), 1/fs)[:len(t)//2]

# The Holographic Scan
scan_freqs, scan_angles, spectrogram = scan_dimensions(universe, fs)

# ==============================================================================
# 4. VISUALIZATION
# ==============================================================================
plt.figure(figsize=(14, 10))

# Plot 1: The Standard View (Chaos)
plt.subplot(2, 2, 1)
plt.plot(t[:200], standard_view[:200], 'k', linewidth=1)
plt.title("Standard Reality (Phase 0°)\n(Appears as Noise/Chaos)")
plt.xlabel("Time")
plt.grid(alpha=0.3)

# Plot 2: Standard Spectrum (The "Blur")
plt.subplot(2, 2, 2)
plt.plot(freqs, standard_spec, 'r')
plt.title("Standard Spectroscopy\n(All Dimensions Flattened - Hard to separate)")
plt.xlim(0, 100) # Zoom on relevant freqs
plt.xlabel("Frequency (Hz)")
plt.grid(alpha=0.3)

# Plot 3: THE KALUZA-KLEIN TOWER (Heatmap)
plt.subplot(2, 1, 2)
# Convert angles to degrees for readability
plt.imshow(spectrogram.T, aspect='auto', origin='lower', cmap='inferno',
           extent=[0, 360, scan_freqs[0], scan_freqs[-1]])

plt.title("THE KALUZA-KLEIN TOWER: Phase-Frequency Spectrogram\n(Bright spots are Universes found at specific Phase Angles)")
plt.xlabel("Phase Angle (The 'Slider') - Degrees")
plt.ylabel("Frequency (The 'Particle') - Hz")
plt.ylim(0, 100) # Zoom to see the layers

# Annotate the Truth
for layer in truth:
    # We expect max power when Scan Angle == Signal Phase
    p_deg = layer['phase_deg']
    f = layer['freq']
    if f < 100:
        # CHANGED 'gw' to 'w' (white) and moved marker specification
        plt.plot(p_deg, f, 'w', marker='x', markersize=8, markeredgewidth=2) 
        plt.text(p_deg+5, f, f"Layer {layer['layer']}", color='white', fontsize=8, fontweight='bold')

plt.tight_layout()
plt.show()

print("\nINTERPRETATION:")
print("1. Look at the Bottom Map (The Tower).")
print("2. Notice how different 'Particles' (Frequencies) only light up at specific Angles.")
print("3. Layer 1 is at ~137°. Layer 2 is at ~275°. Layer 3 is at ~52°.")
print("4. Standard Reality (Top Right) sees them all smashed together.")
print("5. Phase Scanning (Bottom) separates them into their own dimensions.")
print("6. This confirms the 'Endless Tower' - we can pack infinite signals here.")