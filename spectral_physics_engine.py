import numpy as np
from ursina import *
from ursina.shaders import lit_with_shadows_shader
from skimage.measure import marching_cubes
import time

# --- THE RESONANT DNA (The 12-Tone Chord) ---
phi = (1 + 5**0.5) / 2
DNA = np.array([
    [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
    [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
    [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
])

app = Ursina()
window.color = color.rgb(0, 0.05, 0.1)
EditorCamera()

# Lighting for 3D depth
sun = DirectionalLight(y=10, rotation=(45,45,45))
AmbientLight(color=color.rgba(0.2, 0.2, 0.4, 0.2))

# --- SIMULATION STATE ---
# Notice we are using 128 resolution! (8x denser than your 64 sim)
# A voxel sim would crawl at this resolution, but math is instant.
res = 128 
energy_threshold = 1.6
dna_scale = 1.0
sim_time = 0.0
is_playing = True

# Prepare the 3D Coordinate Grid once
x_range = np.linspace(-6, 6, res)
X, Y, Z = np.meshgrid(x_range, x_range, x_range, indexing='ij')
# Pre-calculate the Gaussian Envelope (The 'Gravity' field)
envelope = np.exp(-(X**2 + Y**2 + Z**2) * 0.1)

# The Universe Entity
universe_mesh = Entity(model=None, shader=lit_with_shadows_shader)

def update_universe():
    global sim_time
    # 1. THE PHYSICS STEP: We only evolve the PHASE
    # Instead of moving pixels, we 'slide' the 12 harmonics
    field = envelope * 2.0
    for i, k in enumerate(DNA):
        # Each harmonic has its own frequency/speed
        phase_velocity = sim_time * (1.0 + (i * 0.1))
        field += np.cos((k[0]*X + k[1]*Y + k[2]*Z) * dna_scale + phase_velocity)

    # 2. THE RENDERING STEP: Convert math to geometry
    try:
        verts, faces, normals, _ = marching_cubes(field, level=energy_threshold)
        verts = (verts - (res/2)) * (20 / res) # Scale to Ursina space
        
        # Color by distance (Spectral Glow)
        colors = [color.hsv((np.linalg.norm(v)*0.1 + sim_time*0.1)%1.0, 0.6, 1.0) for v in verts]
        
        universe_mesh.model = Mesh(
            vertices=verts.tolist(), triangles=faces.flatten().tolist(), 
            normals=normals.tolist(), colors=colors
        )
    except:
        universe_mesh.model = None

# --- UI CONTROLS ---
panel = Entity(parent=camera.ui, model='quad', scale=(0.3, 0.5), x=-0.8, color=color.black66)
lbl = Text("HYPER-FAST SPECTRAL SIM", parent=panel, y=0.4, x=0, origin=(0,0), scale=0.8)

speed_slider = Slider(min=0, max=5, default=1, x=0, y=0.2, parent=panel, text="Flow Speed")
scale_slider = Slider(min=0.5, max=3, default=1, x=0, y=0.0, parent=panel, text="DNA Scale")

def update():
    global sim_time
    if is_playing:
        sim_time += time.dt * speed_slider.value
        # Update Scale from slider
        globals()['dna_scale'] = scale_slider.value
        update_universe()

# Initial Build
update_universe()
app.run()