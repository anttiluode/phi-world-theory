import numpy as np
from ursina import *
from ursina.shaders import lit_with_shadows_shader
from skimage.measure import marching_cubes
import colorsys

# --- THE UNIVERSAL DNA ---
phi = (1 + 5**0.5) / 2
DNA = np.array([
    [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
    [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
    [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
])

app = Ursina()
window.color = color.rgb(0, 0.05, 0.15) 
window.title = "Cosmic Buckyball: The Infinite Lattice"

# Camera Setup
cam = EditorCamera()
cam.move_speed = 15
cam.rotation_speed = 100

# Lighting
sun = DirectionalLight(y=20, rotation=(45,45,45))
AmbientLight(color=color.rgba(0.3, 0.3, 0.5, 0.3))

# --- STATE VARIABLES ---
current_res = 64
energy = 1.5
dna_scale = 1.0
envelope_decay = 0.5 # New: How fast the 'gravity' fades
universe_size = 6.0   # New: How much space we render
morph_val = 0.0 
phase = 0.0
is_rotating = False
field = None

# --- MESH ENTITY ---
bucky = Entity(model=None, shader=lit_with_shadows_shader)

def generate_field():
    global field
    res = current_res
    # UNIVERSE SIZE controls the 'Zoom Out'
    x = np.linspace(-universe_size, universe_size, res)
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
    
    # --- TOPOLOGICAL ENVELOPES ---
    env_sphere = (X**2 + Y**2 + Z**2)
    rad_dist = np.sqrt(X**2 + Y**2) - 3.5
    env_torus = (rad_dist**2 + Z**2) * 2.0
    env_box = (np.abs(X)**8 + np.abs(Y)**8 + np.abs(Z)**8) * 0.001
    
    if morph_val <= 1.0:
        envelope = (1-morph_val)*env_sphere + morph_val*env_torus
    else:
        t = morph_val - 1.0
        envelope = (1-t)*env_torus + t*env_box
        
    # ENVELOPE DECAY lets the repetitions appear
    field = 5.0 * np.exp(-envelope * (envelope_decay * 0.2)) 
    
    for i, k in enumerate(DNA):
        arg = (k[0]*X + k[1]*Y + k[2]*Z) * dna_scale + (phase * (i % 3))
        field += 1.2 * np.cos(arg)

def update_geometry():
    try:
        if field is None: return
        verts, faces, normals, _ = marching_cubes(field, level=energy)
        # Rescale vertices to match the Universe Size in Ursina
        verts = (verts - (current_res/2)) * ((universe_size*4) / current_res)
        
        colors = [color.hsv((np.linalg.norm(v)*0.05 + time.time()*0.05)%1.0, 0.7, 1.0) for v in verts]
            
        bucky.model = Mesh(
            vertices=verts.tolist(), triangles=faces.flatten().tolist(), 
            normals=normals.tolist(), colors=colors
        )
        status.text = f"UNIVERSE STABLE | Verts: {len(verts)}"
    except Exception:
        bucky.model = None
        status.text = "PHASE GAP: No matter detected in this range."

# --- UI (LABORATORY STYLE) ---
ui_panel = Entity(parent=camera.ui, model='quad', scale=(0.35, 1), x=-0.82, color=color.black66)
menu = Entity(parent=camera.ui, x=-0.82, y=0.45)

def add_slider(label, v_min, v_max, v_def, y_off, attr_name, needs_field=False):
    Text(label, parent=menu, y=y_off, scale=0.7, x=0, origin=(0,0))
    s = Slider(min=v_min, max=v_max, default=v_def, x=0, y=y_off-0.04, scale=0.3, parent=menu, dynamic=True)
    def on_change():
        globals()[attr_name] = s.value
        if needs_field: generate_field()
        update_geometry()
    s.on_value_changed = on_change
    return s

add_slider("Universe FOV (Zoom Out)", 3, 20, 6, 0.0, "universe_size", True)
add_slider("Envelope Decay (Gravity)", 0.01, 1.0, 0.5, -0.12, "envelope_decay", True)
add_slider("Energy (Threshold)", -2, 8, 1.5, -0.24, "energy")
add_slider("DNA Scale", 0.1, 3.0, 1.0, -0.36, "dna_scale", True)
add_slider("Morph (S->T->B)", 0, 2.0, 0, -0.48, "morph_val", True)

# Grid Size Buttons
Text("Complexity:", parent=menu, y=-0.6, scale=0.7, x=0, origin=(0,0))
res_grid = Entity(parent=menu, y=-0.68)
for i, r in enumerate([64, 128, 256]):
    btn = Button(text=str(r), parent=res_grid, x=(i-1)*0.1, scale=(0.08, 0.04), color=color.azure)
    def make_func(val):
        def change_res():
            global current_res
            current_res = val
            generate_field(); update_geometry()
        return change_res
    btn.on_click = make_func(r)

rot_btn = Button(text="Revolution: OFF", color=color.red, scale=(0.25, 0.05), y=-0.8, parent=menu)
def toggle_rot():
    global is_rotating
    is_rotating = not is_rotating
    rot_btn.text = f"Revolution: {'ON' if is_rotating else 'OFF'}"
    rot_btn.color = color.green if is_rotating else color.red
rot_btn.on_click = toggle_rot

status = Text(text="", position=(0, 0.45), origin=(0,0), color=color.cyan)
generate_field(); update_geometry()

def update():
    if is_rotating and bucky.model:
        bucky.rotation_y += time.dt * 15
    if held_keys['q']: cam.rotation_y += 1
    if held_keys['e']: cam.rotation_y -= 1

app.run()