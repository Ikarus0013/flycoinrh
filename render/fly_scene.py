"""
Free, zero-cost 3D fruit-fly render for the RON-50 shareable video.

Runs headless in the GH-Actions sandbox:
    blender -b -P render/fly_scene.py -- --out render/out --frames 24

Builds a recognizable Drosophila (thorax + abdomen + head + red compound
eyes + wings + legs) procedurally, lights it studio-style, animates a subtle
hover/wing-flap, and renders a transparent loop with Cycles (CPU, headless-safe).

This is the PLUMBING PROOF for the free pipeline. A CC0 scanned Drosophila
mesh (Sketchfab/Free3D) drops in via load_cc0_model() for full photoreal with
zero code change downstream.
"""
import bpy, bmesh, math, sys, os
from mathutils import Vector

# ---- args after '--' -------------------------------------------------------
argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
def arg(name, default):
    return argv[argv.index(name) + 1] if name in argv else default
OUT    = arg("--out", "render/out")
FRAMES = int(arg("--frames", "24"))
RES    = int(arg("--res", "512"))
SAMPLES= int(arg("--samples", "48"))
MODEL  = arg("--model", "")          # optional path to a CC0 .glb/.blend
os.makedirs(OUT, exist_ok=True)

# ---- clean scene -----------------------------------------------------------
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene

# ---- materials -------------------------------------------------------------
def mat(name, base, rough=0.4, metallic=0.0, emit=None, transmit=0.0):
    m = bpy.data.materials.new(name); m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*base, 1)
    b.inputs["Roughness"].default_value = rough
    b.inputs["Metallic"].default_value = metallic
    if "Transmission Weight" in b.inputs: b.inputs["Transmission Weight"].default_value = transmit
    if emit is not None:
        b.inputs["Emission Color"].default_value = (*emit, 1)
        b.inputs["Emission Strength"].default_value = 1.5
    return m

chitin = mat("chitin", (0.12, 0.10, 0.08), rough=0.35, metallic=0.25)
eye    = mat("eye",    (0.55, 0.05, 0.03), rough=0.15, emit=(0.35, 0.02, 0.01))
wing   = mat("wing",   (0.85, 0.87, 0.9),  rough=0.05, transmit=0.9)
wing.blend_method = 'BLEND' if hasattr(wing, 'blend_method') else wing.blend_method

def add_ico(name, loc, scale, m, subd=3):
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=subd, radius=1, location=loc)
    o = bpy.context.object; o.name = name; o.scale = scale
    o.data.materials.append(m)
    bpy.ops.object.shade_smooth()
    return o

def load_cc0_model(path):
    """Swap-in for a CC0 scanned Drosophila. Returns the imported root or None."""
    if not path or not os.path.exists(path): return None
    if path.endswith(".glb") or path.endswith(".gltf"):
        bpy.ops.import_scene.gltf(filepath=path)
    elif path.endswith(".blend"):
        with bpy.data.libraries.load(path) as (src, dst): dst.objects = src.objects
        for o in dst.objects:
            if o is not None: bpy.context.collection.objects.link(o)
    else:
        return None
    return bpy.context.selected_objects[0] if bpy.context.selected_objects else None

body_root = load_cc0_model(MODEL)
if body_root is None:
    # --- procedural Drosophila -------------------------------------------
    thorax  = add_ico("thorax",  (0, 0, 0),      (0.9, 1.1, 0.85), chitin)
    abdomen = add_ico("abdomen", (0, -1.7, -0.1),(0.8, 1.5, 0.75), chitin)
    head    = add_ico("head",    (0, 1.4, 0.15), (0.7, 0.6, 0.7),  chitin)
    eyeL    = add_ico("eyeL",    (0.45, 1.55, 0.25),(0.35,0.42,0.45), eye)
    eyeR    = add_ico("eyeR",    (-0.45,1.55, 0.25),(0.35,0.42,0.45), eye)
    # wings
    for i, sgn in enumerate((1, -1)):
        bpy.ops.mesh.primitive_plane_add(size=1, location=(sgn*1.4, -0.6, 0.7))
        w = bpy.context.object; w.name = f"wing{i}"
        w.scale = (2.3, 1.0, 1.0); w.rotation_euler = (0.25, 0, sgn*0.5)
        w.data.materials.append(wing)
    # legs
    for i in range(6):
        side = 1 if i % 2 == 0 else -1
        y = 0.4 - (i // 2) * 0.7
        bpy.ops.mesh.primitive_cylinder_add(radius=0.05, depth=1.4,
            location=(side*0.8, y, -0.7), rotation=(0, side*0.7, 0))
        leg = bpy.context.object; leg.data.materials.append(chitin)
    parts = [o for o in bpy.data.objects if o.type == 'MESH']
    for p in parts: p.select_set(True)
    bpy.context.view_layer.objects.active = thorax
    bpy.ops.object.join()
    body_root = bpy.context.object
body_root.name = "fly"

# ---- hover / wing-flap animation ------------------------------------------
scene.frame_start = 1; scene.frame_end = FRAMES
for f in range(1, FRAMES + 1):
    t = (f - 1) / FRAMES
    body_root.location = (0, 0, 0.15 * math.sin(t * 2 * math.pi))
    body_root.rotation_euler = (0.05 * math.sin(t * 2 * math.pi), 0,
                                0.08 * math.sin(t * 4 * math.pi))
    body_root.keyframe_insert("location", frame=f)
    body_root.keyframe_insert("rotation_euler", frame=f)

# ---- studio lighting -------------------------------------------------------
world = bpy.data.worlds.new("w"); scene.world = world
world.use_nodes = True
world.node_tree.nodes["Background"].inputs[0].default_value = (0.02, 0.02, 0.03, 1)
world.node_tree.nodes["Background"].inputs[1].default_value = 0.3
def light(name, loc, energy, size=5):
    l = bpy.data.lights.new(name, 'AREA'); l.energy = energy; l.size = size
    o = bpy.data.objects.new(name, l); o.location = loc; scene.collection.objects.link(o)
    o.rotation_euler = (math.radians(50), 0, 0)
    return o
key  = light("key",  (3, -4, 5), 900)
fill = light("fill", (-4, -2, 3), 350)
rim  = light("rim",  (0, 4, 4), 600)

# ---- camera ----------------------------------------------------------------
cam_d = bpy.data.cameras.new("cam"); cam = bpy.data.objects.new("cam", cam_d)
scene.collection.objects.link(cam); scene.camera = cam
cam.location = (0, -7.5, 1.2); cam.rotation_euler = (math.radians(88), 0, 0)
cam_d.lens = 85

# ---- render settings (Cycles CPU, transparent, headless-safe) -------------
scene.render.engine = 'CYCLES'
scene.cycles.device = 'CPU'
scene.cycles.samples = SAMPLES
scene.cycles.use_denoising = True
scene.render.film_transparent = True
scene.render.resolution_x = RES; scene.render.resolution_y = RES
scene.render.image_settings.file_format = 'PNG'
scene.render.image_settings.color_mode = 'RGBA'
scene.render.filepath = os.path.join(OUT, "fly_")
print(f"[fly_scene] rendering {FRAMES} frames @ {RES}px x{SAMPLES}spp -> {OUT}", flush=True)
bpy.ops.render.render(animation=True)
print("[fly_scene] done", flush=True)
