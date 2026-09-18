"""
Free, zero-cost 3D fruit-fly render for the RON-50 shareable video.

Runs headless in the GH-Actions sandbox:
    blender -b -P render/fly_scene.py -- --model render/assets/fly_glowbox.glb --out render/out

Ron picked "photoreal via a free CC0/CC-BY mesh" (card ron50:video:look-decision).
The default asset is a CC-BY textured housefly (Glowbox 3D, via Objaverse) vendored
at render/assets/fly_glowbox.glb -- see render/assets/CREDITS.md for attribution.

When --model is given the imported mesh (with its own PBR textures) is:
  * joined into one object, re-centred and scaled to a target size,
  * parented to an empty that hovers + slowly turns (a seamless loop),
  * framed automatically by the camera from its real bounding box.
The imported materials/textures are kept untouched -> photoreal, no downstream change.

With no --model it falls back to the procedural Drosophila (the original plumbing proof).
Renders a transparent RGBA PNG sequence with Cycles (CPU, headless-safe).
"""
import bpy, math, sys, os
from mathutils import Vector

# ---- args after '--' -------------------------------------------------------
argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
def arg(name, default):
    return argv[argv.index(name) + 1] if name in argv else default
OUT     = arg("--out", "render/out")
FRAMES  = int(arg("--frames", "48"))
RES     = int(arg("--res", "720"))
SAMPLES = int(arg("--samples", "64"))
TARGET  = float(arg("--target", "3.0"))   # world size the fly is scaled to
SPIN    = float(arg("--spin", "0.4"))     # turntable amplitude (radians, seamless)
YAW     = float(arg("--yaw", "90"))       # base yaw so the body lies across frame (deg)
TILT    = float(arg("--tilt", "0"))       # base pitch of the fly (deg)
ELEV    = float(arg("--elev", "32"))      # camera elevation above horizon (deg)
AZIM    = float(arg("--azim", "22"))      # camera azimuth off broadside (deg)
MODEL   = arg("--model", "")              # path to a CC0/CC-BY .glb/.gltf/.blend
os.makedirs(OUT, exist_ok=True)

# ---- clean scene -----------------------------------------------------------
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene

# ---- materials (only used by the procedural fallback) ----------------------
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

def add_ico(name, loc, scale, m, subd=3):
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=subd, radius=1, location=loc)
    o = bpy.context.object; o.name = name; o.scale = scale
    o.data.materials.append(m)
    bpy.ops.object.shade_smooth()
    return o

def load_cc0_model(path):
    """Import a CC0/CC-BY scanned/textured fly. Returns list of imported objects."""
    if not path or not os.path.exists(path):
        return []
    before = set(bpy.data.objects)
    if path.endswith((".glb", ".gltf")):
        bpy.ops.import_scene.gltf(filepath=path)
    elif path.endswith(".blend"):
        with bpy.data.libraries.load(path) as (src, dst):
            dst.objects = src.objects
        for o in dst.objects:
            if o is not None:
                bpy.context.collection.objects.link(o)
    else:
        return []
    return [o for o in bpy.data.objects if o not in before]

def build_procedural():
    chitin = mat("chitin", (0.12, 0.10, 0.08), rough=0.35, metallic=0.25)
    eye    = mat("eye",    (0.55, 0.05, 0.03), rough=0.15, emit=(0.35, 0.02, 0.01))
    wing   = mat("wing",   (0.85, 0.87, 0.9),  rough=0.05, transmit=0.9)
    thorax  = add_ico("thorax",  (0, 0, 0),       (0.9, 1.1, 0.85), chitin)
    add_ico("abdomen", (0, -1.7, -0.1), (0.8, 1.5, 0.75), chitin)
    add_ico("head",    (0, 1.4, 0.15),  (0.7, 0.6, 0.7),  chitin)
    add_ico("eyeL",    (0.45, 1.55, 0.25), (0.35, 0.42, 0.45), eye)
    add_ico("eyeR",    (-0.45, 1.55, 0.25), (0.35, 0.42, 0.45), eye)
    for i, sgn in enumerate((1, -1)):
        bpy.ops.mesh.primitive_plane_add(size=1, location=(sgn*1.4, -0.6, 0.7))
        w = bpy.context.object; w.name = f"wing{i}"
        w.scale = (2.3, 1.0, 1.0); w.rotation_euler = (0.25, 0, sgn*0.5)
        w.data.materials.append(wing)
    for i in range(6):
        side = 1 if i % 2 == 0 else -1
        y = 0.4 - (i // 2) * 0.7
        bpy.ops.mesh.primitive_cylinder_add(radius=0.05, depth=1.4,
            location=(side*0.8, y, -0.7), rotation=(0, side*0.7, 0))
        bpy.context.object.data.materials.append(chitin)
    parts = [o for o in bpy.data.objects if o.type == 'MESH']
    for p in parts: p.select_set(True)
    bpy.context.view_layer.objects.active = thorax
    bpy.ops.object.join()
    return bpy.context.object

# ---- build / import the fly ------------------------------------------------
bpy.ops.object.select_all(action='DESELECT')
imported = load_cc0_model(MODEL)
mesh_objs = [o for o in imported if o.type == 'MESH']

if mesh_objs:
    # bake node transforms into the meshes, then join into one object
    bpy.ops.object.select_all(action='DESELECT')
    for o in mesh_objs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = mesh_objs[0]
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    if len(mesh_objs) > 1:
        bpy.ops.object.join()
    fly = bpy.context.view_layer.objects.active
    # drop any leftover empties from the import
    for o in list(imported):
        if o.name != fly.name and o.type != 'MESH':
            try: bpy.data.objects.remove(o, do_unlink=True)
            except Exception: pass
    print(f"[fly_scene] imported CC-BY mesh from {MODEL}: {len(fly.data.vertices)} verts, "
          f"{len(fly.data.materials)} materials", flush=True)
    bpy.ops.object.shade_smooth()
else:
    if MODEL:
        print(f"[fly_scene] WARNING: could not import '{MODEL}', using procedural fly", flush=True)
    fly = build_procedural()

fly.name = "fly"

# ---- normalise: centre origin on geometry, sit at world origin, scale ------
bpy.ops.object.select_all(action='DESELECT')
fly.select_set(True); bpy.context.view_layer.objects.active = fly
bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
fly.location = (0, 0, 0)
dims = fly.dimensions
longest = max(dims.x, dims.y, dims.z) or 1.0
s = TARGET / longest
fly.scale = (s, s, s)
# base orientation: yaw the body across the frame, optional pitch, for a 3/4 view
fly.rotation_euler = (math.radians(TILT), 0.0, math.radians(YAW))
bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
print(f"[fly_scene] normalised dims -> {tuple(round(v,3) for v in fly.dimensions)}", flush=True)

# ---- rig: parent to an empty that hovers + turntables (seamless loop) ------
piv = bpy.data.objects.new("pivot", None)
scene.collection.objects.link(piv)
piv.location = (0, 0, 0)
fly.parent = piv

scene.frame_start = 1; scene.frame_end = FRAMES
for f in range(1, FRAMES + 1):
    t = (f - 1) / FRAMES               # 0..1, wraps so frame 1 == frame FRAMES+1
    piv.location = (0, 0, 0.12 * math.sin(t * 2 * math.pi))
    piv.rotation_euler = (
        0.05 * math.sin(t * 2 * math.pi),
        0.0,
        SPIN * math.sin(t * 2 * math.pi),
    )
    piv.keyframe_insert("location", frame=f)
    piv.keyframe_insert("rotation_euler", frame=f)
# linear-ish ease so the loop velocity matches at the seam
for fc in piv.animation_data.action.fcurves:
    for kp in fc.keyframe_points:
        kp.interpolation = 'BEZIER'

# ---- studio lighting -------------------------------------------------------
world = bpy.data.worlds.new("w"); scene.world = world
world.use_nodes = True
world.node_tree.nodes["Background"].inputs[0].default_value = (0.03, 0.05, 0.04, 1)
world.node_tree.nodes["Background"].inputs[1].default_value = 0.35
def light(name, loc, energy, size=6):
    l = bpy.data.lights.new(name, 'AREA'); l.energy = energy; l.size = size
    o = bpy.data.objects.new(name, l); o.location = loc; scene.collection.objects.link(o)
    d = (Vector((0, 0, 0.4)) - Vector(loc)); d.normalize()
    o.rotation_euler = d.to_track_quat('-Z', 'Y').to_euler()
    return o
light("key",  (3.5, -4.5, 5.5), 1200)
light("fill", (-4.5, -2.5, 3.0), 450)
light("rim",  (0, 4.5, 4.5), 800)

# ---- camera: auto-frame the fly's bounding sphere --------------------------
bb = [fly.matrix_world @ Vector(c) for c in fly.bound_box]
center = sum(bb, Vector()) / 8.0
radius = max((v - center).length for v in bb)
cam_d = bpy.data.cameras.new("cam"); cam = bpy.data.objects.new("cam", cam_d)
scene.collection.objects.link(cam); scene.camera = cam
cam_d.lens = 85
fov = 2 * math.atan(cam_d.sensor_width / (2 * cam_d.lens))
dist = (radius * 1.9) / math.tan(fov / 2)
el = math.radians(ELEV); az = math.radians(AZIM)
# camera on a sphere around the fly: broadside (-Y) + azimuth swing + elevation
dir_to_cam = Vector((math.sin(az) * math.cos(el),
                     -math.cos(az) * math.cos(el),
                     math.sin(el)))
cam.location = center + dir_to_cam * dist
look = center - Vector(cam.location)
cam.rotation_euler = look.to_track_quat('-Z', 'Y').to_euler()

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
