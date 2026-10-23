"""
Builds the plantation scene XML: the Skydio X2 drone (unmodified) + a soil
floor, a gradient sky and a sun that casts shadows + a vineyard of visual-only
scanned grapevines (CC BY 4.0, see assets/grapevine/README.md) on trellises
(end posts and wires) over grass strips. The drone flies 1.2 m above the centre
row, as the reports describe; the other rows make the field read as a vineyard.

The vines, trellises and grass strips are purely cosmetic (contype="0" conaffinity="0" -- no
collision, no effect on physics or training), so this does not change the RL
problem in any way, just the presentation.
"""
import os
import random

DEFAULT_SCENE_NAME = "x2_plantation_scene.xml"


VINE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "grapevine")
# The scan is real scale (0.83 m tall). MuJoCo scales meshes per asset, not per geom, so a few
# scaled copies give size variety; the largest keeps every vine 0.37 m under a drone at 1.2 m.
VINE_SCALES = (0.85, 0.92, 1.0)


def _vine_assets_xml():
    """Texture, material and scaled meshes of the scanned grapevine (see assets/grapevine/README.md).

    Paths are absolute because x2.xml sets its own assetdir, which would
    otherwise be prepended to these files.
    """
    meshes = "\n".join(
        f'    <mesh name="grapevine_{i}" file="{os.path.join(VINE_DIR, "grapevine.obj")}" '
        f'scale="{s} {s} {s}"/>'
        for i, s in enumerate(VINE_SCALES)
    )
    return (
        f'    <texture name="grapevine" type="2d" file="{os.path.join(VINE_DIR, "grapevine_diff.png")}"/>\n'
        f'    <material name="grapevine" texture="grapevine" specular="0.1" reflectance="0"/>\n'
        f"{meshes}"
    )


def _vine_xml(x, y, rng):
    """One scanned vine with its own size, heading and small offset, so neighbours never look cloned."""
    return (
        f'<geom type="mesh" mesh="grapevine_{rng.randrange(len(VINE_SCALES))}" material="grapevine" '
        f'pos="{x + rng.uniform(-0.08, 0.08):.2f} {y + rng.uniform(-0.05, 0.05):.2f} 0" '
        f'euler="0 0 {rng.uniform(0, 360):.0f}" contype="0" conaffinity="0"/>'
    )


# Trellis ("espaldera"): a wooden post at each end of a row and thin wires running post to post
# through the canopy. The posts stay 0.25 m under a drone flying 1.2 m above the centre row.
POST_HEIGHT = 0.95
WIRE_HEIGHTS = (0.30, 0.55, 0.80)


def _trellis_xml(y, x_start, x_end):
    """Two end posts and the wires strung between them for the row at `y` (visual only)."""
    posts = [
        f'<geom type="cylinder" size="0.04 {POST_HEIGHT / 2:.3f}" pos="{x:.2f} {y:.2f} {POST_HEIGHT / 2:.3f}" '
        f'rgba="0.55 0.38 0.22 1" contype="0" conaffinity="0"/>'
        for x in (x_start, x_end)
    ]
    wires = [
        f'<geom type="capsule" fromto="{x_start:.2f} {y:.2f} {z:.2f} {x_end:.2f} {y:.2f} {z:.2f}" size="0.003" '
        f'rgba="0.55 0.55 0.55 1" contype="0" conaffinity="0"/>'
        for z in WIRE_HEIGHTS
    ]
    return "\n    ".join(posts + wires)


def build_plantation_scene(drone_dir, field_length=20.0, row_spacing=1.8,
                           plant_gap=0.6, out_name=DEFAULT_SCENE_NAME, seed=0,
                           n_rows=5):
    """Writes a scene file next to x2.xml and returns its path.

    field_length: how far the crop rows extend along +x (match your env's
        `field_length` so the visual field matches the flight distance).
    row_spacing: distance between neighbouring vine rows. The centre row is
        at y=0, right under the flight path; the others sit at multiples of
        +/- row_spacing.
    plant_gap: spacing between vines along each row.
    n_rows: number of vine rows (odd, so one row is centred under the drone).
    seed: fixes the vine variation so the scene is identical on every run.

    Only the look of the floor changes; its geometry and collision are the
    same as before.
    """
    drone_dir = os.path.abspath(drone_dir)
    base_xml = os.path.join(drone_dir, "x2.xml")
    if not os.path.isfile(base_xml):
        # Fail loudly here rather than letting MuJoCo raise a confusing parse
        # error about a missing include several frames later.
        raise FileNotFoundError(
            f"Expected the Skydio X2 model at {base_xml}, but it is not there. "
            "Run `make menagerie` to sparse-clone mujoco_menagerie, or pass "
            "--drone-dir pointing at your existing skydio_x2 folder."
        )

    plants = []
    n_plants = max(1, int(field_length / plant_gap))

    rng = random.Random(seed)

    half = n_rows // 2
    for row_y in (k * row_spacing for k in range(-half, half + 1)):
        # grass strip under the row, 2 mm thick so it never pokes through the soil
        plants.append(
            f'<geom type="box" size="{field_length / 2:.2f} 0.5 0.001" '
            f'pos="{field_length / 2:.2f} {row_y:.2f} 0.001" material="grass" '
            f'contype="0" conaffinity="0"/>'
        )
        for i in range(n_plants):
            x = i * plant_gap + plant_gap / 2
            plants.append(_vine_xml(x, row_y, rng))
        plants.append(_trellis_xml(row_y, 0.0, field_length))
    plants_xml = "\n    ".join(plants)

    scene = f"""<mujoco model="Skydio X2 scene (plantation)">
  <include file="x2.xml"/>
  <visual>
    <headlight diffuse="0.35 0.35 0.35" ambient="0.35 0.35 0.35" specular="0 0 0"/>
    <quality shadowsize="4096"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.42 0.65 0.92" rgb2="0.92 0.95 1.0"
             width="512" height="3072"/>
    <texture name="soil" type="2d" builtin="flat" mark="random" random="0.25"
             rgb1="0.58 0.45 0.31" markrgb="0.42 0.32 0.21" width="128" height="128"/>
    <material name="soil" texture="soil" texrepeat="0.6 0.6" texuniform="true" reflectance="0"/>
    <texture name="grass" type="2d" builtin="flat" mark="random" random="0.35"
             rgb1="0.36 0.52 0.22" markrgb="0.26 0.40 0.15" width="128" height="128"/>
    <material name="grass" texture="grass" texrepeat="1.5 1.5" texuniform="true" reflectance="0"/>
{_vine_assets_xml()}
  </asset>
  <worldbody>
    <light name="sun" pos="0 0 6" dir="0.35 0.25 -1" directional="true" castshadow="true"
           diffuse="0.75 0.72 0.65" specular="0 0 0"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="soil"/>
    {plants_xml}
  </worldbody>
</mujoco>"""
    path = os.path.join(drone_dir, out_name)
    with open(path, "w") as f:
        f.write(scene)
    return path
