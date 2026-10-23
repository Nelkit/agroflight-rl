"""
Builds the plantation scene XML: the Skydio X2 drone (unmodified) + a soil
floor, a gradient sky and a sun that casts shadows + two rows of visual-only
vine bushes on grass strips flanking the flight path, so the field reads as a
vineyard instead of a bare plane.

The bushes and grass strips are purely cosmetic (contype="0" conaffinity="0" -- no
collision, no effect on physics or training), so this does not change the RL
problem in any way, just the presentation.
"""
import os
import random

DEFAULT_SCENE_NAME = "x2_plantation_scene.xml"


def _bush_xml(x, y, rng):
    """One vine bush: a short trunk under an irregular canopy of leaf blobs (visual only).

    Each bush gets its own number, placement and shape of blobs (not just a size and colour),
    so no two bushes share a silhouette. `rng` makes the result the same on every run.
    """
    s = rng.uniform(0.80, 1.20)               # overall size of this bush
    h = rng.uniform(0.85, 1.15)               # height stretch: some bushes taller, some flatter
    r, g = rng.uniform(0.12, 0.35), rng.uniform(0.38, 0.62)   # bush tone: dark green ... yellow-green
    x += rng.uniform(-0.10, 0.10)             # not planted on a perfect grid

    trunk_top = 0.3
    trunk = (
        f'<geom type="cylinder" size="0.04 {trunk_top / 2:.2f}" '
        f'pos="{x:.2f} {y:.2f} {trunk_top / 2:.2f}" '
        f'rgba="0.40 0.26 0.13 1" contype="0" conaffinity="0"/>'
    )

    def blob(bx, by, rx, ry, rz, lift):
        """One leaf ellipsoid resting `lift` metres above the trunk top, in a shade of the bush tone."""
        br, bg = r + rng.uniform(-0.04, 0.04), g + rng.uniform(-0.06, 0.06)
        return (
            f'<geom type="ellipsoid" size="{rx:.2f} {ry:.2f} {rz:.2f}" '
            f'pos="{bx:.2f} {by:.2f} {trunk_top + rz + lift:.2f}" '
            f'rgba="{br:.2f} {bg:.2f} 0.12 1" contype="0" conaffinity="0"/>'
        )

    blobs = [blob(x, y, 0.25 * s, 0.22 * s, 0.20 * s * h, 0.0)]   # core blob: always covers the trunk
    for _ in range(rng.randint(3, 6)):
        rr = rng.uniform(0.12, 0.22) * s
        blobs.append(blob(
            x + rng.uniform(-0.25, 0.25) * s,     # along the row
            y + rng.uniform(-0.15, 0.15),         # sideways (kept small: the corridor is at |y| < 1.0)
            rr * rng.uniform(0.85, 1.25),         # each blob gets its own proportions
            rr * rng.uniform(0.85, 1.10),
            rr * rng.uniform(0.75, 0.95) * h,
            rng.uniform(0.0, 0.10) * s,           # some blobs sit higher, breaking the flat top
        ))
    return trunk + "".join(blobs)

def build_plantation_scene(drone_dir, field_length=20.0, row_spacing=3.0,
                           plant_gap=1.1, out_name=DEFAULT_SCENE_NAME, seed=0):
    """Writes a scene file next to x2.xml and returns its path.

    field_length: how far the crop rows extend along +x (match your env's
        `field_length` so the visual field matches the flight distance).
    row_spacing: distance between the two crop rows (the drone flies along
        y=0, between them).
    plant_gap: spacing between bushes along each row.
    seed: fixes the bush variation so the scene is identical on every run.

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

    for row_y in (-row_spacing / 2, row_spacing / 2):
        # grass strip under the row, 2 mm thick so it never pokes through the soil
        plants.append(
            f'<geom type="box" size="{field_length / 2:.2f} 0.5 0.001" '
            f'pos="{field_length / 2:.2f} {row_y:.2f} 0.001" material="grass" '
            f'contype="0" conaffinity="0"/>'
        )
        for i in range(n_plants):
            x = i * plant_gap + plant_gap / 2
            plants.append(_bush_xml(x, row_y, rng=rng))
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
