"""
Builds the plantation scene XML: the Skydio X2 drone (unmodified) + a plain
floor/sky + two rows of visual-only "crop" markers flanking the flight path,
so the field reads as a plantation instead of a bare plane.

The crop markers are purely cosmetic (contype="0" conaffinity="0" -- no
collision, no effect on physics or training), so this does not change the RL
problem in any way, just the presentation.
"""
import os

DEFAULT_SCENE_NAME = "x2_plantation_scene.xml"


def build_plantation_scene(drone_dir, field_length=20.0, row_spacing=3.0,
                           plant_gap=1.5, out_name=DEFAULT_SCENE_NAME):
    """Writes a scene file next to x2.xml and returns its path.

    field_length: how far the crop rows extend along +x (match your env's
        `field_length` so the visual field matches the flight distance).
    row_spacing: distance between the two crop rows (the drone flies along
        y=0, between them).
    plant_gap: spacing between individual plant markers along each row.
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
    for row_y in (-row_spacing / 2, row_spacing / 2):
        for i in range(n_plants):
            x = i * plant_gap + plant_gap / 2
            plants.append(
                f'<geom type="cylinder" size="0.15 0.2" pos="{x:.2f} {row_y:.2f} 0.2" '
                f'rgba="0.25 0.55 0.2 1" contype="0" conaffinity="0"/>'
            )
    plants_xml = "\n    ".join(plants)

    scene = f"""<mujoco model="Skydio X2 scene (plantation)">
  <include file="x2.xml"/>
  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="flat" rgb1="0.55 0.75 0.9" width="2" height="2"/>
  </asset>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" directional="true"/>
    <geom name="floor" size="0 0 0.05" type="plane" rgba="0.62 0.5 0.35 1"/>
    {plants_xml}
  </worldbody>
</mujoco>"""
    path = os.path.join(drone_dir, out_name)
    with open(path, "w") as f:
        f.write(scene)
    return path
