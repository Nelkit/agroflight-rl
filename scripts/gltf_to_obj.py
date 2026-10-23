"""
Convert a downloaded plant model (glTF with a separate .bin) into the files the
plantation scene loads: one Z-up OBJ per mesh node and one PNG colour texture.

MuJoCo reads OBJ, not glTF. Node transforms are applied as glTF defines them
(Y-up world), then the result is turned Z-up for MuJoCo, so models exported by
different tools (Poly Haven, Sketchfab) come out standing upright. When the
leaves are cut out with a separate opacity map, pass it with --alpha: MuJoCo
only does cut-outs when the texture itself carries the alpha channel. Scans
are often too heavy to repeat along a row; --target-triangles simplifies each
mesh and keeps its texture mapping (needs `pip install fast-simplification`,
imported only when the flag is used, so the project gains no dependency).

    python scripts/gltf_to_obj.py scene.gltf src/agroflight/envs/assets/grapevine \
        --name grapevine --target-triangles 12000
    python scripts/gltf_to_obj.py shrub_03_1k.gltf src/agroflight/envs/assets/shrub_03 --alpha alpha_1k.png
"""
import argparse
import json
import os

import numpy as np
from PIL import Image

N_COMPONENTS = {"SCALAR": 1, "VEC2": 2, "VEC3": 3}
DTYPES = {5126: np.float32, 5123: np.uint16, 5125: np.uint32}


def read_accessor(gltf, buffer, index):
    accessor = gltf["accessors"][index]
    view = gltf["bufferViews"][accessor["bufferView"]]
    offset = view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
    n = N_COMPONENTS[accessor["type"]]
    data = np.frombuffer(buffer, DTYPES[accessor["componentType"]], accessor["count"] * n, offset)
    return data.reshape(accessor["count"], n) if n > 1 else data


def local_matrix(node):
    """4x4 transform of one node, from its `matrix` or its translation/rotation/scale."""
    if "matrix" in node:
        return np.array(node["matrix"], dtype=float).reshape(4, 4).T     # glTF stores it column-major
    x, y, z, w = node.get("rotation", [0, 0, 0, 1])
    rotation = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    m = np.eye(4)
    m[:3, :3] = rotation * np.array(node.get("scale", [1, 1, 1]))
    m[:3, 3] = node.get("translation", [0, 0, 0])
    return m


def mesh_nodes(gltf):
    """Yield (node, world matrix) for every node that holds a mesh, walking the scene tree."""
    stack = [(i, np.eye(4)) for i in gltf["scenes"][gltf.get("scene", 0)]["nodes"]]
    while stack:
        index, parent = stack.pop()
        node = gltf["nodes"][index]
        world = parent @ local_matrix(node)
        if "mesh" in node:
            yield node, world
        stack.extend((child, world) for child in node.get("children", []))


def simplify(vertices, uvs, faces, target):
    """Quadric decimation down to about `target` triangles; each kept vertex keeps the UV of a vertex merged into it."""
    import fast_simplification

    _, _, collapses = fast_simplification.simplify(vertices, faces, target_count=target, return_collapses=True)
    vertices, faces, mapping = fast_simplification.replay_simplification(vertices, faces, collapses)
    new_uvs = np.zeros((len(vertices), 2))
    kept = mapping >= 0
    new_uvs[mapping[kept]] = uvs[kept]
    degenerate = (faces[:, 0] == faces[:, 1]) | (faces[:, 1] == faces[:, 2]) | (faces[:, 0] == faces[:, 2])
    return vertices, new_uvs, faces[~degenerate]


def write_obj(path, vertices, uvs, faces):
    with open(path, "w") as f:
        f.writelines(f"v {x:.5f} {y:.5f} {z:.5f}\n" for x, y, z in vertices)
        f.writelines(f"vt {u:.5f} {v:.5f}\n" for u, v in uvs)
        f.writelines(f"f {a}/{a} {b}/{b} {c}/{c}\n" for a, b, c in faces + 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("gltf")
    parser.add_argument("out_dir")
    parser.add_argument("--name", help="Output name (default: node name per mesh, material name for the texture)")
    parser.add_argument("--alpha", help="Separate opacity map to store in the texture's alpha channel")
    parser.add_argument("--texture-size", type=int, default=1024)
    parser.add_argument("--target-triangles", type=int, help="Simplify each mesh to about this many triangles")
    args = parser.parse_args()

    gltf = json.load(open(args.gltf))
    base = os.path.dirname(os.path.abspath(args.gltf))
    buffer = open(os.path.join(base, gltf["buffers"][0]["uri"]), "rb").read()
    os.makedirs(args.out_dir, exist_ok=True)

    nodes = list(mesh_nodes(gltf))
    for node, world in nodes:
        prim = gltf["meshes"][node["mesh"]]["primitives"][0]
        pos = read_accessor(gltf, buffer, prim["attributes"]["POSITION"]).astype(float)
        uv = read_accessor(gltf, buffer, prim["attributes"]["TEXCOORD_0"]).astype(float)
        faces = read_accessor(gltf, buffer, prim["indices"]).astype(int).reshape(-1, 3)

        pos = pos @ world[:3, :3].T + world[:3, 3]               # node transforms, glTF Y-up world
        verts = np.c_[pos[:, 0], -pos[:, 2], pos[:, 1]]          # Y-up -> MuJoCo Z-up
        verts[:, :2] -= verts[:, :2].mean(axis=0)                # centred on the origin...
        verts[:, 2] -= verts[:, 2].min()                         # ...with its base on z = 0
        uv[:, 1] = 1.0 - uv[:, 1]                                # glTF v runs top-down, OBJ bottom-up
        if args.target_triangles and args.target_triangles < len(faces):
            verts, uv, faces = simplify(verts, uv, faces, args.target_triangles)
        name = args.name if args.name and len(nodes) == 1 else node["name"]
        write_obj(os.path.join(args.out_dir, name + ".obj"), verts, uv, faces)
        print(f"{name}: {len(verts)} vertices, {len(faces)} triangles, size {np.ptp(verts, axis=0).round(3)} m")

    material = gltf["materials"][0]
    image = gltf["images"][gltf["textures"][material["pbrMetallicRoughness"]["baseColorTexture"]["index"]]["source"]]
    Image.MAX_IMAGE_PIXELS = None                                # scans ship 8k textures
    texture = Image.open(os.path.join(base, image["uri"])).convert("RGB")
    texture = texture.resize((args.texture_size, args.texture_size), Image.LANCZOS)
    if args.alpha:
        alpha = np.asarray(Image.open(args.alpha)).astype(float)
        alpha = Image.fromarray((255 * alpha / alpha.max()).astype(np.uint8))   # 16-bit maps -> 8-bit
        texture.putalpha(alpha.resize(texture.size, Image.LANCZOS))
    texture_path = os.path.join(args.out_dir, (args.name or material["name"]) + "_diff.png")
    texture.save(texture_path)
    print("texture:", texture_path)


if __name__ == "__main__":
    main()
