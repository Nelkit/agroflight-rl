# Grapevine (visual asset for the plantation scene)

This work is based on ["Day 219: Grapevine plant - 1scanaday"](https://sketchfab.com/3d-models/day-219-grapevine-plant-1scanaday-f59fa3cd369941af8bab9716aac768e2)
by [emm](https://sketchfab.com/edemaistre), licensed under
[CC BY 4.0](http://creativecommons.org/licenses/by/4.0/).

- **What it is:** a photogrammetry scan (iPhone) of a real bush-trained ("gobelet") grapevine in a vineyard in Cassis, southern France, with the stony soil around its foot. Real scale: about 0.83 m tall and 0.75 m across.
- **License requirement:** the author must be credited wherever the scene is shared (README, report, slides, demo video).
- **Used by:** `agroflight/envs/scene_builder.py` (viewer only; training and evaluation never load it).

## Changes from the original

| Original | Here |
| -------- | ---- |
| glTF (Y-up) with a Sketchfab root transform | OBJ, Z-up, centred, base on z = 0 |
| 77 572 triangles | 12 000 (quadric simplification, texture mapping kept): 24 full-detail vines rendered at 11 fps, which is too slow for the live viewer |
| 8192 × 8192 colour texture (99 MB) | 1024 × 1024 (`grapevine_diff.png`) |

At viewer distance the simplified vine is indistinguishable from the original.

## Regenerating

Download the glTF from the Sketchfab page (it needs a free account), unzip it,
then:

```bash
pip install fast-simplification   # only needed for --target-triangles
python scripts/gltf_to_obj.py scene.gltf src/agroflight/envs/assets/grapevine \
    --name grapevine --target-triangles 12000
```
