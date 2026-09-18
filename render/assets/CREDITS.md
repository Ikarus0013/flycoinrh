# Render asset credits

## fly_glowbox.glb — the photoreal fly mesh

- **Model:** "Low Poly House Fly (Diptera)"
- **Author:** Glowbox 3D — https://sketchfab.com/glowbox3d
- **Source:** https://sketchfab.com/3d-models/low-poly-house-fly-diptera-2baa84955f704a4091a274ef4acec24a
- **Mirror used for CI download:** Objaverse (`2baa84955f704a4091a274ef4acec24a`)
- **License:** Creative Commons Attribution 4.0 (CC-BY) — free for commercial use **with attribution**.
- **Textures:** PBR (base colour + normal map) embedded in the GLB.

### Attribution requirement

Any published render/video using this mesh must credit:

> Fly model: "Low Poly House Fly (Diptera)" by Glowbox 3D (sketchfab.com/glowbox3d),
> licensed CC-BY 4.0.

This credit is baked into the shareable video's end card and posted alongside it.

### Why this mesh

Ron picked "photoreal via a free CC0/CC-BY mesh" for the RON-50 shareable video.
The only high-detail CT-scan Drosophila mesh that was CI-downloadable (NHM_Imaging,
1.17M tris) is licensed **CC BY-NC-SA** (non-commercial) — unusable for a business
promo. This Glowbox housefly is the best commercially-usable, textured, account-free
fly mesh, and drops into `load_cc0_model()` with no code change.
