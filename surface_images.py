import numpy as np

from nilearn.surface import InMemoryMesh, PolyMesh

# for tetrahedron
left_coords = np.asarray(
    [
        [0, 0, 0],  # vtx 0
        [1, 0, 0],  # vtx 1
        [0, 1, 0],  # vtx 2
        [0, 0, 1],  # vtx 3
    ]
)
left_faces = np.asarray(
    [
        [1, 0, 2],  # face 1: vtx 1, 0, 2
        [0, 1, 3],  # face 2: vtx 0, 1, 3
        [0, 3, 2],  # face 3: vtx 0, 3, 2
        [1, 2, 3],  # face 4: vtx 1, 2, 3
    ]
)
# for pyramid
right_coords = (
    np.asarray(
        [
            [0, 0, 0],  # vtx 0
            [1, 0, 0],  # vtx 1
            [1, 1, 0],  # vtx 2
            [0, 1, 0],  # vtx 3
            [0, 0, 1],  # vtx 4
        ]
    )
    + 2  # shift all coords by 2
)
right_faces = np.asarray(
    [
        [0, 1, 4],  # face 1: vtx 0, 1, 4
        [0, 3, 1],  # face 2: vtx 0, 3, 1
        [1, 3, 2],  # face 3: vtx 1, 3, 2
        [1, 2, 4],  # face 4: vtx 1, 2, 4
        [2, 3, 4],  # face 5: vtx 2, 3, 4
        [0, 4, 3],  # face 6: vtx 0, 4, 3
    ]
)
# combine left and right meshes
mesh = PolyMesh(
    left=InMemoryMesh(left_coords, left_faces),
    right=InMemoryMesh(right_coords, right_faces),
)

rng = np.random.default_rng(0)  # init rng
left_data = rng.random(mesh.parts["left"].n_vertices)  # rand data for left
right_data = rng.random(mesh.parts["right"].n_vertices)  # rand data for right
# combine data into dict
data = {
    "left": left_data,
    "right": right_data,
}

# %%
# Creating surface img
# ---------------------
#
# Combine mesh and data w/ SurfaceImage
from nilearn.surface import SurfaceImage

surface_image = SurfaceImage(mesh=mesh, data=data)

# %%
# Plot surface img
# -----------------
#
# Plot using view_surf for left and right hemispheres
from nilearn.plotting import view_surf

# %%
# plot left hemi
view_surf(
    surf_map=surface_image,
    hemi="left",
)

# %%
# plot right hemi
view_surf(
    surf_map=surface_image,
    hemi="right",
)

# %%
# Data format - GIFTI
# -------------------
#
# Surface data saved as .gii (GIFTI) files
# Can save/load mesh and data via nilearn

# %%
# Save surface img
# -----------------
#
# Save mesh and data to GIFTI files
from pathlib import Path

output_dir = Path.cwd() / "results" / "plot_surface_101"
output_dir.mkdir(exist_ok=True, parents=True)  # create dir
print(f"Output will be saved to: {output_dir}")
surface_image.mesh.to_filename(output_dir / "surface_image_mesh.gii")  # save mesh
surface_image.data.to_filename(output_dir / "surface_image_data.gii")  # save data

# %%
# Resulting files: 4 total: 2 for mesh, 2 for data (left & right)
# Mesh files: _hemi-L.gii (left), _hemi-R.gii (right)

# %%
# Load surface img
# -----------------
#
# Load saved mesh/data back into SurfaceImage
mesh = {
    "left": output_dir / "surface_image_mesh_hemi-L.gii",  # left mesh
    "right": output_dir / "surface_image_mesh_hemi-R.gii",  # right mesh
}
data = {
    "left": output_dir / "surface_image_data_hemi-L.gii",  # left data
    "right": output_dir / "surface_image_data_hemi-R.gii",  # right data
}

surface_image_loaded = SurfaceImage(
    mesh=mesh,
    data=data,
)

# %%
# Plot loaded surface img (left hemi)
view_surf(
    surf_map=surface_image_loaded,
    hemi="left",
)
