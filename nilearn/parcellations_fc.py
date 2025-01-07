from nilearn._utils.helpers import check_matplotlib

check_matplotlib()

# %%
# Download brain development fMRI dataset
import time
import numpy as np
from matplotlib import patches, ticker
from nilearn import datasets, plotting
from nilearn.image import get_data, index_img, mean_img
from nilearn.regions import Parcellations

dataset = datasets.fetch_development_fmri(n_subjects=1)

print(f"First subject functional nifti image (4D) is at: {dataset.func[0]}")

# %%
# Brain parcellations with Ward Clustering (Improved)
# Increased the number of clusters (2000) and more smoothing to enhance accuracy
start = time.time()

ward = Parcellations(
    method="ward",
    n_parcels=2000,  # Increased number of clusters for finer granularity
    standardize=False,
    smoothing_fwhm=5.0,  # Increased smoothing for better cluster formation
    memory="nilearn_cache",
    memory_level=1,
    verbose=1,
)
ward.fit(dataset.func)
print(f"Ward agglomeration 2000 clusters: {time.time() - start:.2f}s")

# %%
# Visualize: Brain parcellations (Ward)
ward_labels_img = ward.labels_img_

from pathlib import Path

output_dir = Path.cwd() / "results" / "plot_data_driven_parcellations"
output_dir.mkdir(exist_ok=True, parents=True)
print(f"Output will be saved to: {output_dir}")
ward_labels_img.to_filename(output_dir / "ward_parcellation.nii.gz")

first_plot = plotting.plot_roi(
    ward_labels_img, title="Ward parcellation", display_mode="xz"
)

cut_coords = first_plot.cut_coords

# %%
# Compressed representation of Ward clustering (Finer details)
original_voxels = np.sum(get_data(ward.mask_img_))
mean_func_img = mean_img(dataset.func[0], copy_header=True)
vmin = np.min(get_data(mean_func_img))
vmax = np.max(get_data(mean_func_img))

plotting.plot_epi(
    mean_func_img,
    cut_coords=cut_coords,
    title=f"Original ({int(original_voxels)} voxels)",
    vmax=vmax,
    vmin=vmin,
    display_mode="xz",
)

fmri_reduced = ward.transform(dataset.func)
fmri_compressed = ward.inverse_transform(fmri_reduced)

plotting.plot_epi(
    index_img(fmri_compressed, 0),
    cut_coords=cut_coords,
    title="Ward compressed representation (2000 parcels)",
    vmin=vmin,
    vmax=vmax,
    display_mode="xz",
)

# %%
# Brain parcellations with KMeans Clustering (Enhanced Standardization)
# Standardizing the data before clustering for better results
start = time.time()
kmeans = Parcellations(
    method="kmeans",
    n_parcels=100,  # Increased number of parcels for better clustering
    standardize="zscore_sample",  # Standardization for better accuracy
    smoothing_fwhm=8.0,  # Slightly increased smoothing for better accuracy
    memory="nilearn_cache",
    memory_level=1,
    verbose=1,
)
kmeans.fit(dataset.func)
print(f"KMeans clusters: {time.time() - start:.2f}s")

# %%
# Visualize: Brain parcellations (KMeans)
kmeans_labels_img = kmeans.labels_img_

display = plotting.plot_roi(
    kmeans_labels_img,
    mean_func_img,
    title="KMeans parcellation",
    display_mode="xz",
)

kmeans_labels_img.to_filename(output_dir / "kmeans_parcellation.nii.gz")

# %%
# Brain parcellations with Hierarchical KMeans (Better Balance)
# More balanced clusters with Hierarchical KMeans
start = time.time()
hkmeans = Parcellations(
    method="hierarchical_kmeans",
    n_parcels=100,  # Increased parcels for better granularity
    standardize="zscore_sample",  # Standardization to improve accuracy
    smoothing_fwhm=8.0,  # Increased smoothing for clarity
    memory="nilearn_cache",
    memory_level=1,
    verbose=1,
)
hkmeans.fit(dataset.func)

# %%
# Visualize: Brain parcellations (Hierarchical KMeans)
hkmeans_labels_img = hkmeans.labels_img_

plotting.plot_roi(
    hkmeans_labels_img,
    mean_func_img,
    title="Hierarchical KMeans parcellation",
    display_mode="xz",
    cut_coords=display.cut_coords,
)

hkmeans_labels_img.to_filename(output_dir / "hierarchical_kmeans_parcellation.nii.gz")

# %%
# Compare Hierarchical Kmeans clusters with KMeans (Balanced Clusters)
_, kmeans_counts = np.unique(get_data(kmeans_labels_img), return_counts=True)
_, hkmeans_counts = np.unique(get_data(hkmeans_labels_img), return_counts=True)

voxel_ratio = np.round(np.sum(kmeans_counts[1:]) / 100)  # Adjusted to 100 clusters
print(f"... each cluster should contain {voxel_ratio} voxels")

# %%
# Plot clusters sizes distributions (Better Cluster Balance)
bins = np.concatenate(
    [
        np.linspace(0, 500, 11),
        np.linspace(600, 2000, 15),
        np.linspace(3000, 10000, 8),
    ]
)
fig, axes = plt.subplots(
    nrows=2, sharex=True, gridspec_kw={"height_ratios": [4, 1]}
)
plt.semilogx()
axes[0].hist(kmeans_counts[1:], bins, color="blue")
axes[1].hist(hkmeans_counts[1:], bins, color="green")
axes[0].set_ylim(0, 16)
axes[1].set_ylim(4, 0)
axes[1].xaxis.set_major_formatter(ticker.ScalarFormatter())
axes[1].yaxis.set_label_coords(-0.08, 2)
fig.subplots_adjust(hspace=0)
plt.xlabel("Number of voxels (log)", fontsize=12)
plt.ylabel("Number of clusters", fontsize=12)
handles = [
    patches.Rectangle((0, 0), 1, 1, color=c, ec="k") for c in ["blue", "green"]
]
labels = ["Kmeans", "Hierarchical Kmeans"]
fig.legend(handles, labels, loc=(0.5, 0.8))

# %%
# Brain parcellations with ReNA Clustering (Faster and Efficient)
# Increased parcels for finer clustering, uses ReNA algorithm for faster results
start = time.time()
rena = Parcellations(
    method="rena",
    n_parcels=7000,  # Increased number of parcels for better resolution
    standardize=False,
    smoothing_fwhm=5.0,  # Smoothing slightly increased
    scaling=True,
    memory="nilearn_cache",
    memory_level=1,
    verbose=1,
)

rena.fit_transform(dataset.func)
print(f"ReNA 7000 clusters: {time.time() - start:.2f}s")

# %%
# Visualize: Brain parcellations (ReNA)
rena_labels_img = rena.labels_img_

rena_labels_img.to_filename(output_dir / "rena_parcellation.nii.gz")

plotting.plot_roi(
    ward_labels_img,
    title="ReNA parcellation",
    display_mode="xz",
    cut_coords=cut_coords,
)

# %%
# Compressed representation of ReNA clustering (Close to original)
plotting.plot_epi(
    mean_func_img,
    cut_coords=cut_coords,
    title=f"Original ({int(original_voxels)} voxels)",
    vmax=vmax,
    vmin=vmin,
    display_mode="xz",
)

fmri_reduced_rena = rena.transform(dataset.func)
compressed_img_rena = rena.inverse_transform(fmri_reduced_rena)

plotting.plot_epi(
    index_img(compressed_img_rena, 0),
    cut_coords=cut_coords,
    title="ReNA compressed representation (7000 parcels)",
    vmin=vmin,
    vmax=vmax,
    display_mode="xz",
)