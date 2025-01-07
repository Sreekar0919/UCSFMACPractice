# %%
# retrieve atlas & fmri data
# ---------------------------
from nilearn import datasets

atlas = datasets.fetch_atlas_msdl()  # fetch msdl atlas (includes brain regions)
# load atlas img stored in 'maps' (contains regions of interest)
atlas_filename = atlas["maps"]
# load region labels from the atlas
labels = atlas["labels"]

# load fmri dataset for 1 subject
data = datasets.fetch_development_fmri(n_subjects=1)

# print path to functional (4D) data for first subject
print(f"First subject functional nifti images (4D) are at: {data.func[0]}")

# %%
# extract time series for each brain region
# -----------------------------------------
from nilearn.maskers import NiftiMapsMasker

masker = NiftiMapsMasker(
    maps_img=atlas_filename,  # use atlas as mask (region-based)
    standardize="zscore_sample",  # z-score each time series for consistency
    standardize_confounds="zscore_sample",  # z-score confounds (e.g., motion, nuisance)
    memory="nilearn_cache",  # enable caching for efficiency
    verbose=5,  # detailed output for troubleshooting
)

# extract time series data (masking fmri data) and include confounds
time_series = masker.fit_transform(data.func[0], confounds=data.confounds)

# %%
# compute sparse inverse covariance (partial correlations)
# --------------------------------------------------------
from sklearn.covariance import GraphicalLassoCV

estimator = GraphicalLassoCV(alpha=0.05)  # lasso regularization with fine-tuned alpha
estimator.fit(time_series)  # fit model to the time series data

# %%
# display covariance matrix (correlation between regions)
# ------------------------------------------------------
from nilearn import plotting

# plot covariance matrix: each entry is the covariance between pairs of regions
plotting.plot_matrix(
    estimator.covariance_,  # covariance matrix from estimator
    labels=labels,  # region labels
    figure=(9, 7),  # set figure size to match desired output
    vmax=1,  # set max color scale (for visualization)
    vmin=-1,  # set min color scale (for visualization)
    title="covariance",  # plot title
)

# %%
# plot corresponding brain network graph (regions as nodes, covariances as edges)
# ------------------------------------------------------------------------------
coords = atlas.region_coords  # get 3D coords of regions

plotting.plot_connectome(estimator.covariance_, coords, title="covariance")

# %%
# display sparse inverse covariance (partial correlations)
# --------------------------------------------------------
# negate precision matrix to visualize partial correlations (removes direct effects)
plotting.plot_matrix(
    -estimator.precision_,  # precision matrix (partial correlations)
    labels=labels,  # region labels
    figure=(9, 7),  # keep consistent figure size
    vmax=1,  # max scale for color bar (strongest correlation)
    vmin=-1,  # min scale for color bar (strongest negative correlation)
    title="sparse inverse covariance",  # plot title
)

# %%
# plot corresponding brain network graph for partial correlations
# ----------------------------------------------------------------
plotting.plot_connectome(
    -estimator.precision_, coords, title="sparse inverse covariance"
)

plotting.show()  # show all plots

# %%
# 3d visualization in web browser
# -------------------------------
# use view_connectome for interactive, 3D plotting in web browser
# enables exploration of the network (e.g., rotating, zooming)

view = plotting.view_connectome(-estimator.precision_, coords)

view