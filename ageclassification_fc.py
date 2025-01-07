"""
brain connectivity estimation with nilearn
===========================================

this example uses fMRI data to estimate brain connectivity for children and adults, comparing correlation, partial correlation, and tangent connectivity.
"""

from nilearn import datasets, plotting

# %%
# load brain development fMRI dataset and msdl atlas
# ---------------------------------------------------
# study 30 subjects for quicker computations
development_dataset = datasets.fetch_development_fmri(n_subjects=30)

# %%
# use msdl probabilistic rois for region-based analysis
msdl_data = datasets.fetch_atlas_msdl()  # fetch msdl atlas
msdl_coords = msdl_data.region_coords  # roi coordinates for plotting
n_regions = len(msdl_coords)  # number of rois in msdl
print(f"msdl has {n_regions} rois, part of the following networks:\n{msdl_data.networks}")

# %%
# region signals extraction
# ---------------------------
# instantiate nifti masker to extract time series from rois
from nilearn.maskers import NiftiMapsMasker

masker = NiftiMapsMasker(
    msdl_data.maps,  # roi map
    resampling_target="data",  # match fMRI data resolution
    t_r=2,  # repetition time (in seconds)
    detrend=True,  # remove linear trends from data
    low_pass=0.1,  # high freq cutoff (Hz)
    high_pass=0.01,  # low freq cutoff (Hz)
    memory="nilearn_cache",  # cache results for speed
    memory_level=1,  # cache level for optimization
    standardize="zscore_sample",  # z-score standardization
    standardize_confounds="zscore_sample",  # standardize confounds
).fit()

# %%
# extract region signals for each subject
children = []  # store children group
pooled_subjects = []  # all subjects data
groups = []  # child/adult group labels

for func_file, confound_file, phenotypic in zip(
    development_dataset.func,  # functional data
    development_dataset.confounds,  # confound regressors
    development_dataset.phenotypic,  # phenotypic info (child/adult)
):
    time_series = masker.transform(func_file, confounds=confound_file)  # extract roi time series
    pooled_subjects.append(time_series)  # add to pooled subjects
    if phenotypic["Child_Adult"] == "child":
        children.append(time_series)  # add to children group
    groups.append(phenotypic["Child_Adult"])

print(f"data has {len(children)} children.")

# %%
# roi-to-roi correlations for children
# -----------------------------------
# use ConnectivityMeasure to compute correlation (full connectivity) between rois
from nilearn.connectome import ConnectivityMeasure

correlation_measure = ConnectivityMeasure(
    kind="correlation",  # use correlation measure
    standardize="zscore_sample",  # z-score standardization
)

# %%
# compute correlation matrices for each child
correlation_matrices = correlation_measure.fit_transform(children)
print(f"correlations of children are stacked in an array of shape {correlation_matrices.shape}")

# %%
# mean correlation matrix over all children
mean_correlation_matrix = correlation_measure.mean_
print(f"mean correlation has shape {mean_correlation_matrix.shape}.")

# %%
# plot correlation matrices of the first 3 children
import numpy as np
from matplotlib import pyplot as plt

_, axes = plt.subplots(1, 3, figsize=(15, 5))
vmax = np.absolute(correlation_matrices).max()  # scale color bar
for i, (matrix, ax) in enumerate(zip(correlation_matrices, axes)):
    plotting.plot_matrix(
        matrix,
        tri="lower",  # lower triangle matrix (no redundant data)
        colorbar=True,  # include color bar for scale
        axes=ax,  # assign axes for each plot
        title=f"correlation, child {i}",
        vmax=vmax,  # max value for color scale
        vmin=-vmax,  # min value for color scale
    )

# %%
# plot mean correlation matrix across all children
plotting.plot_connectome(
    mean_correlation_matrix,
    msdl_coords,
    title="mean correlation over all children",
)

# %%
# studying partial correlations
# -----------------------------
# study direct connections via partial correlation coefficients
partial_correlation_measure = ConnectivityMeasure(
    kind="partial correlation",  # partial correlation for direct connections
    standardize="zscore_sample",  # z-score standardization
)
partial_correlation_matrices = partial_correlation_measure.fit_transform(children)

# %%
# plot partial correlation matrices of the first 3 children
_, axes = plt.subplots(1, 3, figsize=(15, 5))
vmax = np.absolute(partial_correlation_matrices).max()  # scale color bar
for i, (matrix, ax) in enumerate(zip(partial_correlation_matrices, axes)):
    plotting.plot_matrix(
        matrix,
        tri="lower",  # lower triangle matrix
        colorbar=True,  # include color bar
        axes=ax,  # assign axes
        title=f"partial correlation, child {i}",
        vmax=vmax,  # max color scale
        vmin=-vmax,  # min color scale
    )

# %%
# plot mean partial correlation matrix across all children
plotting.plot_connectome(
    partial_correlation_measure.mean_,
    msdl_coords,
    title="mean partial correlation over all children",
)

# %%
# extract subject variability from group connectivity
# ----------------------------------------------------
# capture subject-to-group variability via tangent space embedding
tangent_measure = ConnectivityMeasure(
    kind="tangent",  # use tangent space for variability
    standardize="zscore_sample",  # z-score standardization
)

# %%
# fit group connectivity matrix and individual deviations from it
tangent_matrices = tangent_measure.fit_transform(children)

# %%
# visualize individual deviations from group connectivity
_, axes = plt.subplots(1, 3, figsize=(15, 5))
for i, (matrix, ax) in enumerate(zip(tangent_matrices, axes)):
    plotting.plot_matrix(
        matrix,
        tri="lower",  # lower triangle matrix
        colorbar=True,  # include color bar
        axes=ax,  # assign axes
        title=f"tangent offset, child {i}",
    )

# %%
# classification accuracy of connectivity measures
# ------------------------------------------------
# use cross-validation to compare classification accuracy for different connectivity measures
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.svm import LinearSVC

kinds = ["correlation", "partial correlation", "tangent"]  # different connectivity types
_, classes = np.unique(groups, return_inverse=True)  # encode class labels (child/adult)
cv = StratifiedShuffleSplit(n_splits=15, random_state=0, test_size=5)  # stratified CV for balanced splits
pooled_subjects = np.asarray(pooled_subjects)

scores = {}
for kind in kinds:
    scores[kind] = []
    for train, test in cv.split(pooled_subjects, classes):  # iterate over splits
        connectivity = ConnectivityMeasure(
            kind=kind,
            vectorize=True,  # vectorize connectivity matrix for classification
            standardize="zscore_sample",  # z-score standardization
        )
        connectomes = connectivity.fit_transform(pooled_subjects[train])  # build training connectomes
        classifier = LinearSVC(dual=True).fit(connectomes, classes[train])  # fit linear SVM classifier
        predictions = classifier.predict(
            connectivity.transform(pooled_subjects[test])  # predict on test set
        )
        scores[kind].append(accuracy_score(classes[test], predictions))  # store accuracy for this fold

# %%
# display classification results
mean_scores = [np.mean(scores[kind]) for kind in kinds]  # mean accuracy for each connectivity type
scores_std = [np.std(scores[kind]) for kind in kinds]  # std deviation of accuracy

plt.figure(figsize=(6, 4), constrained_layout=True)

positions = np.arange(len(kinds)) * 0.1 + 0.1
plt.barh(positions, mean_scores, align="center", height=0.05, xerr=scores_std)  # plot accuracy with error bars
yticks = [k.replace(" ", "\n") for k in kinds]  # add line breaks to labels
plt.yticks(positions, yticks)
plt.gca().grid(True)
plt.gca().set_axisbelow(True)
plt.gca().axvline(0.8, color="red", linestyle="--")  # add chance level line
plt.xlabel("classification accuracy\n(red line = chance level)")

plotting.show()
