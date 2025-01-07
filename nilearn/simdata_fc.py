from nilearn import plotting

# %%
# generate synthetic data for group sparse gaussian graphs
# --------------------------------------------------------
from nilearn._utils.data_gen import generate_group_sparse_gaussian_graphs

n_subjects = 20  # num subjects (simulation)
n_displayed = 3  # num subjects to display
# generate group sparse gaussian graphs for connectivity structure
subjects, precisions, _ = generate_group_sparse_gaussian_graphs(
    n_subjects=n_subjects,
    n_features=10,  # num features (nodes in graph)
    min_n_samples=30,  # min num samples per subject
    max_n_samples=50,  # max num samples per subject
    density=0.1,  # sparsity (density of edges)
)

# %%
# run connectome estimations & plot results
# -----------------------------------------
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(10, 7))  # create figure
plt.subplots_adjust(hspace=0.4)  # adjust subplot spacing
for n in range(n_displayed):  # iterate over subjects to display
    ax = plt.subplot(n_displayed, 4, 4 * n + 1)  # create subplot
    max_precision = precisions[n].max()  # find max precision (value in matrix)
    # plot precision matrix (ground truth connectivity)
    plotting.plot_matrix(
        precisions[n],
        vmin=-max_precision,  # min color scale
        vmax=max_precision,  # max color scale
        axes=ax,
        colorbar=False,  # no colorbar for now
    )

    if n == 0:
        plt.title("ground truth")  # add title to first plot
    plt.ylabel(f"subject {int(n)}")  # label subjects

# %%
# run group-sparse covariance on all subjects
# -------------------------------------------
from nilearn.connectome import GroupSparseCovarianceCV

gsc = GroupSparseCovarianceCV(max_iter=50, verbose=1)  # init model (max 50 iterations)
gsc.fit(subjects)  # fit group-sparse model

# plot group-sparse covariance results for subjects
for n in range(n_displayed):  # display results for few subjects
    ax = plt.subplot(n_displayed, 4, 4 * n + 2)  # create subplot
    max_precision = gsc.precisions_[..., n].max()  # find max precision
    # plot group-sparse precision matrix (connectivity)
    plotting.plot_matrix(
        gsc.precisions_[..., n],
        axes=ax,
        vmin=-max_precision,  # min color scale
        vmax=max_precision,  # max color scale
        colorbar=False,  # no colorbar
    )
    if n == 0:
        plt.title(f"group-sparse\n$\\alpha={gsc.alpha_:.2f}$")  # add alpha value

# %%
# fit one graph lasso per subject
# ------------------------------
from sklearn.covariance import GraphicalLassoCV

gl = GraphicalLassoCV(verbose=1)  # init graph lasso model (cross-validation)

# fit model for each subject & display results
for n, subject in enumerate(subjects[:n_displayed]):  # loop over subjects
    gl.fit(subject)  # fit model to subject data

    ax = plt.subplot(n_displayed, 4, 4 * n + 3)  # create subplot
    max_precision = gl.precision_.max()  # max precision value
    # plot graph lasso precision matrix
    plotting.plot_matrix(
        gl.precision_,
        axes=ax,
        vmin=-max_precision,  # min color scale
        vmax=max_precision,  # max color scale
        colorbar=False,  # no colorbar
    )
    if n == 0:
        plt.title("graph lasso")  # title for first plot
    plt.ylabel(f"$\\alpha={gl.alpha_:.2f}$")  # add alpha value to label

# %%
# fit one graph lasso for all subjects at once
# -------------------------------------------
import numpy as np

gl.fit(np.concatenate(subjects))  # fit model to all subjects combined

ax = plt.subplot(n_displayed, 4, 4)  # final subplot for all subjects
max_precision = gl.precision_.max()  # find max precision
# plot combined graph lasso precision matrix
plotting.plot_matrix(
    gl.precision_,
    axes=ax,
    vmin=-max_precision,  # min color scale
    vmax=max_precision,  # max color scale
    colorbar=False,  # no colorbar
)
plt.title(f"graph lasso, all subjects\n$\\alpha={gl.alpha_:.2f}$")  # title with alpha

plotting.show()  # show all plots