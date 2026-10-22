# Neuroimaging ML practice (nilearn + scikit-learn)

Practice exercises (January 2025) building up to fMRI functional-connectivity analysis. The scripts
are adapted from the official nilearn and scikit-learn example galleries.

## nilearn: fMRI connectivity and decoding

| File | Topic |
|---|---|
| `nilearn/Basics of nilearn.ipynb` | Loading, masking and plotting brain images |
| `nilearn/ageclassification_fc.py` | Child vs. adult classification from connectomes: correlation, partial correlation and tangent-space connectivity compared |
| `nilearn/covariance_fc.py` | Sparse inverse covariance on the MSDL atlas |
| `nilearn/parcellations_fc.py` | Data-driven parcellations of the brain-development fMRI dataset |
| `nilearn/simdata_fc.py` | Group-sparse Gaussian graphs on synthetic data |
| `nilearn/Reconstruction_Miyawaki .ipynb` | Visual-image reconstruction from fMRI (Miyawaki et al., 2008) |
| `nilearn/encoding.ipynb` | Encoding models |
| `nilearn/surface_images.py` | Building surface meshes in memory |

## scikit-learn

| File | Topic |
|---|---|
| `scikit-learn/classifier_comp.py` | Decision boundaries of common classifiers on toy datasets |
| `scikit-learn/feature_discritization.py` | Feature discretization ahead of linear models |

## Run

```bash
pip install nilearn scikit-learn matplotlib
python nilearn/ageclassification_fc.py   # datasets download automatically on first run
```
