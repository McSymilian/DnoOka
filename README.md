# Retinal Blood Vessel Segmentation — CHASE_DB1

Jupyter Notebook application for automatic detection of blood vessels in fundus eye images.
For each pixel the algorithm determines whether it belongs to a blood vessel (positive class)
or to the background (negative class), i.e. binary per-pixel classification.

The project is implemented at three requirement levels using the same dataset and the same
independent test set throughout, which enables a fair comparison of all methods.

---

## Dataset

**CHASE_DB1** — 28 RGB retinal fundus images (999 x 960 px) with corresponding binary
ground-truth masks produced by the first human observer (1stHO).
Images cover 14 subjects, each with a left-eye (L) and right-eye (R) scan.

Source: https://blogs.kingston.ac.uk/retinal/chasedb1/

Expected directory layout after download:

```
resource/
  CHASEDB1/
    images/   Image_01L.png ... Image_14R.png
    masks/    Image_01L_1stHO.png ... Image_14R_1stHO.png
```

**Train / test split:** first 20 images (subjects 01–10) for training,
last 8 images (subjects 11–14) as the hold-out test set.
Blood vessel pixels constitute approximately 10 % of all FOV pixels — a strongly
imbalanced binary classification problem.

---

## Methods

### Level 3.0 — Classical image processing (baseline)

A three-stage pipeline without any learning component:

1. **Pre-processing** — green channel extraction, CLAHE (adaptive histogram equalisation),
   median blur for noise suppression.
2. **Vessel detection** — multi-scale Frangi vesselness filter (`black_ridges=True`,
   sigmas 1–7) detecting dark tubular structures.
3. **Post-processing** — percentile thresholding within the FOV, morphological closing,
   hole filling, and small-object removal.

### Level 4.0 — Machine learning on patch features

After pre-processing, a 5 x 5 pixel patch is centred on every pixel and the following
19 features are computed via vectorised convolution over the full image:

- Mean and variance of the green channel; variance of the red and blue channels.
- Response of the Frangi filter.
- Seven normalised central moments (eta_20, eta_02, eta_11, eta_30, eta_03, eta_21, eta_12).
- Seven Hu invariant moments (log-scale).

The label for each sample is taken from the expert mask at the central pixel.
Class imbalance is addressed by per-image RandomUnderSampler (imbalanced-learn).
Classifier: **Random Forest** (scikit-learn, 80 trees, max depth 18, balanced class weights).
All 20 training images are used for feature extraction.

### Level 5.0 — Deep neural network (U-Net)

A convolutional encoder–decoder network with skip connections trained on whole images
resized to 512 x 512. Input: single-channel preprocessed green image (CLAHE-normalised).
Loss: combined BCE (pos_weight = 3.0) and soft Dice loss to handle class imbalance.
Framework: **PyTorch**. Device: MPS (Apple Silicon) / CUDA / CPU (selected automatically).

---

## Evaluation

All metrics are computed exclusively within the Field-of-View (FOV) mask, which excludes
the dark corners outside the retinal disc. The following measures are reported per image
and as means over the test set:

| Metric | Formula |
|---|---|
| Accuracy | (TP + TN) / (TP + TN + FP + FN) |
| Sensitivity (TPR) | TP / (TP + FN) |
| Specificity (TNR) | TN / (TN + FP) |
| Precision | TP / (TP + FP) |
| Balanced accuracy | (Sensitivity + Specificity) / 2 |
| G-mean | sqrt(Sensitivity x Specificity) |
| Dice (F1) | 2 TP / (2 TP + FP + FN) |
| MCC | (TP x TN - FP x FN) / sqrt(...) |

Balanced accuracy and G-mean are the primary measures because plain accuracy is inflated
by the dominant background class (~90 % of FOV pixels).

Vessel pixels are treated as the positive class; background pixels as the negative class.
A confusion matrix (TP, FP, FN, TN) is provided for each method.

---

## Results (hold-out test set, 8 images)

| Method | Accuracy | Sensitivity | Specificity | Balanced Acc | G-mean | Dice |
|---|---|---|---|---|---|---|
| 3.0 Classical (Frangi) | 0.752 | 0.773 | 0.751 | 0.762 | 0.761 | 0.370 |
| 4.0 Random Forest | 0.859 | 0.776 | 0.867 | 0.822 | 0.819 | 0.510 |
| 5.0 U-Net | 0.929 | 0.875 | 0.935 | 0.905 | 0.904 | 0.698 |

Results shown are from the full training configuration (`FAST=False`, all 20 training images,
12 U-Net epochs). Re-running the notebook with `LOAD_MODELS=True` reproduces the same values
from the saved model files without retraining.

---

## Repository Structure

```
DnoOka.ipynb            main notebook (all code and results)
requirements.txt        Python dependencies
resource/
  CHASEDB1/             dataset (not committed — download separately)
    images/
    masks/
models/
  rf_pipeline.joblib    saved Random Forest pipeline (generated on first run)
  unet_weights.pth      saved U-Net weights (generated on first run)
```

---

## Dependencies

```
numpy
pandas
matplotlib
scikit-image
scikit-learn
scipy
opencv-python-headless
imbalanced-learn
joblib
torch
```

Install with:

```bash
pip install -r requirements.txt
```

---

## Usage

1. Download CHASE_DB1 and place images and masks under `resource/CHASEDB1/` as shown above.
2. Open `DnoOka.ipynb` in JupyterLab or Jupyter Notebook.
3. Set `LOAD_MODELS = True` at the top of cell 1 to reuse previously saved models,
   or `LOAD_MODELS = False` to retrain from scratch.
4. Set `CFG["FAST"] = False` for full results (default) or `True` for a quick demo
   with fewer epochs and training samples.
5. Run all cells in order (Kernel > Restart & Run All).

Trained models are saved automatically to `models/` on the first run and loaded on
subsequent runs when `LOAD_MODELS = True`.

---

## References

- CHASE_DB1 database: https://blogs.kingston.ac.uk/retinal/chasedb1/
- scikit-learn: https://scikit-learn.org/stable/
- imbalanced-learn: https://imbalanced-learn.org/stable/
- P. Liskowski, K. Krawiec: Segmenting Retinal Blood Vessels With Deep Neural Networks,
  IEEE Transactions on Medical Imaging, 2016. https://ieeexplore.ieee.org/document/7440871
