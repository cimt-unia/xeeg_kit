# MEEGKit Artifact Cleaning Framework

This document provides an in-depth, mathematically grounded explanation of the artifact cleaning pipeline implemented in `xeegkit` via the `execute_meegkit` function. It leverages three core algorithms vendored from `meegkit`: **Artifact Subspace Reconstruction (ASR)**, **Sparse Time-Artifact Removal (STAR)**, and **Sensor Noise Suppression (SNS)**.

<br>

## **The Framework**
1. **ASR:** Fix massive movement/burst artifacts first so they don't corrupt the spatial statistics.
2. **STAR:** Fix random, isolated electrode pops using global spatial correlation.
3. **SNS:** Apply a final continuous spatial filter to suppress baseline thermal hiss across all channels.

<br>

## **Summary Table**

| Feature | **ASR** | **STAR** | **SNS** |
| :--- | :--- | :--- | :--- |
| **Unit of Operation** | **Time Window** (e.g., 500ms chunks) | **Single Time Point** (sample-by-sample) | **Continuous Signal** (Entire timeline) |
| **Channels Used** | **All Channels** (Multivariate PCA) | **All Other Channels** (Linear Prediction) | **Most Correlated Channels** (Statistical neighbors) |
| **Math Core** | Covariance Eigenvalue Thresholding | Linear Regression / Orthogonal Projection | Spatial Projection / Regression |
| **Analogy** | "This 0.5s snapshot has impossible variance; project out the bad spatial directions." | "Channel 3 just spiked; ignore it and calculate what it *should* be based on the rest of the scalp." | "Every channel has a little bit of static; filter it out by averaging the shared brain signal across correlated channels." |
| **Corrects** | Bursts, movements, high-variance transients | Sparse spikes, electrode pops, muscle twitches | Continuous broadband thermal sensor noise |

<br>

## **Pipeline Flow**

In `xeegkit`, the algorithms are applied strictly in this dependency chain:
1. **ASR:** Removes macroscopic artifacts. If left uncorrected, their massive variance would dominate the covariance matrices, rendering subsequent steps useless.
2. **STAR:** Removes microscopic, channel-specific spikes. Because ASR has already stabilized the global variance, STAR can safely compute a reliable baseline covariance.
3. **SNS:** Applies a gentle, continuous spatial filter to maximize the signal-to-noise ratio of the remaining clean neural activity.



<br>

## 1. Artifact Subspace Reconstruction (ASR)

### **Core Intuition**
ASR operates on **multivariate time windows**. It assumes genuine EEG has a stable, predictable spatial covariance structure. Transient artifacts (blinks, jaw clenching, head movements) violate this structure by introducing abnormally high variance in specific spatial directions (subspaces).

### **Mathematical Mechanics**
1. **Spectral Shaping:** Before analysis, data is passed through a Yule-Walk IIR filter. This shapes the spectrum to be *more* sensitive to delta (blinks) and gamma (muscle) frequencies, and *less* sensitive to alpha/beta rhythms, preventing genuine brain activity from triggering false positives.
2. **Calibration (`fit`):** ASR finds a "clean" 30-second segment. It computes a robust average covariance matrix `C_bar` using the **Oracle Approximating Shrinkage (OAS)** estimator, which is mathematically optimal for high-dimensional data (many channels, limited samples). It derives a mixing matrix `M = sqrt(C_bar)`, projects the data, and fits a truncated generalized Gaussian distribution to establish a baseline mean (`μ`) and standard deviation (`σ`) for each principal component.
3. **Processing (`transform`):** For each new time window, ASR computes its running covariance `C_run` and performs PCA to find eigenvectors `V` and eigenvalues `D` (variances).
4. **Subspace Rejection:** An eigenvector `v_j` is flagged as an artifact if its variance exceeds the directional threshold: `D_jj > ||T * v_j||^2`, where `T` is the threshold matrix (`μ + cutoff × σ`).
5. **Reconstruction:** A reconstruction matrix `R` is built that preserves "clean" eigenvectors and zeroes out "artifact" eigenvectors, projecting the data back to sensor space.

### 🧮 **Math Toy Example**
* **Setup:** 3 Channels (`C1`, `C2`, `C3`), 4-sample window.
* **Clean Baseline:** PCA yields clean variance limits (eigenvalues) `λ = [2.0, 1.0, 0.5]`. With a `cutoff` of 3, max allowed variance ≈ `[18, 9, 4.5]`.
* **Artifact Window:** A massive blink hits `C3`.

```text
      [  1   1   1   1  ]   <- C1
  X = [  2   2   2   2  ]   <- C2
      [ 50  50  50  50  ]   <- C3 (Huge Artifact)
```

* **ASR Action:** ASR computes the covariance of `X`. The first principal component (aligned with `C3`) has a variance of **2500**. Because `2500 > 18`, ASR flags this spatial direction as an artifact. It constructs a projection matrix that zeroes out the `C3` direction. The value `50` is replaced by a value consistent with the clean correlation of `C1` and `C2` (e.g., `2.1`).

### ⚙️ **Key Parameters & Tuning (`xeegkit` defaults)**
| Parameter | Default | Description & Tuning Guide |
| :--- | :--- | :--- |
| `asr_cutoff` | `3.0` | Standard deviation cutoff for rejection. **2.5** is aggressive (may remove high-amplitude ERPs). **5.0** is conservative (may let large artifacts slip through). `3.0` is the sweet spot for most HD-EEG. |
| `estimator` | `'oas'` | Covariance estimator. **OAS** is used because it provides a robust, regularized estimate even when the number of channels approaches the number of samples, preventing singular matrices. |
| `win_len` | `0.5` | Window length in seconds. Should be roughly the duration of the artifacts you want to catch (e.g., a blink is ~0.3-0.5s). |

<br>

<br>

## 2. Sparse Time-Artifact Removal (STAR)

### **Core Intuition**
STAR operates **sample-by-sample** and **channel-by-channel**. It relies on spatial redundancy: due to volume conduction, the signal at any channel can be accurately predicted by a linear combination of *all other channels*. If a channel's actual value deviates massively from this prediction at a specific time point, that exact sample is flagged and replaced.

### **Mathematical Mechanics**
1. **Phase 1 (Iterative Clean Covariance):** STAR iteratively estimates a clean covariance matrix `C0`. It calculates the "eccentricity" (prediction error) of each channel. Time points with high eccentricity are masked out (`w=0`), and `C0` is recomputed using *only* the clean data. This prevents the initial covariance from being skewed by the very spikes it is trying to find.
2. **Phase 2 (Sparse Interpolation):** For each channel `k`, STAR uses `C0` to compute a linear projection of `k` onto all other channels. It first applies PCA to the neighboring channels to orthogonalize them and discard weak, noisy dimensions.
3. **Eccentricity Calculation:** The residual error is calculated and normalized by the robust standard deviation of the residuals during clean periods:
   `Eccentricity_k(t) = |X_k(t) - X_k_pred(t)| / robust_std(residuals)`
4. **Replacement:** If the eccentricity exceeds the `thresh` parameter, the specific time sample is marked as bad (`w=0`). STAR replaces the corrupted sample `X_k(t)` entirely with the predicted value `X_k_pred(t)`.

### 🧮 **Math Toy Example**
* **Setup:** 3 Channels. Clean model learns that `C3 ≈ 0.5 * C1 + 0.2 * C2`.
* **Artifact:** At time `t=5`, an electrode pop hits *only* `C3`.
  `Data at t=5:  C1=2,  C2=4,  C3=100`
* **STAR Action:** STAR predicts what `C3` should be: `C3_pred = 0.5(2) + 0.2(4) = 1.8`.
* **Decision:** The residual error is `|100 - 1.8| = 98.2`. This vastly exceeds the threshold (e.g., 2.5). STAR surgically replaces the `100` with `1.8`, leaving `C1` and `C2` completely untouched.

### ⚙️ **Key Parameters & Tuning (`xeegkit` defaults)**
| Parameter | Default | Description & Tuning Guide |
| :--- | :--- | :--- |
| `star_thresh` | `2.5` | Threshold for the eccentricity measure (in robust standard deviations). Lower values (e.g., `1.5`) will catch more subtle muscle twitches but risk over-smoothing genuine high-frequency neural spikes (like gamma). |
| `n_iter` | `3` | Number of iterations to refine the clean covariance matrix `C0`. More iterations yield a more robust baseline but increase computation time. |
| `depth` | `1` | Maximum number of channels to fix at each sample. Keeping this at `1` ensures STAR only fixes the *most* eccentric channel at any given millisecond, preventing cascading over-correction. |

<br>

<br>

## 3. Sensor Noise Suppression (SNS)

### **Core Intuition**
Unlike spherical spline interpolation (which uses *physical distance* to fix dead channels), SNS is a **continuous spatial filter** applied to *all* channels to suppress independent thermal noise. It assumes genuine brain signals are spatially correlated, while sensor thermal noise is spatially independent (uncorrelated).

### **Mathematical Mechanics**
1. **Covariance Computation:** SNS computes the full spatial covariance matrix `C` of the cleaned data.
2. **Statistical Neighbor Selection:** For each channel `k`, SNS calculates its correlation with all other channels. It sorts these correlations and selects the top `n_neighbors` (e.g., 8) *most statistically correlated* channels, regardless of physical proximity.
3. **Orthogonalization & Projection:** SNS performs PCA on these neighbors to orthogonalize them. It then computes a regression matrix `R` that projects channel `k` onto this correlated subspace. 
4. **Noise Suppression:** Because the regression captures the shared variance (the brain signal) but ignores the independent variance (the thermal noise), the output is a continuously denoised version of the channel. The algorithm explicitly forces the diagonal of `R` to zero, ensuring the channel cannot simply "predict" itself.

### 🧮 **Math Toy Example**
* **Setup:** True underlying brain signal is a steady `5 µV` on both `C1` and `C2`.
* **Sensor Noise:** `C1`'s amplifier adds independent noise `+2 µV`. `C2`'s amplifier adds `-1 µV`.
* **Measured Data:** `C1 = 7 µV`, `C2 = 4 µV`.
* **SNS Action:** To clean `C1`, SNS regresses it onto `C2`. The regression mathematically extracts the shared component (`5 µV`) and discards the independent component (`+2 µV`). The output for `C1` is pulled away from `7` and pushed back toward the true signal of `5`.

### ⚙️ **Key Parameters & Tuning (`xeegkit` defaults)**
| Parameter | Default | Description & Tuning Guide |
| :--- | :--- | :--- |
| `sns_neighbors` | `8` | Number of correlated neighbors to use for projection. If too high (e.g., >15), SNS acts like a heavy spatial low-pass filter, blurring distinct neural topographies. If too low (e.g., 2), it fails to adequately suppress noise. `8` is optimal for ~64+ channel setups. |
| **Failure Mode** | N/A | If ASR and STAR over-clean the data, the data becomes "low-rank" (lacking independent variance). SNS will throw a `RuntimeError: SNS operator should be zero along diagonal`. `xeegkit` gracefully catches this and skips SNS to prevent pipeline crashes. |

<br>

<br>

## **How `xeegkit` Orchestrates This (`execute_meegkit`)**

The `execute_meegkit` function wraps these algorithms in a robust, production-ready pipeline specifically tuned for high-density EEG:

1. **Preprocessing:** Applies a 1–100 Hz bandpass filter and a 60/120 Hz notch filter. Drops the `Cz` reference channel to prevent it from dominating the average reference.
2. **Bad Channel Detection:** Uses Median Absolute Deviation (MAD) on channel variance and peak-to-peak amplitude (`mad_threshold=10.0`, `min_amplitude_uv=0.1`) to identify flat or excessively noisy channels, excluding them from the cleaning process.
3. **Common Average Reference (CAR):** Re-references *only the good channels* to their average. (Referencing before cleaning can spread artifacts; referencing only good channels after detection is optimal).
4. **ASR Calibration:** Automatically calls `find_cleanest_segment`, which slides a 30-second window across the data, scoring each window by the MAD z-score of its variance and amplitude. It selects the absolute cleanest segment to fit the ASR model, preventing artifact-contaminated baselines.
5. **The Cleaning Chain:** Executes `ASR.transform` → `star` → `sns` sequentially on the good channels.
6. **Reconstruction:** Injects the cleaned good-channel data back into the original `Raw` object. Finally, it uses MNE's spherical spline interpolation to reconstruct the originally detected bad channels, leveraging the now-pristine spatial topology of the cleaned good channels.


