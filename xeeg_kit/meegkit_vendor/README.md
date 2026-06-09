## **Framework:**
1.  **ASR:** Fix the massive movement/burst artifacts first so they don't corrupt the spatial statistics.
2.  **STAR:** Fix the random, isolated electrode pops using global spatial correlation.
3.  **SNS:** Apply a final continuous spatial filter to suppress the baseline thermal hiss of the sensors across all channels.

## Table Summary

| Feature | **ASR** | **STAR** | **SNS** |
| :--- | :--- | :--- | :--- |
| **Unit of Operation** | **Time Window** (e.g., 500ms chunks) | **Single Time Point** (sample-by-sample) | **Continuous Signal** (Entire timeline) |
| **Channels Used** | **All Channels** (Multivariate PCA) | **All Other Channels** (Linear Prediction) | **Most Correlated Channels** (Statistical neighbors) |
| **Math Core** | Covariance Eigenvalue Thresholding | Linear Regression / Projection | Spatial Projection / Regression |
| **Analogy** | "This 0.5s snapshot has impossible variance; project out the bad spatial directions." | "Channel 3 just spiked; ignore it and calculate what it *should* be based on the rest of the scalp." | "Every channel has a little bit of static; filter it out by averaging the shared brain signal across correlated channels." |
| **Corrects** | Bursts, movements, high-variance transients | Sparse spikes, electrode pops, muscle twitches | Continuous broadband thermal sensor noise |


