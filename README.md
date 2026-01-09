# Dynamic-and-robust-EEG-functional-connectivity-biomarkers-of-cognitive-load-across-age-groups
Repository containing supplementary figures and detailed descriptions supporting the study on dynamic EEG functional connectivity biomarkers of cognitive load across age groups. Includes graph-theoretical analyses, frequency-band results, and frontal and parietal network metrics for older and young adults.

## Overview 
The repository includes **six supplementary figures (Figs. S1–S6)** that report additional graph-theoretical analyses of EEG functional connectivity under increasing cognitive load (CL). These figures complement the main manuscript by presenting results for metrics not included in the main text, namely:

- **Node strength**
- **Characteristic path length**
- **Clustering coefficient**

All analyses follow the same dynamic, capacity-normalized framework described in the article and are reported separately for **older-adult** and **young-adult datasets**.

### Figure organization and interpretation 
Each supplementary figure follows a common structure to facilitate comparison across metrics, frequency bands, regions, and cognitive load classes.

#### Common layout (aplies to all figures) 
**Rows** correspond to frequency bands:

- A: Theta (4–8 Hz)
- B: Alpha (8–13 Hz)
- C: Low beta (13–20 Hz)
- D: High beta (20–30 Hz)
- E: Gamma (>30 Hz)

**Left panel**: Violin plots with overlaid boxplots showing the distribution of the metric across cognitive load classes.

- Left plot: frontal region
- Right plot: parietal region

**Central panel**: Matrices of **FDR-corrected p-values** from pairwise Wilcoxon signed-rank tests between cognitive load classes.

- Non-white cells indicate statistically significant differences (p < 0.05).
- Lower-left triangle: frontal region
- Upper-right triangle: parietal region

**Right panel**: Corresponding **effect size matrices (r-values)**, where darker shades indicate larger effects.

#### List of supplementary figures 

- **Fig. S1** – Node strength across frequency bands and cognitive load classes in the **older-adult dataset**.
- **Fig. S2** – Node strength across frequency bands and cognitive load classes in the **young-adult dataset**.
- **Fig. S3** – Characteristic path length across frequency bands and cognitive load classes in the **older-adult dataset**.
- **Fig. S4** – Characteristic path length across frequency bands and cognitive load classes in the **young-adult dataset**.
- **Fig. S5** – Clustering coefficient across frequency bands and cognitive load classes in the **older-adult dataset**.
- **Fig. S6** – Clustering coefficient across frequency bands and cognitive load classes in the **young-adult dataset**.
