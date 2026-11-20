# Supplementary Material — Dynamic EEG Functional-Connectivity Biomarkers of Cognitive Load

**Associated manuscript:** *Dynamic EEG functional-connectivity biomarkers of cognitive load: evaluation in two independent cohorts*

This repository accompanies the manuscript and provides additional figures, descriptive tables, participant-level summaries, and sensitivity analyses. The purpose of this README is to make each supplementary item understandable even when it is opened separately from the article. Figure and table identifiers (S1–S18 and S1–S6) are consistent with those cited in the manuscript and reviewer-response material.

## Study at a glance

The analysis examines EEG-derived functional connectivity during the **encoding phase** of the Corsi Block-Tapping Test in two independently analyzed cohorts: **60 older adults** and **21 young adults**. Recordings were segmented into **1-s stimulus-aligned epochs**, and connectivity was estimated using pairwise orthogonalized **amplitude envelope correlation (AEC)** in five bands: theta (4–8 Hz), alpha (8–13 Hz), low beta (13–20 Hz), high beta (20–30 Hz), and gamma (30–60 Hz).

Each item at serial position **k** is assigned a *relative cognitive-load (CL) class* according to the participant's estimated Corsi span **S**, using `class = ceil(4k/S)` for `1 ≤ k ≤ S`. Classes C1–C4 represent successive quarters of the participant's estimated capacity, **not independently measured cognitive-load levels or confirmed states of overload**. Items with `k > S` are unclassified. Absolute sequence length is denoted by **L**.

The revised **main functional-connectivity analyses use clean epochs from successfully completed trials only**. Some supplementary descriptive tables deliberately include epochs from **both successful and unsuccessful trials** to document the original trial-outcome composition; these should not be mistaken for the sample used in the revised main connectivity contrasts. The supplementary sequence-length, bootstrap, and ICA analyses answer separate methodological questions, described below.

### How to read the statistical figures

- **F / P**: frontal / parietal EEG regions.
- **C1–C4**: relative CL classes, ordered from lower to higher relative position within estimated Corsi span.
- **Participant-level analysis**: graph metrics are calculated at the epoch level and summarized within participant, class, frequency band, and region; inferential tests compare paired participants, not individual epochs as independent observations.
- **Significance**: pairwise comparisons use the Wilcoxon signed-rank test and Benjamini–Hochberg false-discovery-rate (FDR) correction. The revised main analysis defines one family of 300 comparisons per dataset (5 graph metrics × 5 bands × 2 regions × 6 class contrasts), corrected separately for each dataset.
- **Effect size**: Wilcoxon `r = |z| / sqrt(n)`, where `n` is the number of valid participant pairs. Values above 0.3 and 0.5 are conventionally described as medium and large, respectively.
- **Graph metrics**: **C_G** = Shannon graph complexity (entropy and dispersion of connectivity weights); **k_i** = node strength; **CPL_i** = characteristic path length; **CC_i** = clustering coefficient; **Q** = weighted modularity. **C_G is not, by itself, a measure of graph topology**. Weighted Q supplies complementary information about community organization, but is also sensitive to connection strengths.

## Supplementary figures: S1–S8 — Additional graph metrics

These figures complement the main-text Shannon graph complexity results. Each figure compares **four relative CL classes across the five frequency bands** in **frontal and parietal regions**. The plots display participant-level class distributions, corrected pairwise statistical comparisons, and Wilcoxon effect sizes. The connectivity results reflect the **successful-trial-only** inclusion criterion adopted in the revised manuscript.

| ID | Suggested stand-alone caption / content | Cohort |
| --- | --- | --- |
| **Fig. S1** | **Node strength across frequency bands and relative cognitive-load classes.** Participant-level distributions of `k_i` in frontal and parietal regions, with corrected within-participant class comparisons and effect sizes. | Older adults |
| **Fig. S2** | **Node strength across frequency bands and relative cognitive-load classes.** Same analysis and display conventions as Fig. S1. | Young adults |
| **Fig. S3** | **Characteristic path length across frequency bands and relative cognitive-load classes.** Participant-level distributions of `CPL_i`, with corrected class comparisons and effect sizes in frontal and parietal regions. | Older adults |
| **Fig. S4** | **Characteristic path length across frequency bands and relative cognitive-load classes.** Same analysis and display conventions as Fig. S3. | Young adults |
| **Fig. S5** | **Clustering coefficient across frequency bands and relative cognitive-load classes.** Participant-level distributions of `CC_i`, with corrected class comparisons and effect sizes in frontal and parietal regions. | Older adults |
| **Fig. S6** | **Clustering coefficient across frequency bands and relative cognitive-load classes.** Same analysis and display conventions as Fig. S5. | Young adults |
| **Fig. S7** | **Modularity across frequency bands and relative cognitive-load classes.** Weighted `Q` distributions for frontal and parietal regional networks, accompanied by FDR-corrected paired Wilcoxon comparisons and effect-size matrices. | Older adults |
| **Fig. S8** | **Modularity across frequency bands and relative cognitive-load classes.** Same analysis and display conventions as Fig. S7. | Young adults |

**Interpretation of figures S1–S8:** These are complementary descriptions of the functional networks, not independent evidence that a particular physiological mechanism caused the observed changes. The two cohorts were acquired under different experimental conditions and should not be treated as a direct age-group comparison.

## `Epoch counts/` — Distribution of clean epochs

**Purpose:** Document how data availability varies with relative CL class, absolute sequence length, item position, and trial outcome. These descriptive counts include eligible *clean* epochs from both successful and unsuccessful trials, and include unclassified (`k > S`) epochs where relevant. This section is especially useful for understanding class imbalance and the smaller number of observations at high relative CL.

### `Old-adults-dataset/`

**Table S1 — Distribution of clean epochs across relative classes and trial outcomes in the older-adult dataset.** Number of clean epochs in C1–C4 and the unclassified category, divided between correctly and incorrectly completed trials. The *correct-trial epoch percentage* is the fraction of clean epochs in each class originating from successful trials; it is **not** a trial-level success rate. For example, these proportions were **70.78% in C1** and **24.97% in C4**.

**File:** `Table S1 – Clean epoch distribution by cognitive load class and trial outcome in the older-adult dataset.csv`

**Table S2 — Distribution of clean epochs by absolute sequence length and relative class in the older-adult dataset.** Cross-tabulation of sequence length `L` against C1–C4, showing how each sequence length contributes clean epochs to the relative classes.

**File:** `Table S2 – Clean epoch distribution by sequence length and cognitive load class in the older-adult dataset.csv`

**Fig. S9 — Joint distribution of sequence length, serial position, and relative class in the older-adult dataset.** Heatmaps of clean-epoch counts for each combination of `L` and `k`, displayed separately by relative class. These make visible how the same absolute serial position can belong to different relative classes across participants with different spans.

**File:** `Fig. S9 – Clean epoch distribution by sequence length, serial position, and cognitive load class in the older-adult dataset.png`

**Detailed participant-level counts.** Workbook breaking down clean-epoch availability for individual participants by Corsi span `S`, sequence length `L`, serial position `k`, relative CL class, and trial outcome. Use it when aggregate tables obscure substantial differences between participants.

**File:** `Detailed clean epoch counts – Older-adult dataset.xlsx`

### `Young-adults-dataset/`

**Table S3 — Distribution of clean epochs across relative classes and trial outcomes in the young-adult dataset.** Equivalent to Table S1. The percentage of clean epochs from successful trials was **84.77% in C1** and **68.72% in C4**; these are *epoch-composition percentages*, not trial-level success rates.

**File:** `Table S3 – Clean epoch distribution by cognitive load class and trial outcome in the young-adult dataset.csv`

**Table S4 — Distribution of clean epochs by absolute sequence length and relative class in the young-adult dataset.** Equivalent to Table S2; the table documents the contribution of each available sequence length to C1–C4.

**File:** `Table S4 – Clean epoch distribution by sequence length and cognitive load class in the young-adult dataset.csv`

**Fig. S10 — Joint distribution of sequence length, serial position, and relative class in the young-adult dataset.** Equivalent to Fig. S9, showing clean-epoch counts for combinations of `L`, `k`, and relative CL class.

**File:** `Fig. S10 – Clean epoch distribution by sequence length, serial position, and cognitive load class in the young-adult dataset.png`

**Detailed participant-level counts.** Participant-level workbook analogous to the older-adult file, including `S`, `L`, `k`, relative class, and trial outcome.

**File:** `Detailed clean epoch counts – Young-adult dataset.xlsx`

**Important distinction:** Tables S1/S3 address *class × outcome*; Tables S2/S4 address *sequence length × class*; Figs. S9/S10 address *sequence length × item position × class*; the Excel workbooks provide the more granular *participant-level* information. The workbook records should be used for participant-specific queries rather than inferring those values from cohort totals.

## `Sequence-length analysis/` — Figures S11–S15

**Purpose:** Explore whether absolute sequence length contributes systematically to Shannon graph complexity (`C_G`) when comparing observations at the **same serial position `k` and the same relative CL class**. This additional analysis was conducted **in the older-adult dataset** across frontal and parietal regions. It compares available sequence lengths at fixed `k` and class; it is **not** a complete experimental separation of relative CL from serial position, participant capacity, fatigue, or other task characteristics.

**Common caption and reading guide for Figs. S11–S15.** Each figure presents one frequency band. **Rows** correspond to absolute serial position (`k = 1–6`), **columns** to frontal (left) and parietal (right) regions, and the **horizontal axis** to absolute sequence length `L`. Each plotted point is one participant's mean `C_G` from eligible clean epochs. Boxplots summarize participant distributions (median and interquartile range); vertical scales are shared within each row but can differ between rows. Available sequence lengths were tested using paired Wilcoxon signed-rank tests at fixed `k` and class, with a global Benjamini–Hochberg FDR correction **for this sequence-length analysis**. **No comparisons remained significant after FDR correction** (`p_adj ≥ 0.05`). Lack of detected differences is not proof that sequence length or within-sequence fatigue has no effect.

| ID | Band | File |
| --- | --- | --- |
| **Fig. S11** | Theta (4–8 Hz) | `Fig. S11 – Shannon graph complexity across sequence lengths at fixed serial positions in the theta band in the older-adult dataset.png` |
| **Fig. S12** | Alpha (8–13 Hz) | `Fig. S12 – Shannon graph complexity across sequence lengths at fixed serial positions in the alpha band in the older-adult dataset.png` |
| **Fig. S13** | Low beta (13–20 Hz) | `Fig. S13 – Shannon graph complexity across sequence lengths at fixed serial positions in the low-beta band in the older-adult dataset.png` |
| **Fig. S14** | High beta (20–30 Hz) | `Fig. S14 – Shannon graph complexity across sequence lengths at fixed serial positions in the high-beta band in the older-adult dataset.png` |
| **Fig. S15** | Gamma (30–60 Hz) | `Fig. S15 – Shannon graph complexity across sequence lengths at fixed serial positions in the gamma band in the older-adult dataset.png` |

**Additional report:** `Sequence-length effects at fixed serial positions – Statistical analysis.pdf` — accompanying statistical analysis and details of the sequence-length comparisons (where included in the shared repository).

## Fig. S16 — Bootstrap stability of averaged AEC networks

**Fig. S16 — Bootstrap stability of theta-band AEC connectivity in the older-adult dataset.** This analysis measures how consistently the *participant- and class-averaged* connectivity pattern is recovered when the constituent 1-s epoch-level AEC matrices are resampled.

For each older participant and CL class, a reference theta-band AEC matrix was formed by averaging all available epoch-level matrices. **1,000 bootstrap samples**, each containing the same number of epoch matrices resampled with replacement, were averaged and compared with the reference using **Spearman correlation across the 120 unique network connections**. Stability for a participant/class combination was summarized by the median correlation across bootstrap iterations. Reported median correlations exceeded **0.7**, supporting the **resampling stability of averaged connectivity patterns**. This does **not** establish the reliability of individual 1-s AEC matrices or equivalence to connectivity estimated from longer continuous windows.

**File:** `Fig. S16 – Bootstrap stability of theta-band AEC connectivity in the older-adult dataset.png`

## `ICA Sensitivity Analysis/` — Tables S5–S6 and Figures S17–S18

**Purpose:** Assess whether the principal `C_G` findings in the **older-adult dataset** are materially altered by additional independent component analysis (ICA)-based correction for EEG components automatically classified as ocular or muscular artifacts.

**Comparison:** Original preprocessing (automatic amplitude-based epoch rejection at a 4-standard-deviation threshold) versus an alternative pipeline incorporating **EEGLAB ICA and ICLabel** before applying the same epoch-rejection procedure. The ICA analysis covered **1,043 recordings from 60 participants**, with **15 components estimated per recording** to account for rank reduction after common-average referencing. Eye or Muscle components were removed when the corresponding ICLabel probability was **≥ 0.80**. The downstream `C_G` comparison used **clean epochs from successful trials only** in both pipelines.

**Table S5 — Epoch rejection rates before and after ICA correction in the older-adult dataset.** Participant-level percentages of eligible epochs rejected in each CL class under the original and ICA-corrected pipelines, with descriptive summaries. The workbook provides detailed participant-level results; any compact table presented in the reviewer response is a **summary extracted from this workbook, not a duplicate of the full file**. Across classes, mean rejection rates were similar before and after ICA correction.

**File:** `Table S5 – Epoch rejection rates by cognitive load class before and after ICA correction in the older-adult dataset.xlsx`

**Table S6 — ICA-based artifact component characterization in the older-adult dataset.** Record-level characterization of ICLabel-classified ocular and muscular components and components removed using the specified threshold. The analysis identified **238 Eye** and **206 Muscle** components, with **444 components removed across 357 recordings (34.2%)**. These are *classification counts*, **not direct measures of residual EOG/EMG contamination**; independent EOG and EMG recordings were not available. A compact summary in the reviewer response is extracted from the detailed workbook.

**File:** `Table S6 – ICA-based artifact component characterization in the older-adult dataset.xlsx`

**Fig. S17 — Shannon graph complexity distributions before and after ICA correction in the older-adult dataset.** Violin plots compare participant-level `C_G` across C1–C4 in the theta, alpha, low-beta, high-beta, and gamma bands (rows) and frontal/parietal regions (columns). **Solid violins** represent the original pipeline; **hatched violins** the ICA-corrected pipeline. White markers denote medians and dark vertical marks the interquartile ranges. Distributions are based on **60 participants** and **successful-trial epochs only**.

**File:** `Fig. S17 – Shannon graph complexity distributions before and after ICA correction in the older-adult dataset.png`

**Fig. S18 — Comparison of Shannon graph complexity effect sizes before and after ICA correction in the older-adult dataset.** Paired Wilcoxon `r` values for relative-CL class contrasts under the two pipelines, separately for frontal and parietal regions across five frequency bands. **Solid bars** indicate original preprocessing and **hatched bars** indicate ICA correction; labels above bars show effect-size values. The 60 unique pairwise class contrasts (6 contrasts × 5 bands × 2 regions) remained statistically significant after FDR correction in both pipelines. This supports sensitivity-analysis consistency but does **not** prove that all artifacts were removed.

**File:** `Fig. S18 – Comparison of Shannon graph complexity effect sizes before and after ICA correction in the older-adult dataset.png`

**Detailed statistical results workbook.** Full paired class comparisons of `C_G` across bands and frontal/parietal regions, including results from both the original and ICA-corrected pipelines; use this spreadsheet for detailed test-by-test values rather than extracting estimates from figure graphics.

**File:** `Detailed statistical results – Shannon graph complexity before and after ICA correction in the older-adult dataset.xlsx`

## Code and data availability

**Analysis code.** The analysis scripts associated with this study are provided in the [`Code`](Code/) folder of this repository. These scripts accompany the supplementary figures, tables, and statistical results to support methodological transparency and reuse. Some analyses require the original EEG recordings, which are not included in this public repository.

**EEG data.** Raw and processed EEG recordings are not publicly available because their collection and use are subject to ethical and data-governance restrictions. Researchers interested in accessing the datasets may contact the corresponding author with a justified research request. Requests will be referred to the responsible ethics committee for evaluation; any data sharing will be conditional on the necessary ethical authorization and applicable data-protection requirements. **Access cannot be guaranteed.**

**Supplementary results.** The aggregate results and participant-level summary files described above are publicly available in this repository. These materials do not include the underlying raw or processed EEG recordings.

## Scope, limitations, and reuse

1. **Independent cohorts, not an age comparison.** The older- and young-adult datasets differ in acquisition configuration, montage, original reference, and task rules. Results should be interpreted as independent evaluations, not direct comparisons of age effects or strict replication.
2. **Relative CL is a task-derived proxy.** C1–C4 encode progression through a sequence relative to estimated capacity; serial position, sequence length, and performance are not fully disentangled by labeling alone.
3. **Short-window connectivity.** One-second epochs preserve the task's item-level timing, but are short for AEC estimation, particularly in theta. The bootstrap evaluates the stability of *averaged matrices under resampling*, not absolute accuracy or longer-window equivalence.
4. **Artifact sensitivity.** ICA correction is a sensitivity analysis, not a substitute for independent EOG/EMG recordings or a guarantee of artifact-free data.
5. **Reuse and restricted data access.** The public scripts and supplementary summaries facilitate inspection and reuse of the analyses, but reproducing analyses that depend on individual EEG signals requires access to restricted data. Such access is subject to ethics-committee review and authorization, as described above.

### Recommended citation and provenance

Please cite the **associated manuscript** when using the analyses. When referring to a particular result, also cite its supplementary identifier (for example, *Fig. S16* or *Table S5*) and the relevant file name. A publication or archive DOI should be added here **only after one is assigned**. The presence of scripts in `Code/` does not imply that the restricted EEG datasets or a DOI-backed archive are publicly available.

---

*This README is an index and explanatory guide. The actual figures, CSV tables, workbooks, and any accompanying analysis report remain the authoritative sources for numerical results. Check the `Code/` directory, file names, and folder placement against the final repository before publication.*
