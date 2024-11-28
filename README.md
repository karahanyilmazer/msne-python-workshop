# MSNE Python Workshop: EEG Analysis

This repository accompanies the **MSNE Python Workshop** on offline EEG analysis, focusing on motor imagery data from the BCICIV dataset. The aim is to explore preprocessing, visualization, and classification techniques using the MNE library.

## Objectives

1. **Understand EEG Data**: Learn how motor imagery affects brain waves (mu rhythm) and observe Event-Related Desynchronization (ERD) and Synchronization (ERS).
2. **Preprocessing**: Apply band-pass filters, visualize Power Spectral Density (PSD), and use ICA to remove artifacts.
3. **Feature Extraction**: Use Common Spatial Patterns (CSP) to enhance class separability.
4. **Classification**: Train a classifier (LDA) to distinguish motor imagery tasks.

## What You’ll Do

- **Load and Explore EEG Data**: Analyze EEG recordings from one participant.
- **Visualize ERD/ERS**: Calculate and plot ERD/ERS curves to observe motor imagery effects.
- **Spatial Filtering**: Compare results with and without spatial filtering to understand its impact.
- **Train a Classifier**: Use CSP and LDA to achieve accurate classification of left and right hand motor imagery.

## Instructions

1. Clone the repository by running:

`git clone git@github.com:karahanyilmazer/msne-python-workshop.git`

2. Follow the [setup tutorial](MSNE_Python_Workshop-EEG_Analysis.pdf) to set up your Python environment and install required libraries.

3. Read the [EEG motor imagery tutorial](MSNE_Python_Workshop-EEG_Analysis.pdf) 
for background information on EEG, motor imagery, CSP and ERD/ERS curves.

4. Open the `berlin_mi_analysis-problem.py` script:
   - Follow guided steps to preprocess, visualize, and classify EEG data.
   - Complete the `calc_erds()` and `plot_erds()` functions.

5. Refer to `berlin_mi_analysis-solution.py` for the complete implementation.

## Insights from Results

- **ERD/ERS Analysis**: Spatial filtering highlights key features in the data, making ERD/ERS patterns clearer.
- **Classification Performance**: CSP and LDA yield high accuracy in distinguishing left and right hand motor imagery.

## Key Takeaway

Through this workshop, you’ll see how spatial filtering and preprocessing transform raw EEG data into meaningful insights, paving the way for high-performing classifiers.

Happy coding!

## Repository Structure

```
msne-python-workshop
├─ .gitignore
├─ MSNE_Python_Workshop-EEG_Analysis.pdf
├─ README.md
├─ berlin_mi_analysis-problem.py
├─ berlin_mi_analysis-solution.py
├─ data
│  └─ BCICIV_calib_ds1e.mat
├─ img
│  ├─ erds_base.png
│  └─ erds_car.png
└─ utils.py

```