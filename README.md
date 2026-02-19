<p align="center">
  <img src="https://www.especial.gr/wp-content/uploads/2019/03/panepisthmio-dut-attikhs.png" alt="UNIWA" width="150"/>
</p>

<p align="center">
  <strong>UNIVERSITY OF WEST ATTICA</strong><br>
  SCHOOL OF ENGINEERING<br>
  DEPARTMENT OF COMPUTER ENGINEERING AND INFORMATICS
</p>

<p align="center">
  <a href="https://www.uniwa.gr" target="_blank">University of West Attica</a> ·
  <a href="https://ice.uniwa.gr" target="_blank">Department of Computer Engineering and Informatics</a>
</p>

<hr/>

<p align="center">
  <strong>Big Data Management</strong>
</p>

<h1 align="center" style="letter-spacing: 1px;">
  Analysis of Unemployment and Police Killings in United States
</h1>

<p align="center">
  <strong>Vasileios Evangelos Athanasiou</strong><br>
  Student ID: 19390005
</p>

<p align="center">
  <a href="https://github.com/Ath21" target="_blank">GitHub</a> ·
  <a href="https://www.linkedin.com/in/vasilis-athanasiou-7036b53a4/" target="_blank">LinkedIn</a>
</p>

<p align="center">
  <strong>Panagiotis Petropoulos</strong><br>
  Student ID: 20390188
</p>

<p align="center">
  <a href="https://github.com/PanosPetrop" target="_blank">GitHub</a> 
</p>

<p align="center">
  <strong>Pantelis Tatsis</strong><br>
  Student ID: 20390226
</p>

<p align="center">
  <a href="https://github.com/PanthegrammerPRO" target="_blank">GitHub</a> ·
  <a href="https://www.linkedin.com/in/pantelis-tatsis-8287852a2/" target="_blank">LinkedIn</a>
</p>

<hr>

<p align="center">
  <strong>Supervision</strong>
</p>

<p align="center">
  Supervisor: Periklis Andritsos, Professor
</p>
<p align="center">
  <a href="https://ice.uniwa.gr/en/emd_person/periklis-andritsos/" target="_blank">UNIWA Profile</a> ·
  <a href="https://www.linkedin.com/in/periklisandritsos/" target="_blank">LinkedIn</a>
</p>

</hr>

---

<p align="center">
  Athens, June 2024
</p>

---

<p align="center">
  <img src="https://i.guim.co.uk/img/static/sys-images/Guardian/Pix/pictures/2015/6/8/1433799599893/a72b9c69-6f14-4ef2-b8e4-f74f41c26188-620x372.png?width=465&dpr=1&s=none&crop=none" width="250"/>
</p>

---

# README

## Analysis of Unemployment and Police Killings in United States

This project investigates the potential relationship between socio-economic factors—specifically unemployment and poverty rates and incidents of police killings across various U.S. states between 2015 and 2016

---

## Table of Contents

| Section | Folder                                | Description                                                            |
| ------: | ------------------------------------- | ---------------------------------------------------------------------- |
|       1 | `assign/`                             | Assignment material for the Business Data & Management course          |
|     1.1 | `assign/Assignment-BDMGMT-Apr24.pdf`  | Assignment description in English                                      |
|     1.2 | `assign/Εργασία-ΔΧΜΚ-Απρ24.pdf`       | Assignment description in Greek                                        |
|       2 | `docs/`                               | Documentation and reports on US police killings                        |
|     2.1 | `docs/Police-Killings-US.pdf`         | English report                                                         |
|     2.2 | `docs/Δολοφονίες-Αστυνομικών-ΗΠΑ.pdf` | Greek report                                                           |
|       3 | `graphs/`                             | Visualizations and charts of datasets                                  |
|     3.1 | `graphs/2015-*.png`                   | Various 2015 charts: elbow method, optimal clusters, percentages, logs |
|     3.2 | `graphs/2016-*.png`                   | Various 2016 charts: elbow method, optimal clusters, percentages, logs |
|     3.3 | `graphs/avg-*.png`                    | Average charts across years                                            |
|     3.4 | `graphs/Clustering3-*.png`            | Charts for 3-dataset clustering experiments                            |
|       4 | `src/`                                | Source code, datasets, and notebooks                                   |
|     4.1 | `src/datasets/`                       | Raw datasets in CSV and JSON formats                                   |
|     4.2 | `src/jupyter/`                        | Jupyter notebooks for analysis and preprocessing                       |
|     4.3 | `src/processed_datasets/`             | Cleaned and processed datasets                                         |
|     4.4 | `src/python/`                         | Python scripts for clustering and preprocessing                        |
|       5 | `README.md`                           | Project documentation                                                  |
|       6 | `INSTALL.md`                          | Usage instructions                                                     |

---

## 1. Project Overview

The research explores whether states with higher levels of unemployment and poverty also exhibit higher frequencies of police brutality. By grouping data into clusters, the study seeks to identify patterns and correlations that can inform social policy and crime understanding.

---

## 2. Key Objectives

- **Correlation Analysis**  
  Investigating whether unemployment rates correlate with homicide incidents.

- **Pattern Detection**  
  Using clustering techniques to observe geographical and economic patterns in police killings.

- **Policy Support**  
  Providing data-driven insights for sociologists, economists, and policymakers to address social issues.

---

## 3. Datasets

The analysis utilizes three primary datasets sourced from platforms such as **Kaggle** and **Opendatasoft**:

- **Police Killings Dataset**  
  Includes data on victims (age, sex, race), location (state/city), cause of death, and whether body cameras were used.

- **Unemployment / Poverty Dataset**  
  Contains poverty rates and absolute numbers of unemployed individuals per state.

- **US City Populations Dataset**  
  Experimental data for cities with more than 65,000 inhabitants, used to normalize results against total state populations.

---

## 4. Methodology

The team employed **Cluster Analysis** using the **K-Means algorithm**. This method was chosen for its efficiency with relatively small datasets (51 entries) and its ability to detect hidden patterns without requiring pre-labeled data.

---

## 5. Data Preprocessing

- **Time Alignment**  
  Data was pruned to include only the years **2015 and 2016** to ensure a common temporal scale.

- **Morphology Normalization**  
  State names were standardized to abbreviations (e.g., _New York → NY_) to enable successful dataset joining.

- **Normalization**  
  Logarithmic normalization and standard deviation scaling were applied to account for significant differences in population size between states.

- **Handling Data Gaps**  
  States with zero recorded murders (e.g., _Rhode Island in 2015_) were manually assigned a value of `0` rather than being omitted.

---

## 6. Evaluation Metrics

To assess cluster quality, the following non-predictive evaluation metrics were used:

- **SSE (Sum of Squared Error)**  
  Measures the deviation of actual values from cluster centroids, indicating how close data points are within clusters.

- **Silhouette Coefficient**  
  Measures how well data points are separated between clusters; values closer to `1` indicate better clustering.

---

## 7. Experimental Results

The analysis was conducted in **four distinct stages**, progressing from percentage-based metrics to log-normalized absolute values to improve accuracy.

| Analysis Stage   | Metric     | Year 2015 | Year 2016 |
| ---------------- | ---------- | --------- | --------- |
| Percentage Rates | SSE        | 25,464    | 29,151    |
|                  | Silhouette | 0.518     | 0.495     |
| Net Numbers      | SSE        | 18,885    | 19,478    |
|                  | Silhouette | 0.481     | 0.484     |
| Log Normalized   | SSE        | 22,391    | —         |
|                  | Silhouette | 0.554     | —         |

---

## 8. Technologies Used

- **Programming Language:** Python 3
- **Data Analysis & Machine Learning:**
  - K-Means Clustering (Unsupervised Learning)
  - Z-score Normalization
  - Logarithmic Normalization
- **Big Data & Statistical Concepts:**
  - Cluster Analysis
  - Socio-economic Data Correlation
  - Population Normalization Factors
- **Data Structures:**
  - Pandas DataFrames
  - NumPy Arrays
- **Libraries & Frameworks:**
  - `pandas` (data loading, preprocessing, merging)
  - `numpy` (numerical operations, transformations)
  - `scikit-learn` (KMeans, Silhouette Coefficient, SSE)
  - `scipy` (`zscore` normalization)
  - `matplotlib` (data visualization)
  - `json` (US states name mapping)
  - `os` (console handling)
- **Development Environment:**
  - Python scripts (`.py`)
  - Jupyter Notebooks (`.ipynb`)
