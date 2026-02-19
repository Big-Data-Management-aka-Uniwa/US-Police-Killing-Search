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
  Supervisor: Periklis Andritsos, Associate Professor
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

# INSTALL

## Analysis of Unemployment and Police Killings in United States

This guide explains how to set up, build, and run the project on your local machine.

---

## 1. Prerequisites

This project requires **Python 3** to be installed on your system.

Verify your Python installation by running:

```bash
python --version
```

or

```bash
python3 --version
```

If Python is not installed, download it from:

https://www.python.org/downloads/

Additionally, install the required Python libraries:

```bash
pip install pandas numpy scikit-learn scipy matplotlib
```

---

## 2. Installation

Clone the repository to your local machine:

```bash
git clone https://github.com/Big-Data-Management-aka-Uniwa/US-Police-Killing-Search.git
```

Navigate to the project directory:

```bash
cd US-Police-Killing-Search/src/python
```

Ensure the following folder structure exists:

```
datasets/
processed_datasets/
```

---

## 3. Data Preprocessing

Before running the clustering experiments, preprocess the raw datasets:

```bash
python preprocessData.py
```

This step:

- Filters police killing and poverty data for 2015–2016
- Normalizes U.S. state names to two-letter abbreviations
- Computes population statistics
- Produces cleaned datasets in processed_datasets/
- Generates average datasets for 2015–2016

---

## 3. Run Clustering

### 3.1 Cluster with 2 Datasets

Execute clustering using Police Killings & Poverty datasets:

```bash
python Clustering_2_Datasets.py
```

You will be prompted to select:

- Year (2015 / 2016 / Average)
- Data representation (rates, absolute numbers, logarithmic normalization)
- Number of clusters (k)

The program outputs:

- Cluster visualizations
- SSE (Sum of Squared Errors)
- Silhouette Coefficient
- Elbow Method plot for optimal k

### 3.2 Cluster with 3 Datasets

Execute clustering with Police Killings, Poverty, and Population normalization:

```bash
python Clustering_3_Datasets.py
```

This version:

- Normalizes killings and poverty by total state population
- Performs K-Means clustering
- Visualizes clusters and centroids
- Reports SSE and Silhouette metrics

---

## 4. Jupyter Notebook Support

All source files are also implemented as Jupyter Notebooks (.ipynb), allowing:

- Interactive execution
- Step-by-step analysis
- Inline visualizations

Launch Jupyter Notebook with:

```bash
jupyter notebook
```

Then open the corresponding .ipynb files by navigating to project directory

Navigate to the project directory:

```bash
cd US-Police-Killing-Search/src/jupyter
```

---

## 5. Output

- Cluster scatter plots with centroids
- Quantitative clustering metrics (SSE, Silhouette Coefficient)
- CSV files containing processed and merged datasets

The analysis terminates after all clusters and evaluation metrics are displayed.

---

## 6. Open the Documentation

1. Navigate to the `docs/` directory
2. Open the report corresponding to your preferred language:
   - English: `Police-Killings-US.pdf`
   - Greek: `Δολοφονίες-Αστυνομικών-ΗΠΑ.pdf`
