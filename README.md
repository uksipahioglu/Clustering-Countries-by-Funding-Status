# 🌍 Unsupervised Clustering on Country Development Indicators

![Python](https://img.shields.io/badge/Python-3.x-blue?style=flat&logo=python)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Unsupervised-green)
![Visualization](https://img.shields.io/badge/Visualization-Plotly%20%7C%20Seaborn-orange)

## 📖 Project Overview
This project applies multiple unsupervised learning algorithms to the **Country Development Dataset** to identify patterns related to economic and social development. By analyzing macro-economic indicators, the project aims to categorize countries based on their funding needs.

The workflow includes extensive **EDA (Exploratory Data Analysis)**, **preprocessing**, **dimensionality reduction (PCA)**, **clustering**, and **interactive global mapping**.

## 📌 Project Objectives
* [x] Explore the dataset with histograms and correlation analysis.
* [x] Normalize numeric features using `MinMaxScaler`.
* [x] Apply **PCA** for dimensionality reduction.
* [x] Perform clustering using four different algorithms.
* [x] Compare all models using **Silhouette Scores**.
* [x] Visualize cluster outputs on a world choropleth map.
* [x] Categorize countries into:
    * **Budget Needed**
    * **In Between**
    * **No Budget Needed**

## 📁 Dataset
The dataset includes key macro-economic indicators. Each country is represented by a single row of numeric features:

* **Child Mortality:** Death of children under 5 years of age per 1000 live births.
* **Exports / Imports:** Exports and imports of goods and services per capita.
* **Health Spending:** Total health spending as % of GDP.
* **Income:** Net income per person.
* **Inflation:** The measurement of the annual growth rate of the Total GDP.
* **Life Expectancy:** The average number of years a new born child would live.
* **Fertility Rate:** The number of children that would be born to each woman.
* **GDP per Capita:** The GDP divided by the total population.

## 🔧 Technologies Used
* **Language:** Python
* **Data Manipulation:** Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn, Plotly
* **Machine Learning:** Scikit-Learn
* **Advanced Clustering:** HDBSCAN

## 📉 Dimensionality Reduction (PCA)
**Principal Component Analysis (PCA)** is used to capture the maximum variance in the data while reducing noise.
* Only the **first 3 principal components** are used for clustering to optimize performance.
* Variance coverage is visualized using cumulative explained variance plots.

## 🤖 Clustering Models
The following algorithms were trained and compared to find the best fit:

| Model | Notes |
| **KMeans** | Baseline model, determined 3 optimal clusters. |
| **Agglomerative** | Hierarchical clustering approach. |
| **DBSCAN** | Density-based model capable of detecting noise/outliers. |
| **HDBSCAN** | Improved DBSCAN that auto-determines optimal clusters and density. |

## 🗺️ Global Choropleth Maps
Every clustering model is visualized on an interactive world map. The colors represent the calculated status of each country:

* 🟥 **Budget Needed** (High urgency)
* 🟨 **In Between** (Developing)
* 🟩 **No Budget Needed** (Developed)
* ⬛ **Noise** (Outliers detected by DBSCAN/HDBSCAN)
* ⚪ **No Data**

These maps provide intuitive, visual insight into the global economic conditions.

## 🏆 Silhouette Score Comparison
All models are scored using **Silhouette metrics**, and results are visualized in a comparison bar chart.

![Silhouette Score Comparison](silhouette_comparison.png)

* **Higher score** → Better cluster separation.
* This metric helps identify the most meaningful algorithm for this specific dataset.

## ▶️ How to Run This Project

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/uksipahioglu/Clustering-Countries-by-Funding-Status.git](https://github.com/uksipahioglu/Clustering-Countries-by-Funding-Status.git)
    ```

2.  **Install required dependencies:**
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn plotly hdbscan
    ```

3.  **Run the analysis:**
    ```bash
    python Clustering_Countries_by_Funding_Status.py
    ```

📬 Contact

Author: Umut Kıvanç Sipahioğlu
Kaggle: https://www.kaggle.com/umutkvansipahiolu
LinkedIn: https://www.linkedin.com/in/umut-kıvanc-sipahioğlu-410251237/
















