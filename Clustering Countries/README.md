# 🌍 Unsupervised Clustering on Country Development Indicators

PCA • KMeans • Agglomerative • DBSCAN • HDBSCAN • Choropleth Maps

This project applies multiple unsupervised learning algorithms to the Country Development Dataset to identify patterns related to economic and social development.
The analysis includes preprocessing, visualization, dimensionality reduction, clustering, and global mapping.

## 📌 Project Objectives

**Explore the dataset with histograms and correlation analysis**
* **Normalize numeric features using MinMaxScaler**
* **Apply PCA for dimensionality reduction**
* **Perform clustering with:**
    *  **KMeans**
    *  **Agglomerative Clustering**
    *  **DBSCAN**
    *  **HDBSCAN**
* **Compare all models using Silhouette Scores**
* **Visualize cluster outputs on a world choropleth map**
* **Identify groups:**
    *  **Budget Needed**
    *  **In Between**
    *  **No Budget Needed**
    *  **Noise / No Data**

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
|---|---|
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
### 1. K-Means Clustering

![K-Means Plot](https://github.com/uksipahioglu/Clustering-Countries-by-Funding-Status/blob/main/Clustering%20Countries/kmeansplot.png)





[![Interactive Map](https://img.shields.io/badge/🌍%20Click%20to%20Open-Interactive%20Map-blue?style=for-the-badge)](https://htmlpreview.github.io/?https://github.com/uksipahioglu/Clustering-Countries-by-Funding-Status/blob/main/Clustering%20Countries/kmeans_map.html)

### 2. Agglomerative Hierarchical Clustering

![Agglomerative Clustering Plot](https://github.com/uksipahioglu/Clustering-Countries-by-Funding-Status/blob/main/Clustering%20Countries/Aggplot.png)




Explore the map interactively:
[![Interactive Map](https://img.shields.io/badge/🌍%20Click%20to%20Open-Interactive%20Map-blue?style=for-the-badge)](https://htmlpreview.github.io/?https://github.com/uksipahioglu/Clustering-Countries-by-Funding-Status/blob/main/Clustering%20Countries/agg_map.html)


### 3. DBSCAN (Density-Based Spatial Clustering)

![DBSCAN Clustering Plot](https://github.com/uksipahioglu/Clustering-Countries-by-Funding-Status/blob/main/Clustering%20Countries/dbscanplot.png)




Explore the map interactively:
[![Interactive Map](https://img.shields.io/badge/🌍%20Click%20to%20Open-Interactive%20Map-blue?style=for-the-badge)](https://htmlpreview.github.io/?https://github.com/uksipahioglu/Clustering-Countries-by-Funding-Status/blob/main/Clustering%20Countries/dbscan_map.html)


### 4. HDBSCAN (Hierarchical DBSCAN)

![HDBSCAN Clustering Plot](https://github.com/uksipahioglu/Clustering-Countries-by-Funding-Status/blob/main/Clustering%20Countries/hdbscanplot.png)




Explore the map interactively:
[![Interactive Map](https://img.shields.io/badge/🌍%20Click%20to%20Open-Interactive%20Map-blue?style=for-the-badge)](https://htmlpreview.github.io/?https://github.com/uksipahioglu/Clustering-Countries-by-Funding-Status/blob/main/Clustering%20Countries/hdbscan_map.html)



## 🏆 Silhouette Score Comparison
All models are scored using **Silhouette metrics**, and results are visualized in a comparison bar chart.

![Silhouette Score Comparison](https://github.com/uksipahioglu/Clustering-Countries-by-Funding-Status/blob/main/Clustering%20Countries/comparisonplot.png)



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

## 📬 Contact

**Author:** Umut Kıvanç Sipahioğlu

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/umut-kıvanc-sipahioğlu-410251237/)
[![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=Kaggle&logoColor=white)](https://www.kaggle.com/umutkvansipahiolu)
















