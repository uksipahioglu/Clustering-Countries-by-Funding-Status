import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
import math
import warnings
warnings.filterwarnings('ignore')
# %%
df = pd.read_csv("Country-data.csv")
# %%
df.info()
# %%
def plot_all_histograms(df,title_prefix=""):
    num_cols = df.select_dtypes(include=[np.number]).columns
    n_cols = 3
    n_rows = math.ceil(len(num_cols)/n_cols)
    
    plt.figure(figsize=(10*n_cols,4*n_rows))
    
    for i,col in enumerate(num_cols,1):
        plt.subplot(n_rows,n_cols,i)
        sns.histplot(df[col],kde=True,bins=30)
        plt.title(f"{title_prefix} {col}")
        
    plt.tight_layout()
    plt.show()
# %%
plot_all_histograms(df)
# %%
sns.heatmap(df.corr(numeric_only=True),annot=True)
plt.show()
# %%
df2 = df.drop("country",axis = 1)
# %%
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df2 = scaler.fit_transform(df2)
df.columns
df2 = pd.DataFrame(df2,columns=['child_mort', 'exports', 'health', 'imports', 'income',
       'inflation', 'life_expec', 'total_fer', 'gdpp'])
# %%
plot_all_histograms(df2)
# %%
from sklearn.decomposition import PCA
pca = PCA()
pca_df2 = pd.DataFrame(pca.fit_transform(df2))
# %%
pca.explained_variance_ratio_
# %%
plt.step(list(range(1,10)),np.cumsum(pca.explained_variance_ratio_))
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.ylabel("Variance Covered")
plt.title("Variance Covered")
plt.show()
# %%
pca_df2 = pca_df2.drop(columns=[3,4,5,6,7,8])
# %%
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
# %%
wcss = []
for k in range(1,11):
    kmeans = KMeans(n_clusters=k,random_state=15)
    kmeans.fit(pca_df2)
    wcss.append(kmeans.inertia_)
# %%
wcss
# %%
plt.plot(range(1,11),wcss)
plt.xticks(range(1,11))
plt.xlabel("number of clusters")
plt.ylabel("WCSS")
plt.show()
# %%
model = KMeans(n_clusters=3,random_state=15)
model.fit(pca_df2)
model.labels_
labels = model.labels_
silhouette_score(pca_df2,labels)
df["Class"] = labels
# %%
fig, ax = plt.subplots(nrows= 1,ncols= 2, figsize=(15,5))
plt.subplot(1,2,1)
sns.boxplot(data=df,x="Class",y="child_mort",order=[0,1,2])
plt.title("child_mort vs class")

plt.subplot(1,2,2)
sns.boxplot(data=df,x="Class",y="income",order=[0,1,2])
plt.title("income vs class")
plt.show()
# %%
# 0 -> in between
# 1 -> buget needed
# 2 -> no budget needed
# %%
import plotly.express as px
# %%
pca_df2.insert(0,column = "Country", value = df["country"])
pca_df2["Class"] = labels
# %%
pca_df2.loc[pca_df2["Class"]== 1, "Class"] = "Budget Needed"
pca_df2.loc[pca_df2["Class"]== 0, "Class"] = "In Between"
pca_df2.loc[pca_df2["Class"]== 2, "Class"] = "No Budget Needed"
# %%
pca_df2
# %%
#Finding missing countries
all_countries_df = px.data.gapminder()
all_countries_names = set(all_countries_df['country'].unique())
data_countries = set(pca_df2["Country"].unique())
missing_countries = all_countries_names - data_countries
# %%
missing_data = []
for country in missing_countries:
    missing_data.append({
        "Country": country,
        "Class": "No Data",
        0: np.nan,          
        1: np.nan,
        2: np.nan
    })
# %%
countries_to_add = {
    "Country": ["Papua New Guinea", "North Korea", "South Sudan"],
    "Class": ["No Data", "No Data", "No Data"],
    0: [np.nan, np.nan, np.nan],
    1: [np.nan, np.nan, np.nan],
    2: [np.nan, np.nan, np.nan]
}
# %%
countries_to_add = pd.DataFrame(countries_to_add)
missing_df = pd.DataFrame(missing_data)
combined_df = pd.concat([pca_df2, missing_df,countries_to_add], ignore_index=True)
combined_df["Country"] = combined_df["Country"].replace({
    "Korea, Dem. Rep." : "South Korea"
    })
# %%
import plotly.io as pio
pio.renderers.default = "browser"
# %%
fig=px.choropleth(
    combined_df,
    locationmode = "country names",
    locations = "Country",
    title = "Funding Status by Country (KMeans)",
    color = "Class",
    color_discrete_map = {
        
        "Budget Needed" : "Red" ,
        "No Budget Needed" : "Green",
        "In Between" : "Yellow",
        "No Data": "#d3d3d3"
        },
   labels={'Class': 'Funding_Status'} 
   )
fig.update_layout(
        geo=dict(
            scope='world',
            projection=dict(type='natural earth') 
        )
    ) 
fig.show()
# %%

from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score
import hdbscan
# %%
missing_df_plot = missing_df[['Country', 'Class']]
countries_to_add_plot = countries_to_add[['Country', 'Class']]
global_color_map = {
    "Budget Needed" : "Red" ,
    "No Budget Needed" : "Green",
    "In Between" : "Yellow",
    "No Data": "#d3d3d3",
    "Noise": "#555555" 
    }
# %%
clean_pca_df = pd.DataFrame(pca.fit_transform(df2)).drop(columns=[3,4,5,6,7,8])
# %%
kmeans_score = silhouette_score(clean_pca_df, labels) 
silhouette_scores = {'KMeans': kmeans_score}
print(f"KMeans Silhouette Score: {kmeans_score:.4f}")
# %%
## Agglomerative Clustering
agg_model = AgglomerativeClustering(n_clusters=3)
agg_labels = agg_model.fit_predict(clean_pca_df)
agg_score = silhouette_score(clean_pca_df, agg_labels)
silhouette_scores['Agglomerative'] = agg_score
print(f"Agglomerative Silhouette Score: {agg_score:.4f}")
# %%
##Map
df_agg = df[['country', 'income']].copy() 
df_agg['Class_Label'] = agg_labels
agg_means = df_agg.groupby('Class_Label')['income'].mean()
agg_budget_needed_label = agg_means.idxmin()
agg_no_budget_label = agg_means.idxmax()
agg_in_between_label = [l for l in agg_means.index if l not in [agg_budget_needed_label, agg_no_budget_label]][0]
# %%
agg_map = {
    agg_budget_needed_label: "Budget Needed",
    agg_no_budget_label: "No Budget Needed",
    agg_in_between_label: "In Between"
}
# %%
df_agg['Class'] = df_agg['Class_Label'].map(agg_map)
df_agg = df_agg.rename(columns={'country': 'Country'})
combined_df_agg = pd.concat([df_agg[['Country', 'Class']], missing_df_plot, countries_to_add_plot], ignore_index=True)
# %%
fig_agg = px.choropleth(
    combined_df_agg,
    locationmode = "country names",
    locations = "Country",
    title = "Funding Status by Country (Agglomerative Clustering)",
    color = "Class",
    color_discrete_map = global_color_map,
    labels={'Class': 'Funding_Status'}
)
fig_agg.update_layout(
        geo=dict(
            scope='world',
            projection=dict(type='natural earth') 
        )
    )
fig_agg.show()
# %%
# DBScan Parameters
from sklearn.neighbors import NearestNeighbors
k = 6
nn = NearestNeighbors(n_neighbors=k)
nn.fit(clean_pca_df)
distances, indices = nn.kneighbors(clean_pca_df)
dist_k_neighbor = np.sort(distances[:, k-1])
# %%
plt.figure(figsize=(12, 6))
plt.plot(dist_k_neighbor)
plt.title(f'k={k} KNN Distance Plot for DBSCAN (Elbow Method)') 
plt.xlabel("Points (sorted by distance)") 
plt.ylabel(f"Distance to {k}th Neighbor") 
plt.grid(True)
plt.show()
# %%
## According to the graph, we should choose 0.4 for the best result.
# %%
db_model = DBSCAN(eps=0.4, min_samples=6) 
db_labels = db_model.fit_predict(clean_pca_df)
# %%
db_mask = db_labels != -1
n_clusters_db = len(set(db_labels[db_mask])) 
print(f"DBSCAN found {n_clusters_db} clusters and {np.sum(~db_mask)} noise points.")
# %%
if n_clusters_db > 1:
    db_score = silhouette_score(clean_pca_df[db_mask], db_labels[db_mask])
    silhouette_scores['DBSCAN'] = db_score
    print(f"DBSCAN Silhouette Score (excluding noise): {db_score:.4f}")
else:
    silhouette_scores['DBSCAN'] = -1 
    print("DBSCAN found less than 1 cluster, Silhouette Score could not be calculated.")
# %%
df_db = df[['country', 'income']].copy()
df_db['Class_Label'] = db_labels
db_means = df_db[df_db['Class_Label'] != -1].groupby('Class_Label')['income'].mean()
db_map = { -1: "Noise" } 
# %%
if len(db_means) > 0: # If at least 1 cluster was found
    db_budget_needed = db_means.idxmin()
    db_map[db_budget_needed] = "Budget Needed"
    
    if len(db_means) > 1: # If more than 1 cluster was found
        db_no_budget = db_means.idxmax()
        db_map[db_no_budget] = "No Budget Needed"
        
        # Make the rest 'In Between'
        for label in db_means.index:
            if label not in [db_budget_needed, db_no_budget]:
                db_map[label] = "In Between"
# %%
df_db['Class'] = df_db['Class_Label'].map(db_map)
df_db = df_db.rename(columns={'country': 'Country'})
combined_df_db = pd.concat([df_db[['Country', 'Class']], missing_df_plot, countries_to_add_plot], ignore_index=True)
# %%
fig_db = px.choropleth(
    combined_df_db,
    locationmode = "country names",
    locations = "Country",
    title = "Funding Status by Country (DBSCAN)",
    color = "Class",
    color_discrete_map = global_color_map,
    labels={'Class': 'Funding_Status'}
)
fig_db.update_layout(
        geo=dict(
            scope='world',
            projection=dict(type='natural earth') 
        )
    )
fig_db.show()
# %%
# HDBscan 
hdb_model = hdbscan.HDBSCAN()
hdb_labels = hdb_model.fit_predict(clean_pca_df)
# %%
hdb_mask = hdb_labels != -1
n_clusters_hdb = len(set(hdb_labels[hdb_mask]))
print(f"HDBSCAN found {n_clusters_hdb} clusters and {np.sum(~hdb_mask)} noise points.")
# %%
if n_clusters_hdb > 1:
        hdb_score = silhouette_score(clean_pca_df[hdb_mask], hdb_labels[hdb_mask])
        silhouette_scores['HDBSCAN'] = hdb_score
        print(f"HDBSCAN Silhouette Score (excluding noise): {hdb_score:.4f}")
else:
        silhouette_scores['HDBSCAN'] = -1
        print("HDBSCAN found less than 1 cluster, Silhouette Score could not be calculated.")
# %%
df_hdb = df[['country', 'income']].copy()
df_hdb['Class_Label'] = hdb_labels
hdb_means = df_hdb[df_hdb['Class_Label'] != -1].groupby('Class_Label')['income'].mean()
hdb_map = { -1: "Noise" }
# %%
if len(hdb_means) > 0:
        hdb_budget_needed = hdb_means.idxmin()
        hdb_map[hdb_budget_needed] = "Budget Needed"
        
        if len(hdb_means) > 1:
            hdb_no_budget = hdb_means.idxmax()
            hdb_map[hdb_no_budget] = "No Budget Needed"
            
            for label in hdb_means.index:
                if label not in [hdb_budget_needed, hdb_no_budget]:
                    hdb_map[label] = "In Between"
# %%
df_hdb['Class'] = df_hdb['Class_Label'].map(hdb_map)
df_hdb = df_hdb.rename(columns={'country': 'Country'})
combined_df_hdb = pd.concat([df_hdb[['Country', 'Class']], missing_df_plot, countries_to_add_plot], ignore_index=True)
# %%
fig_hdb = px.choropleth(
        combined_df_hdb,
        locationmode = "country names",
        locations = "Country",
        title = "Funding Status by Country (HDBSCAN)",
        color = "Class",
        color_discrete_map = global_color_map,
        labels={'Class': 'Funding_Status'}
    )
fig_hdb.update_layout(
        geo=dict(
            scope='world',
            projection=dict(type='natural earth') 
        )
    )
fig_hdb.show()
# %%
# Score comparison plot
scores_df = pd.DataFrame(list(silhouette_scores.items()), columns=['Algorithm', 'Silhouette_Score'])
scores_df = scores_df.sort_values(by='Silhouette_Score', ascending=False)
# %%
plt.figure(figsize=(12, 7))
sns.set(style="whitegrid", font_scale=1.1) 
bar = sns.barplot(
    data=scores_df,
    x="Silhouette_Score",
    y="Algorithm",        
    palette="viridis",    
    edgecolor="black"     
)
plt.title('Silhouette Score Comparison for All Models', fontsize=16, fontweight="bold", pad=20)
plt.xlabel("Silhouette Score", fontsize=14)
plt.ylabel("Algorithm", fontsize=14)
min_score = scores_df['Silhouette_Score'].min()
plt.xlim(min_score - 0.1, 1.0) 
for i, (value, name) in enumerate(zip(scores_df["Silhouette_Score"], scores_df["Algorithm"])):
    plt.text(value + 0.01,i, f"{value:.3f}", va="center", fontsize=11 , color="black")                   
plt.tight_layout()
plt.show()
# %%











    
    
    





















