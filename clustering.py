import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import warnings
from skfeature.function.similarity_based import lap_score

pd.set_option("display.max_columns", None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
warnings.filterwarnings("ignore")

data = pd.read_csv("tracks.csv")
df = data.copy()

df["Duration"] = df["Duration"] / 60000

X = df.drop("Track Name", axis=1)

scaler = MinMaxScaler()

X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

feature_names = X.columns
scores = lap_score.lap_score(X.to_numpy())

pca = PCA(n_components=2)
pca_fit = pca.fit_transform(X)
scores = np.hstack((scores, pca_fit[:, 0]))
indices = np.argsort(scores)[::-1]
top_k_features = feature_names[indices[:19]]

X_new = X[top_k_features]

pca = PCA(n_components=2)
pca_fit = pca.fit_transform(X_new)

model = KMeans(n_clusters=4, init="k-means++", random_state=42, n_init=1000)

model.fit(pca_fit)

df["Cluster"] = model.labels_

df.to_excel("tracks_with_labels.xlsx", index=False)
