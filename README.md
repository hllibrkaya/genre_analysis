# Clustering Turkish Songs with Machine Learning Techniques

## Overview

This study presents a methodical approach to the classification of Turkish songs using machine learning
techniques. It involves collecting a substantial dataset of Turkish music, utilizing the Spotify API for
extracting key audio features via Python's librosa library. The glspca method, commonly used in bioinformatics for
feature selection, has been adapted for use in
the research. The research employs unsupervised learning algorithms like K-Means, DBSCAN, and HDBSCAN to identify
intricate patterns within the music data. Subsequently, the study shifts to supervised learning, focusing on model
selection and hyperparameter optimization, leading to the implementation of a Voting Classifier. This classifier
integrates various models to enhance the accuracy of genre classification. The project tries to contribute to the
field of music information retrieval by categorizing Turkish music genres.

## Project Structure

- **scrap.py**: Python script that collects data from Spotify, and process
- **clustering.py**: Clustering gathered dataset with unsupervised learning
- **classification.py** : Python script that involves the transition from unsupervised learning to supervised. To
  eliminate the need for analyzing audio for each new song, only features provided by Spotify were used to predict
  labels, which are assigned in clustering.py.
- **genre_research.ipynb** : Researches conducted before clustering. A Jupyter notebook file where feature selection,
  model selection and dimensionality reduction options are considered.
- **model/genre.pkl**: Includes final trained model for classification
- **images/**: Includes graphs of Confusion matrix, elbow method and clustering results for HDBSCAN and K-means models.
- **tracks.csv**: Gathered dataset. Not listed in this repo due to Spotify's copyright issues.
- **tracks_with_labels.xlsx**: The dataset created after labels have been assigned. Not shared because of same reason
  above.
- **songs/**: Songs downloaded (1 minute versions) with Spotify API. Not listed due to folder size. Make sure to create
  this folder before running scrap.py.

## Clustering and Classification Results

After the clustering process, the resulting silhouette score for K-means is observed to be _0.6_. In the classification
part, the model achieved an accuracy score of _0.94_ on test data, and the confusion matrix can be viewed in the images
folder.

## Methods

- A backoff-retry strategy was implemented for avoiding API request rejections.- Correlation's were observed
- Audio features extracted from songs with Python's librosa library
- Dataset normalized
- GLSPCA method applied for feature selection
- PCA, t-SNE, ICA methods observed for dimensionalty reduction
- K-means, DBSCAN, and HDBSCAN models were explored.
- The optimal number of clusters for K-means was determined based on the silhouette score and the elbow method.
- Grid search-like approach was developed to determine the epsilon and min_samples parameters of DBSCAN
- Automated hyperparameter optimization was performed for 10 different machine learning models, and the models were
  ranked based on the best cross-validation accuracy scores.
- The top 5 models were selected to create a voting classifier, and its performance was tested on the test data.

## Requirements

Make sure to install the required packages before running the script:

```bash
pip install -r requirements.txt
```