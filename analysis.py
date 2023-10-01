import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV

tracks = pd.read_csv("tracks.csv")
data = tracks.copy()

label_encoder = LabelEncoder()

data["Genre"] = label_encoder.fit_transform(data["Genre"])

X = data.drop(["Genre", "Track Name"], axis=1)
y = data.loc[:, "Genre"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

random = RandomForestClassifier(bootstrap=True, criterion="entropy", max_depth=None, min_samples_leaf=2,
                                min_samples_split=7, n_estimators=100)

random.fit(X_train, y_train)

"""
param_grid = {
    'n_estimators': [50, 70, 100],
    'max_depth': [None, 1, 3, 20],
    'min_samples_split': [5, 7, 9],
    'min_samples_leaf': [1, 2, 3],
    'bootstrap': [True, False],
    'criterion': ['gini', 'entropy']
}

grid_search = GridSearchCV(estimator=random, param_grid=param_grid, cv=5)
grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
print("Best Parameters:", best_params)

"""
random_pred = random.predict(X_test)

print("Random report:")
print(classification_report(y_test, random_pred))

print("*******************")

print("*******************")

print("Random accuracy:")
print(accuracy_score(y_test, random_pred))
