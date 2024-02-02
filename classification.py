import pandas as pd
import matplotlib.pyplot as plt
import warnings
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, matthews_corrcoef

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier, \
    VotingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
warnings.filterwarnings("ignore")

data = pd.read_excel("tracks_with_labels.xlsx")

data.head()

spo_cols = ["Speechiness", "Duration", "Valence",
            "Instrumentalness", "Key", "Danceability",
            "Energy", "Loudness", "Mode", "Acousticness", "Cluster"]

df = data[spo_cols]

X = df.drop("Cluster", axis=1)
y = df["Cluster"]

scaler = MinMaxScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

rf_params = {"max_depth": [8, 15, None],
             "max_features": [5, 7, "auto"],
             "n_estimators": [300, 500, 1000]}

cart_params = {'max_depth': range(1, 20),
               "min_samples_split": range(2, 30)}

gb_params = {"max_depth": [5, 8, 12, None],
             'learning_rate': [0.01, 0.1, 0.2, 0.3]}

svc_params = {'C': [0.1, 1, 10],
              'kernel': ['linear', 'rbf'],
              'gamma': ['scale', 'auto', 0.1, 1],
              'degree': [2, 3, 4]}

ada_params = {'n_estimators': [50, 100, 200],
              'learning_rate': [0.01, 0.1, 0.2, 0.5],
              'algorithm': ['SAMME', 'SAMME.R'],
              'estimator': [None, DecisionTreeClassifier(), RandomForestClassifier()]}

bagging_params = {"n_estimators": [300, 500, 1000],
                  "max_features": [0.5, 0.7, 1.0],
                  "max_samples": [0.5, 0.7, 1.0]}

nb_params = {'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]}

boost_params = {"learning_rate": [0.01, 0.1, 0.2],
                "max_depth": range(3, 10),
                'reg_alpha': [0, 0.1, 0.2],
                'reg_lambda': [0, 0.1, 0.2],
                "n_estimators": [300, 500, 1000]}

classifiers = [("Random Forest", RandomForestClassifier(), rf_params),
               ("CART", DecisionTreeClassifier(), cart_params),
               ("Naive Bayes", GaussianNB(), nb_params),
               ("SVM", SVC(), svc_params),
               ("ADA", AdaBoostClassifier(), ada_params),
               ("Bagging", BaggingClassifier(), bagging_params),
               ("Gradient Boosting", GradientBoostingClassifier(), gb_params),
               ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss', verbosity=0), boost_params),
               ('LightGBM', LGBMClassifier(verbose=-1), boost_params)
               ]


def hyperparameter_optimization(X, y, classifiers, cv=5, scoring="accuracy"):
    print("Hyperparameter Optimization....")
    best_models = {}
    for name, classifier, params in classifiers:
        print(f"########## {name} ##########")
        cv_results = cross_val_score(classifier, X, y, cv=cv, scoring=scoring, n_jobs=-1)
        print(f"{scoring} (Before): {cv_results.mean()}")

        gs_best = GridSearchCV(classifier, params, cv=cv, n_jobs=-1, verbose=False).fit(X, y)
        final_model = classifier.set_params(**gs_best.best_params_)

        cv_results = cross_val_score(final_model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
        print(f"{scoring} (After): {cv_results.mean()}")
        print(f"{name} best params: {gs_best.best_params_}", end="\n\n")
        best_models[name] = {"Model": final_model, "Accuracy": cv_results.mean()}
    return best_models


best_models = hyperparameter_optimization(X_train, y_train, classifiers=classifiers)

top_models = sorted(best_models.values(), key=lambda x: x["Accuracy"], reverse=True)[:5]

voting_model = VotingClassifier(
    estimators=[
        ("1st", top_models[0]["Model"]),
        ("2nd", top_models[1]["Model"]),
        ("3rd", top_models[2]["Model"]),
        ("4th", top_models[3]["Model"]),
        ("5th", top_models[4]["Model"]),
    ], voting="hard")

voting_model.fit(X_train,y_train)

predictions = voting_model.predict(X_test)

print(classification_report(y_test,predictions))

cm = confusion_matrix(y_test,predictions)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=voting_model.classes_)

disp.plot(cmap='plasma')
plt.title('Confusion Matrix')
plt.savefig("images/conf_matrix.png")
plt.show()

joblib.dump(voting_model, "model/genre.pkl")
