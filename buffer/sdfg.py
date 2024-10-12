import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns

from scipy.stats import skew, norm
import scipy.stats as stats
from scipy.stats import norm
from sklearn.cluster import KMeans

import warnings

warnings.filterwarnings(action="ignore")

data = pd.read_csv('data_3.csv')
df = data.copy()



too_much_na_columns = ['Количество подъездов', 'Жилая', 'number_of_owners', 'number_of_elevators']
df.drop(columns=too_much_na_columns, inplace=True)

columns_to_remove = ['Unnamed: 0', 'Unnamed: 0.1', 'Серия дома']
df.drop(columns=[col for col in columns_to_remove if col in df.columns], inplace=True)

medians = df.select_dtypes(include='number').median()
df.fillna(value=medians, inplace=True)

modes = df.select_dtypes(include='object').mode().iloc[0]
df.fillna(value=modes, inplace=True)

df = df[df['longitude'] > 100]

co = {
    'latitude': df['latitude'],
    'longitude': df['longitude']
}

array = pd.DataFrame(co).dropna()

kmeans = KMeans(n_clusters=5, algorithm='lloyd')
kmeans.fit(array)

df['cluster'] = kmeans.fit_predict(array)

too_much_na_columns = ['latitude', 'longitude']
df.drop(columns=too_much_na_columns, inplace=True)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.utils import resample


X = df.drop(['price'], axis=1)
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, y_train = resample(X_train, y_train, n_samples=len(X_train) * 10, random_state=42)


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV

rf_model = RandomForestRegressor()

categorical_features = X.select_dtypes(include=['object']).columns.tolist()

numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
numerical_features = [feat for feat in numerical_features if feat not in categorical_features]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

rf_param_grid = {
    'model__n_estimators': [100, 200, 300],
    'model__max_features': ['auto', 'sqrt', 'log2'],
    'model__max_depth': [None, 10, 20, 30],
    'model__min_samples_split': [2, 5, 10],
    'model__min_samples_leaf': [1, 2, 4]
}

rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', rf_model)
])

rf_search = RandomizedSearchCV(rf_pipeline, rf_param_grid, n_iter=10, cv=3, verbose=1, random_state=42)
rf_search.fit(X_train, y_train)

print(f'Best parameters for Random Forest: {rf_search.best_params_}')

y_pred_rf = rf_search.predict(X_test)
print(f'Random Forest Validation MSE: {mean_squared_error(y_test, y_pred_rf)}')