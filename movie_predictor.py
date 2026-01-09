import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

from scipy.sparse import hstack

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
import numpy as np

import matplotlib.pyplot as plt

movies = pd.read_csv("Movie Success Predictor/movies.csv")

print("Shape:", movies.shape)
print("\nColumns:")
print(movies.columns)

# Check for missing values
print("\nMissing values:")
print(movies.isnull().sum())

#Drop rows with missing values
df = movies.dropna(subset=["overview"])
print(df.isnull().sum())

# Features and target
X_text = df["overview"]
scaler = StandardScaler()
X_numeric_scaled = scaler.fit_transform(df[["vote_average", "vote_count"]])
y = np.log1p(df["popularity"])

# Vectorize text data
tfidf = TfidfVectorizer(
    max_features=5000,
    stop_words="english"
)

X_text_tfidf = tfidf.fit_transform(X_text)

# Combine text and numeric features
X = hstack([X_text_tfidf, X_numeric_scaled])

# Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Linear Regression model
model = Ridge(alpha=1.0)
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Evaluate performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation (Ridge Regression with log popularity):")
print("Mean Squared Error:", mse)
print("R² Score:", r2)

# Convert back from log scale for plotting
y_test_actual = np.expm1(y_test)
y_pred_actual = np.expm1(y_pred)

plt.figure(figsize=(8,6))
plt.scatter(y_test_actual, y_pred_actual, alpha=0.3)
plt.xlabel("Actual Popularity")
plt.ylabel("Predicted Popularity")
plt.title("Predicted vs Actual Movie Popularity")
plt.show()

