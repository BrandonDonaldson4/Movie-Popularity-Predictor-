import pandas as pd

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

