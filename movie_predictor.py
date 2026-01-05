import pandas as pd

movies = pd.read_csv("Movie Success Predictor/movies.csv")

print("Shape:", movies.shape)
print("\nColumns:")
print(movies.columns)
