# Movie Success Predictor

# Overview
This project predicts the popularity of movies using features such as vote count, vote average, and release date. 
The goal is to explore movie trends and build a regression model to estimate popularity scores for new movies.

## Dataset
- TMDB 10,000 Movies Dataset from Kaggle
- Columns used:
  - vote_average
  - vote_count
  - release_date
  - popularity (target)
- Download link: [Kaggle The movie database top 10000 movies all time] https://www.kaggle.com/datasets/xcufx0qc2os1/the-movie-database-top-10000-movies-all-time 

# Features & Methods
1. Loaded and cleaned the dataset (removed missing overview rows)
2. Converted movie overviews to TF-IDF vectors (max_features=5000)
3. Scaled numeric features (vote_average, vote_count)
4. Combined text and numeric features
5. Trained Ridge Regression to predict popularity
6. Evaluated using Mean Squared Error and R² Score

# Results & Evaluation 
Model Evaluation (Ridge Regression with log-transformed popularity):
- Mean Squared Error: 0.34
- R² Score: 0.28

A scatter plot of predicted vs actual popularity is included to visualize model performance.

# How to run 
1. Clone the repository
2. Install dependencies: pandas, scikit-learn, matplotlib, numpy, scipy
3. Run `movie_success_predictor.py`
