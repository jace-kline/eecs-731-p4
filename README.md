# EECS 731 - Project 4 (Regression)
### Author: Jace Kline

## Project Description
Topic: NFL, MLB, NBA and Soccer scores
1. Set up a data science project structure in a new git repository in your GitHub account
2. Pick one of the game data sets depending your sports preference
   * https://github.com/fivethirtyeight/nfl-elo-game
   * https://github.com/fivethirtyeight/data/tree/master/mlb-elo
   * https://github.com/fivethirtyeight/data/tree/master/nba-carmelo
   * https://github.com/fivethirtyeight/data/tree/master/soccer-spi
3. Load the data set into panda data frames
4. Formulate one or two ideas on how feature engineering would help the data set to establish additional value using exploratory data analysis
5. Build one or more regression models to determine the scores for each team using the other columns as features
6. Document your process and results
7. Commit your notebook, source code, visualizations and other supporting files to the git repository in GitHub

## Project Summary
In this project, we performed regression score predictions using the nfl-elo-game dataset. The general strategy that we used was to transform our features into predictors for the score differential in any particular NFL game. We computed a feature that represented an average/expected differential in the team scores based on the average points that each team scored and gave up during the particular season in question. In addition, we bounded the complexity and error of our regression prediction by not only predicting the difference in score instead of the raw scores themselves, but also dividing this difference by 10. This essentially allowed our model to have an error of 10 points in guessing the point differential in any game. We achieved a success of 50.9% using a gradient boosting model. See the full report [here.](./notebooks/scores.md)
