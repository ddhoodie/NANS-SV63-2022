from enum import Enum
import statsmodels.api as sm
import pandas as pd
import json
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import HuberRegressor
from sklearn.linear_model import RANSACRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import seaborn as sb
from scipy.stats import zscore
from sklearn.metrics import mean_squared_error
import numpy as np
import statsmodels.stats.api as sms

TO_DROP = ["players_on_launch", "players_after_1year", "platform_availability.PC", "genre.action", "genre.adventure",
              "genre.rpg", "genre.simulation", "genre.sports", "genre.puzzle", "genre.horror",
              "genre.survival", "genre.indie", "genre.fps", "genre.mmo",
              "genre.open_world", "genre.story_mode", "genre.strategy"
           ]
TO_CONVERT = ["multiplayer", "platform_availability.PC", "platform_availability.PLAYSTATION",
              "platform_availability.XBOX", "genre.action", "genre.adventure",
              "genre.rpg", "genre.simulation", "genre.sports", "genre.puzzle", "genre.horror",
              "genre.survival", "genre.indie", "genre.fps", "genre.mmo",
              "genre.open_world", "genre.story_mode", "genre.strategy"]


class RegressionModel(Enum):
    OLS = "OLS"
    Huber = "Huber"
    ElasticNet = "ElasticNet"
    RANSAC = "RANSAC"
    RandomForest = "RandomForest"


def adjusted_r_squared(r_squared, X):
    n = len(X)
    p = X.shape[1]
    adjusted = 1 - ((1 - r_squared) * (n - 1) / (n - p - 1))
    return adjusted


# PRAVLJENJE MODELA I PROCENA BROJA IGRACA
def predict(num, filename):
    if num == 1:
        y_var = "players_on_launch"
    else:
        y_var = "players_after_1year"
    with open('data/dataset.json', 'r') as file:
        data = json.load(file)
    games_list = data['games']
    flattened_data = pd.json_normalize(games_list)
    dataset = pd.DataFrame(flattened_data).drop('name', axis=1)
    for col in TO_CONVERT:
        dataset[col] = dataset[col].astype(int)
    X = dataset.drop(TO_DROP, axis=1)
    y = dataset[y_var]
    model = LinearRegression()
    model.fit(X, y)

    with open(filename, 'r') as file:
        data = json.load(file)
    flattened_data = pd.json_normalize(data)
    dataset = pd.DataFrame(flattened_data).drop('name', axis=1)
    for col in TO_CONVERT:
        dataset[col] = dataset[col].astype(int)
    X = dataset.drop(TO_DROP, axis=1)
    predictions = model.predict(X)[0]
    return int(predictions)


# TESTIRANJE MODELA
def test_model(num, model):
    if num == 1:
        y_var = "players_on_launch"
    else:
        y_var = "players_after_1year"
    pd.set_option('display.max_columns', None)

    with open('data/dataset.json', 'r') as file:
        data = json.load(file)

    games_list = data['games']
    flattened_data = pd.json_normalize(games_list)
    dataset = pd.DataFrame(flattened_data).drop('name', axis=1)
    for col in TO_CONVERT:
        dataset[col] = dataset[col].astype(int)

    # TRAZENJE OUTLAJERA
    # z_scores = zscore(dataset[['company_budget', y_var]])
    # threshold = 2.5
    # outliers = (abs(z_scores) > threshold).all(axis=1)
    # outlier_rows = dataset[outliers]
    # print(outlier_rows)

    X = dataset.drop(TO_DROP, axis=1)
    y = dataset[y_var]

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # PRINT DIMENZIJE DATASETA
    # print("Dimenzije Train seta:", X_train.shape[0])
    # print("Dimenzije Validation seta:", X_val.shape[0])

    # MATRICA KORELACIJE (NEMA ZNACAJNIH KORELACIJA)
    # correlation_matrix = X_train.corr()
    # plt.figure(figsize=(12, 8))
    # sb.heatmap(correlation_matrix, annot=True, cmap="coolwarm", linewidths=.5)
    # plt.title("Matrica Korelacija")
    # plt.show()

    if model == RegressionModel.OLS:
        model = LinearRegression()
        model.fit(X_train, y_train)
        model_statsmodel = sm.OLS(y_train, X_train)
        results = model_statsmodel.fit()
        print(results.summary())

        # STVARNE VS PREDVIĐANE VREDNOSTI
        train_predictions = model.predict(X_val)
        plt.scatter(y_val, train_predictions)
        plt.xlabel("Stvarne vrednosti")
        plt.ylabel("Predviđene vrednosti")
        plt.title("Scatter plot: Stvarne vs. Predviđene vrednosti")
        plt.show()

        rmse = np.sqrt(mean_squared_error(y_val, train_predictions))
        print("RMSE on validation set:", rmse)

        # GOLDFELD-QUANDT TEST ZA JEDNAKU VARIJANSU
        gq_test = sms.het_goldfeldquandt(results.resid, X_train)
        print("\nGoldfeld-Quandt test:")
        print("Test statistic:", gq_test[0])
        print("p-value:", gq_test[1])

        # HISTOGRAM GREŠAKA
        # residuals = y_val - train_predictions
        # sb.histplot(residuals, kde=True)
        # plt.xlabel("Greške (reziduali)")
        # plt.ylabel("Brojčane vrednosti")
        # plt.title("Histogram grešaka (reziduala)")
        # plt.show()

        # HOMOSKEDASTIČNOST GREŠAKA (JEDNAKOST VARIJANSI GREŠAKA)
        # plt.scatter(train_predictions, residuals)
        # plt.xlabel("Predviđene vrednosti")
        # plt.ylabel("Greške (reziduali)")
        # plt.title("Homoskedastičnost: Predviđene vrednosti vs. Greške ")
        # plt.axhline(y=0, color='r', linestyle='--')
        # plt.show()

    elif model == RegressionModel.Huber:
        model = HuberRegressor(epsilon=1.35)
        try:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)

            model.fit(X_train_scaled, y_train)

            huber_pred = model.predict(X_train_scaled)
            huber_residuals = y_train - huber_pred
            X_train_sm = sm.add_constant(X_train_scaled)
            model_sm = sm.RLM(huber_residuals, X_train_sm, M=sm.robust.norms.HuberT())
            results_sm = model_sm.fit()
            print(results_sm.summary())

            ess = ((huber_pred - y_train.mean()) ** 2).sum()
            tss = ((y_train - y_train.mean()) ** 2).sum()
            r_squared = ess / tss
            adjusted_r = adjusted_r_squared(r_squared, X_train_scaled)

            print("R-squared:", r_squared)
            print("Adjusted R-squared:", adjusted_r)

        except Exception as e:
            print(f"Error occurred: {e}")
    elif model == RegressionModel.ElasticNet:
        model = ElasticNet(alpha=0.5, l1_ratio=0.5)
        model.fit(X_train, y_train)

        train_predictions = model.predict(X_val)
        r_squared = model.score(X_train, y_train)
        print("R-squared (ElasticNet):", r_squared)

        adjusted_r = adjusted_r_squared(r_squared, X_train)
        print("Adjusted R-squared:", adjusted_r)

        rmse = np.sqrt(mean_squared_error(y_val, train_predictions))
        print("RMSE on validation set:", rmse)

        # print("ElasticNet model coefficients:")
        # for feature, coef in zip(X_train.columns, model.coef_):
        #     print(f"{feature}: {coef}")
        #
        # print("ElasticNet model intercept:", model.intercept_)

        # train_predictions = model.predict(X_train)
        # plt.scatter(y_train, train_predictions)
        # plt.xlabel("Stvarne vrednosti")
        # plt.ylabel("Predviđene vrednosti")
        # plt.title("Scatter plot: Stvarne vs. Predviđene vrednosti (ElasticNet)")
        # plt.show()
        #
        # residuals = y_train - train_predictions
        # sb.histplot(residuals, kde=True)
        # plt.xlabel("Greške (reziduali)")
        # plt.ylabel("Brojčane vrednosti")
        # plt.title("Histogram grešaka (reziduala) (ElasticNet)")
        # plt.show()
        #
        # plt.scatter(train_predictions, residuals)
        # plt.xlabel("Predviđene vrednosti")
        # plt.ylabel("Greške (reziduali)")
        # plt.title("Homoskedastičnost: Predviđene vrednosti vs. Greške (ElasticNet)")
        # plt.axhline(y=0, color='r', linestyle='--')
        # plt.show()
    elif model == RegressionModel.RANSAC:
        model = RANSACRegressor()
        try:
            model.fit(X_train, y_train)

            ransac_pred = model.predict(X_train)
            ransac_residuals = y_train - ransac_pred
            X_train_sm = sm.add_constant(X_train)
            model_sm = sm.RLM(ransac_residuals, X_train_sm, M=sm.robust.norms.HuberT())
            results_sm = model_sm.fit()
            print(results_sm.summary())

            ess = ((ransac_pred - y_train.mean()) ** 2).sum()
            tss = ((y_train - y_train.mean()) ** 2).sum()
            r_squared = ess / tss
            adjusted_r = adjusted_r_squared(r_squared, X_train)

            print("R-squared:", r_squared)
            print("Adjusted R-squared:", adjusted_r)

        except Exception as e:
            print(f"Error occurred: {e}")
    elif model == RegressionModel.RandomForest:
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        try:
            model.fit(X_train, y_train)
            forest_pred = model.predict(X_train)
            forest_residuals = y_train - forest_pred
            X_train_sm = sm.add_constant(X_train)
            model_sm = sm.RLM(forest_residuals, X_train_sm, M=sm.robust.norms.HuberT())
            results_sm = model_sm.fit()
            print(results_sm.summary())

            ess = ((forest_pred - y_train.mean()) ** 2).sum()
            tss = ((y_train - y_train.mean()) ** 2).sum()
            r_squared = ess / tss
            adjusted_r = adjusted_r_squared(r_squared, X_train)

            print("R-squared:", r_squared)
            print("Adjusted R-squared:", adjusted_r)

        except Exception as e:
            print(f"Error occurred: {e}")
