import numpy as np
import pandas as pd
import seaborn as sns
from datetime import datetime
from facebook_prophet.models_configuration_data import models_data_center

from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error

import warnings
warnings.filterwarnings("ignore")


models_predictiion_outputs = []

example_regressors = regressors = [
    {
        "label": "film",
        "path": "C:/Users/alkrd/Desktop/graduation_project/the_project/Classification/output/regions/north_america_australia/genral_labeled_data_with_relative_traffic_rates/Entertainment/extended_data/canada_trend_data/regressors/film.csv",
    },
    {
        "label": "game",
        "path": "C:/Users/alkrd/Desktop/graduation_project/the_project/Classification/output/regions/north_america_australia/genral_labeled_data_with_relative_traffic_rates/Entertainment/extended_data/canada_trend_data/regressors/game.csv",
    },
    {
        "label": "soccer",
        "path": "C:/Users/alkrd/Desktop/graduation_project/the_project/Classification/output/regions/north_america_australia/genral_labeled_data_with_relative_traffic_rates/Entertainment/extended_data/canada_trend_data/regressors/soccer.csv",
    },
]

example_country_trend_data_path = "C:/Users/alkrd/Desktop/graduation_project/the_project/Classification/output/regions/north_america_australia/genral_labeled_data_with_relative_traffic_rates/Entertainment/extended_data/canada.csv"

def build_model(country_trend_data_path, regressors, split_date=pd.to_datetime("2016-02-01")):
    regressors_data = []
    data = None
    error_metrics = {
        "MSE":0,
        "MAE":0,
    }
    
    if regressors:
        for regressor in regressors:
            data = pd.read_csv(regressor["path"], parse_dates=["date"])
            data.set_index("date", inplace=True)
            data.rename(columns={"value": regressor["label"]}, inplace=True)
            regressors_data.append(data)


    country_trend_data = pd.read_csv(country_trend_data_path, parse_dates=["date"])
    country_trend_data.set_index("date", inplace=True)

    data = country_trend_data.join(
        regressors_data,
        how="left",
    )
    
    train_subset = data.loc[data.index < split_date].copy()
    test_subset = data.loc[data.index > split_date].copy()
    
    # rename the date and target columns to make them suitable for the prophet model
    Pmodel_train_subset = train_subset.reset_index() \
        .rename(columns={
            'date':'ds',
            'score':'y'
        })
        
    print("===== next step is initializing the prophet model amogos")
    model = Prophet()
    
    if regressors:
        for regressor in regressors:
            model.add_regressor(regressor["label"])
    print("====== next step is fitting the model.")
    model.fit(Pmodel_train_subset)

    Pmodel_test_subset = test_subset.reset_index() \
        .rename(columns={
            'date':'ds',
            'score':'y' 
        })
    
    print("====== next step is forecasting the future")
    forecasting_result = model.predict(Pmodel_test_subset)
    
    MSE = np.sqrt(mean_squared_error(y_true=test_subset['score'], y_pred=forecasting_result['yhat']))
    MAE = mean_absolute_error(y_true=test_subset['score'], y_pred=forecasting_result['yhat'])
    error_metrics["MSE"] = MSE
    error_metrics["MAE"] = MAE
    print("*****************")
    print("model:",model)
    print("*****************")
    print("forecasting_result:",forecasting_result)
    print("*****************")
    print("error_metrics:",error_metrics)
    
    return model, forecasting_result, error_metrics, test_subset

def predict_future(model=None, future_preiods=358, data=None, regressors=None):
    if not model:
        raise ValueError("Model is not provided. Please provide a trained model.")
    if data is None or regressors is None:
        raise ValueError("Data and regressors must be provided.")

    future = model.make_future_dataframe(periods=future_preiods, freq='d', include_history=False)

    if regressors:
        for regressor in regressors:
            label = regressor['label']
            future[label] = data[label].mean()
            print(f"Adding regressor: {label}")

    forecast = model.predict(future)
    return forecast

# Build the model and get the forecasting result and error metrics
# model, forecasting_result, error_metrics = build_model(example_country_trend_data_path, example_regressors)
# print("\nmodel is built.\n\n\n")

# Use the model directly without subscripting
# prediction = predict_future(model, data=forecasting_result, regressors=example_regressors)
# print("predictions below:")
# print(prediction[['ds', 'yhat']])

# for model_data in models_data_center:
#     print("###### LOG START ######\n")
#     print(model_data["identifier"],"\n")
#     # print(model_data["country_trend_data_path"],"\n")
#     print("###### LOG END ######")
    
#     model, forecasting_result, error_metrics = build_model(model_data["country_trend_data_path"], model_data["regressors"])
#     prediction = predict_future(model, data=forecasting_result, regressors=model_data["regressors"])
#     output_object = {
#         "identifier":model_data["identifier"],
#         "country":model_data["country"],
#         "topic":model_data["topic"],
#         "prediction":prediction[['ds', 'yhat']]
#     }
#     models_predictiion_outputs.append(output_object)

# print("FINAL OUTPUT AMOGOS:")
# print(models_predictiion_outputs)