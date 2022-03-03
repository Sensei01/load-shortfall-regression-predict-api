"""

    Helper functions for the pretrained model to be used within our API.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.

    Importantly, you will need to modify this file by adding
    your own data preprocessing steps within the `_preprocess_data()`
    function.
    ----------------------------------------------------------------------

    Description: This file contains several functions used to abstract aspects
    of model interaction within the API. This includes loading a model from
    file, data preprocessing, and model prediction.  

"""

# Helper Dependencies
import numpy as np
import pandas as pd
import pickle
import json

def _preprocess_data(data):
    """Private helper function to preprocess data for model prediction.

    NB: If you have utilised feature engineering/selection in order to create
    your final model you will need to define the code here.
    


    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.

    Returns
    -------
    Pandas DataFrame : <class 'pandas.core.frame.DataFrame'>
        The preprocessed data, ready to be used our model for prediction.
    """
    # Convert the json string to a python dictionary object
    new_variable = json.loads(data)
    # Load the dictionary as a Pandas DataFrame.
    new_variable = pd.DataFrame.from_dict([new_variable])

    # ---------------------------------------------------------------
    # NOTE: You will need to swap the lines below for your own data
    # preprocessing methods.
    #
    # The code below is for demonstration purposes only. You will not
    # receive marks for submitting this code in an unchanged state.
    # ---------------------------------------------------------------

    new_variable = ['Barcelona_pressure', 'Bilbao_wind_deg', 'Seville_clouds_all', 'Barcelona_wind_deg', 'Madrid_clouds_all', 'Bilbao_clouds_all', 'Bilbao_weather_id', 'Barcelona_weather_id', 'Madrid_humidity', 'Bilbao_snow_3h', 'Seville_humidity', 'Madrid_weather_id', 'Seville_weather_id', 'Valencia_humidity', 'Day_of_month', 'Hour_of_day', 'Seville_pressure', 'Barcelona_rain_1h', 'Valencia_wind_speed', 'Month_of_year', 'Valencia_wind_deg', 'Bilbao_wind_speed', 'Madrid_wind_speed', 'Day_of_week', 'Seville_wind_speed', 'Barcelona_wind_speed', 'Bilbao_rain_1h', 'Seville_rain_1h', 'Madrid_rain_1h', 'Madrid_pressure']    

    return new_variable


def load_model(path_to_model:'/assets/trained-models/Linear_model.pkl'):
    
    
    return pickle.load(open(path_to_model, 'rb'))

""" You may use this section (above the make_prediction function) of the python script to implement 
    any auxiliary functions required to process your model's artifacts.
"""

def make_prediction(data, model):
    """Prepare request data for model prediction.

    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.
    model : <class: sklearn.estimator>
        An sklearn model object.

    Returns
    -------
    list
        A 1-D python list containing the model prediction.

    """
    # Data preprocessing.
    prep_data = _preprocess_data(data)
    # Perform prediction with model and preprocessed data.
    prediction = model.predict(prep_data)
    # Format as list for output standardisation.
    return prediction[0].tolist()
