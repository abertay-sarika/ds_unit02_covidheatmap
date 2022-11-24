#importing requered libraries
import warnings
warnings.filterwarnings('ignore')

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import seaborn as sns
import geopandas as gpd
from os.path import exists
from pandas.core.frame import DataFrame
from datetime import timedelta
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error
from sklearn import metrics
from sklearn.metrics import accuracy_score
from geopy.geocoders import Nominatim
from typing import List
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation

def read_from_csv(file_path):
    '''
    Function for reading csv file
        Args
            file_path(string):path of the csv file
        Returns 
            (dictionary):csv parsed data  or None  
    '''
    return pd.read_csv(file_path) 


def cleaning_data(covid_data: DataFrame, tobe_dropped: List[str]) -> DataFrame:
    '''
    Function for preprocessing the input data
        Args
            covid_data(DataFrame): raw input data as DataFrame
        Returns 
            (DataFrame): cleaned DataFrame  
    '''
    #dropping column as "Province/State" contains too many missing values
    covid_data.drop(tobe_dropped, 1, inplace=True)
    covid_data["ObservationDate"] = pd.to_datetime(covid_data["ObservationDate"]) #coverting "ObservationDate" in to Datetime format
    covid_data.rename(columns={"Country/Region":"Country"}, inplace=True)
    covid_data.__setitem__("longitude", 0.0)
    covid_data.__setitem__("latitude", 0.0)
    return covid_data

   
def extract_data(covid_data: DataFrame, start_date: str, end_date: str) -> DataFrame:
    '''
    Function for extracting the data
        Args
            covid_data(DataFrame): raw input data as DataFrame
        Returns 
            (DataFrame): extracted data 
    '''
    mask = (covid_data["ObservationDate"] > start_date) & (covid_data["ObservationDate"] <= end_date)
    covid_data = covid_data.loc[mask]
    
    #grouping different types of cases as per the date
    covid_data = covid_data.groupby(["ObservationDate"]).agg({"Confirmed":"sum","Recovered":"sum","Deaths":"sum"})
    return covid_data

def get_confirmed_cases_count(covid_data: DataFrame) -> float:
    return covid_data["Confirmed"].iloc[-1]

def get_recovered_cases_count(covid_data: DataFrame) -> float:
    return covid_data["Recovered"].iloc[-1]

def get_death_count(covid_data: DataFrame) -> float:
    return covid_data["Death"].iloc[-1]


def get_active_cases_count(covid_data: DataFrame) -> float:
    return covid_data["Confirmed"].iloc[-1] - covid_data["Recovered"].iloc[-1] - covid_data["Deaths"].iloc[-1]

def get_closed_cases_count(covid_data: DataFrame) -> float:
    return (covid_data["Recovered"].iloc[-1] + covid_data["Deaths"].iloc[-1])

def print_data_sanity_check_output(covid_data: DataFrame):
    print("size/shape of the dataset:", covid_data.shape)
    print("checking for null values:\n", covid_data.isnull().sum())
    print("checking Data-type of each column:\n", covid_data.dtypes)

def print_global_covid_status(covid_data: DataFrame):
    print("Basic Information")
    print("Total number of countries with Disease Spread:", len(covid_data["Country/Region"].unique()))
    print("Total number of Confirmed cases around the world", get_confirmed_cases_count(covid_data))
    print("Total number of Recovered cases around the world", get_recovered_cases_count(covid_data))
    print("Total number of Death cases around the world", get_death_count(covid_data))
    print("Total number of Active cases around the world", get_active_cases_count(covid_data))
    print("Total number of Closed cases around the world", get_closed_cases_count(covid_data))


if __name__ == '__main__':

    '''
    program name: data_salary.py

    program accepts command line arguments as inputs

    sys.argv is an array containing all the input arguments. The first argument is the program name itself 
    '''
    if len (sys.argv) < 2:
        print ("Usage: python3 main.py <input csv filename>  ")
        sys.exit (1)

    '''
    Accept file name as a Command-Line Argument
    '''
    path = str(sys.argv[1])
    # print(file_path)
    print(sys.argv)
    '''
    Checking if the file exist. None existense of file will fail the program with exit code 1
    '''
    if not exists(path):
        print('file not found')
        exit(1)

    csv_data = read_from_csv(file_path=path)

    drop_list = ["Province/State","SNo"]
    cleaning_data(covid_data=csv_data, tobe_dropped=drop_list)
