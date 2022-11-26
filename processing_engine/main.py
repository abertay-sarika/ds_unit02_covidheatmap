#importing requered libraries
import warnings
warnings.filterwarnings('ignore')

import sys
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
import argparse
from os.path import exists
from pandas.core.frame import DataFrame
from datetime import timedelta
from sklearn.svm import SVR
#from sklearn.model_selection import train_test_split # Import train_test_split function
#from sklearn.preprocessing import StandardScaler
#from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error
from sklearn import metrics
from sklearn.metrics import accuracy_score
from geopy.geocoders import Nominatim
from typing import List
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from reader import read_from_csv, group_by_date, extract_data, clean_data
from visualisation import bar_plot, line_graph

  

def confirmed_cases_count(covid_data: DataFrame) -> float:
    '''
    Function for getting count of Confirmed covid cases globally by extracted_data
        Args
            covid_data(DataFrame): extracted input data
        Returns 
            (float): count of confirmed covid cases  
    '''
    return covid_data["Confirmed"].iloc[-1]

def recovered_cases_count(covid_data: DataFrame) -> float:
    '''
    Function for getting count of Recovered covid cases globally by extracted_data
        Args
            covid_data(DataFrame): extracted input data
        Returns 
            (float): count of Recovered covid cases 
    '''
    return covid_data["Recovered"].iloc[-1]

def death_count(covid_data: DataFrame) -> float:
    '''
    Function for getting count of covid deaths globally by extracted_data
        Args
            covid_data(DataFrame): extracted input data
        Returns 
            (float): count of covid deaths
    '''
    return covid_data["Deaths"].iloc[-1]

def active_cases_count(covid_data: DataFrame) -> float:
    '''
    Function for getting count of active cases globally by extracted_data
        Args
            covid_data(DataFrame): extracted input data
        Returns 
            (float): count of active cases 
    '''
    return covid_data["Confirmed"].iloc[-1] - covid_data["Recovered"].iloc[-1] - covid_data["Deaths"].iloc[-1]

def closed_cases_count(covid_data: DataFrame) -> float:
    '''
    Function for getting count of closed cases globally by extracted_data
        Args
            covid_data(DataFrame): extracted input covid data
        Returns 
            (float): count of closed cases
    '''
    return (covid_data["Recovered"].iloc[-1] + covid_data["Deaths"].iloc[-1])

def data_sanity_check_output(covid_data: DataFrame):
    '''
    Function for printing shape and type of the data, and checking null values 
        Args
            covid_data(DataFrame): extracted input data
        Returns 
            None
    '''
    print("size/shape of the dataset:", covid_data.shape)
    print("checking for null values:\n", covid_data.isnull().sum())
    print("checking Data-type of each column:\n", covid_data.dtypes)

def global_covid_status(covid_data: DataFrame):
    '''
    Function for printing basic information in the covid dataset
        Args
            covid_data(DataFrame): extracted input data
        Returns 
            None
    '''
    print("---Basic Information---")
    print("First five raws as sample", covid_data.head(5))
    print("Total number of countries with Disease Spread:", len(covid_data["Country"].unique()))
    print("Total number of Confirmed cases around the world", confirmed_cases_count(covid_data))
    print("Total number of Recovered cases around the world", recovered_cases_count(covid_data))
    print("Total number of Covid Death around the world", death_count(covid_data))
    print("Total number of Active cases around the world", active_cases_count(covid_data))
    print("Total number of Closed cases around the world", closed_cases_count(covid_data))

def report_weekly_data(covid_data: DataFrame):
    '''
    Function for getting global covid data by week
        Arg
            covid_data(DataFrame) : extracted_data covid data
        Returns 
            data(dictionary) : weekwise covid data
    '''
    covid_data["WeekofYear"] = covid_data.index.weekofyear
    week_num = []
    weekwise_confirmed = []
    weekwise_recovered = []
    weekwise_deaths = []
    w = 1
    for i in list(covid_data["WeekofYear"].unique()):
        weekwise_confirmed.append(covid_data[covid_data["WeekofYear"]==i]["Confirmed"].iloc[-1])
        weekwise_recovered.append(covid_data[covid_data["WeekofYear"]==i]["Recovered"].iloc[-1])
        weekwise_deaths.append(covid_data[covid_data["WeekofYear"]==i]["Deaths"].iloc[-1])
        week_num.append(w)
        w=w+1

    data =[
        {
            'val': weekwise_confirmed,
            'label':str("weekly growth of confirmed case")
        },
        {
            'val' : weekwise_recovered,
            'label' :str("weekly growth of recovered case")
        },
        {
            'val' : weekwise_deaths,
            'label' : str("weekly growth of death case")
        }
    ]
    line_graph(week_num, data)

def parse_cmd_args():
    # Create the parser and add arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--filepath', type= str, help="path to the input csv file")
    parser.add_argument('--startdate', type= str, help="path to the input csv file")
    parser.add_argument('--enddate', type= str, help="path to the input csv file")

    # Parse and print the results
    args = parser.parse_args()
    return args
 

#-----------main------------#
if __name__ == '__main__':
    start_date = '2020-01-21'
    end_date = '2020-04-30'
    cmd_args = parse_cmd_args()
    path = cmd_args.filepath
    if cmd_args.startdate is not None:
        start_date = cmd_args.startdate
    if cmd_args.enddate is not None:    
        end_date = cmd_args.enddate
    '''
    Checking if the file exist. None existense of file will fail the program with exit code 1
    '''
    if not exists(path):
        print('file not found')
        exit(1)

    csv_data = read_from_csv(file_path=path)

    drop_list = ["Province/State","SNo"]
    cleaned_data = clean_data(covid_data=csv_data, tobe_dropped=drop_list)
    
    extracted_data = extract_data(covid_data = cleaned_data, start_date = start_date, end_date = end_date)
    datewise = group_by_date(covid_data=extracted_data)
    #print(extracted_data.head(5))

    global_covid_status(covid_data=extracted_data)
    data_sanity_check_output(covid_data=extracted_data)

    # Distribution plot for Active cases
    x_axis = datewise.index.date
    activecases_y_axis = datewise["Confirmed"] - datewise["Recovered"] - datewise["Deaths"]
    activecases_plot_title = "Distribution plot for Active cases"
    plot_1 = bar_plot(
        x = x_axis,
        y=activecases_y_axis,
        figsize = (15,5),
        title=activecases_plot_title,
        rotation=90)

    # Distribution plot for Closed cases
    closedcases_y_axis = datewise["Recovered"] + datewise["Deaths"]
    closedcases_plot_title = "Distribution plot for Closed cases"
    plot_2 = bar_plot(
        x = x_axis,
        y=closedcases_y_axis,
        figsize = (15,5),
        title=closedcases_plot_title,
        rotation=90)

    report_weekly_data(covid_data=datewise)    







