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
            (DataFrame): cleaned covid_data as Dataframe 
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
    #to be done -we should check whether the start date is less than end data
    mask = (covid_data["ObservationDate"] > start_date) & (covid_data["ObservationDate"] <= end_date)
    covid_data = covid_data.loc[mask]
    
    #grouping different types of cases as per the date
    covid_data = covid_data.groupby(["ObservationDate"]).agg({"Confirmed":"sum","Recovered":"sum","Deaths":"sum"})
    return covid_data

def confirmed_cases_count(covid_data: DataFrame) -> float:
    '''
    Function for getting count of Confirmed covid cases globally by datewise
        Args
            covid_data(DataFrame): extracted input data as DataFrame
        Returns 
            (float): count of confirmed covid cases  
    '''
    return covid_data["Confirmed"].iloc[-1]

def recovered_cases_count(covid_data: DataFrame) -> float:
    '''
    Function for getting count of Recovered covid cases globally by datewise
        Args
            covid_data(DataFrame): extracted input data as DataFrame
        Returns 
            (float): count of Recovered covid cases 
    '''
    return covid_data["Recovered"].iloc[-1]

def death_count(covid_data: DataFrame) -> float:
    '''
    Function for getting count of covid deaths globally by datewise
        Args
            covid_data(DataFrame): extracted input data as DataFrame
        Returns 
            (float): count of covid deaths
    '''
    return covid_data["Death"].iloc[-1]

def active_cases_count(covid_data: DataFrame) -> float:
    '''
    Function for getting count of active cases globally by datewise
        Args
            covid_data(DataFrame): extracted input data as DataFrame
        Returns 
            (float): count of active cases 
    '''
    return covid_data["Confirmed"].iloc[-1] - covid_data["Recovered"].iloc[-1] - covid_data["Deaths"].iloc[-1]

def closed_cases_count(covid_data: DataFrame) -> float:
    '''
    Function for getting count of closed cases globally by datewise
        Args
            covid_data(DataFrame): extracted input data as DataFrame
        Returns 
            (float): count of closed cases
    '''
    return (covid_data["Recovered"].iloc[-1] + covid_data["Deaths"].iloc[-1])

def data_sanity_check_output(covid_data: DataFrame):
    '''
    Function for printing shape and type of the data, and checking null values 
        Args
            covid_data(DataFrame): extracted input data as DataFrame
        Returns 
            printed results
    '''
    print("size/shape of the dataset:", covid_data.shape)
    print("checking for null values:\n", covid_data.isnull().sum())
    print("checking Data-type of each column:\n", covid_data.dtypes)

def global_covid_status(covid_data: DataFrame):
    '''
    Function for printing basic information in the covid dataset
        Args
            covid_data(DataFrame): extracted input data as DataFrame
        Returns 
            printed results 
    '''
    print("---Basic Information---")
    print("First five raws as sample", covid_data.head(5))
    print("Total number of countries with Disease Spread:", len(covid_data["Country"].unique()))
    print("Total number of Confirmed cases around the world", confirmed_cases_count(covid_data))
    print("Total number of Recovered cases around the world", recovered_cases_count(covid_data))
    print("Total number of Covid Death around the world", death_count(covid_data))
    print("Total number of Active cases around the world", active_cases_count(covid_data))
    print("Total number of Closed cases around the world", closed_cases_count(covid_data))

def plot_data(x, y, figsize, title, xticks):
    '''
    Function for plotting barchart
        Arg
            covid_data(DataFrame) : datewise covid data as DataFrame
            x,y(position) : label
            figsize(tuple) : width and height of the graph
            title(str) : title for bar chart
            xticks(sequence) : positioning of values ?
        Returns 
            barplot : visualized data
    '''
    plt.figure(figsize)
    sns.barplot(x,y)
    plt.title(title)
    plt.xticks(xticks)
    return             ?


def weekwise_data(covid_data: DataFrame):
    '''
    Function for getting global covid data by week
        Arg
            covid_data(DataFrame) : datewise covid data as DataFrame
        Returns 
            data(dictionary) : calculated log values
    '''
    data = {
       'log_data' : []
    } 
    for l in (data_list):
        data['log_data'].append(math.log(l))
    return data    


# #reported cases by week
# datewise["WeekofYear"] = datewise.index.weekofyear
# week_num = []
# weekwise_confirmed = []
# weekwise_recovered = []
# weekwise_deaths = []
# w = 1
# for i in list(datewise["WeekofYear"].unique()):
#     weekwise_confirmed.append(datewise[datewise["WeekofYear"]==i]["Confirmed"].iloc[-1])
#     weekwise_recovered.append(datewise[datewise["WeekofYear"]==i]["Recovered"].iloc[-1])
#     weekwise_deaths.append(datewise[datewise["WeekofYear"]==i]["Deaths"].iloc[-1])
#     week_num.append(w)
#     w=w+1

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
    cleaned_data = cleaning_data(covid_data=csv_data, tobe_dropped=drop_list)

    start_date = '2020-01-21'
    end_date = '2020-04-30'
    extracted_data = extract_data(covid_data = cleaned_data,start_date= start_date, end_date= end_date)
    datewise = extract_data(covid_data = extracted_data)
    
    confirmed_cases = confirmed_cases_count(covid_data=datewise)
    recovered_cases = recovered_cases_count(covid_data=datewise)
    death_counts = death_count(covid_data=datewise)
    active_cases = active_cases_count(covid_data=datewise)
    closed_cases = closed_cases_count(covid_data=datewise)

    dataset_info = data_sanity_check_output(covid_data=datewise)
    global_covid_info = global_covid_status(covid_data= datewise)

    fig_cize = (15,5)
    x1_axis = x2_axis = datewise.index.date
    y1_axis = datewise["Confirmed"] - datewise["Recovered"] - datewise["Deaths"]
    y2_axis = datewise["Recovered"] + datewise["Deaths"]
    plot1_title = "Distribution plot for Active cases"
    plot2_title = "Distribution plot for Closed cases"
    plot1_xticks = plot2_xticks = rotation=90 ?
    plot_1 = plot_data(x = x1_axis, y=y1_axis, figsize = fig_cize, title=plot1_title, xticks=plot1_xticks)
    plot_1 = plot_data(x = x2_axis, y=y2_axis, figsize = fig_cize, title=plot2_title, xticks=plot2_xticks)

    






plt.figure(figsize=(15,5))
sns.barplot(x=datewise.index.date, y=datewise["Confirmed"] - datewise["Recovered"] - datewise["Deaths"])
plt.title("Distribution plot for Active cases")
plt.xticks(rotation=90)

plt.figure(figsize=(15,5))
sns.barplot(x=datewise.index.date, y=datewise["Recovered"] + datewise["Deaths"])
plt.title("Distribution plot for Closed cases")
plt.xticks(rotation=90)
