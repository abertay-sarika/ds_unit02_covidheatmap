#importing requered libraries
import warnings
warnings.filterwarnings('ignore')

import sys
import argparse
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
import plotly.express as px
import plotly.graph_objects as go
from os.path import exists
from pandas.core.frame import DataFrame
from datetime import timedelta
from sklearn.svm import SVR
#from sklearn.model_selection import train_test_split # Import train_test_split function
#from sklearn.preprocessing import StandardScaler
#from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
from sklearn import metrics
from sklearn.metrics import accuracy_score
from geopy.geocoders import Nominatim
from typing import List
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from preprocess_data import csv_reader, group_by_date, extract_data, clean_data
from visualisation import bar_plot, line_graph, bar_plot_weekwise
import reporting

  

def confirmed_cases_count(covid_data: DataFrame, r: reporting.Output):
    '''
    Function for getting count of Confirmed covid cases globally by extracted_data
        Args
            covid_data(DataFrame): extracted input data
        Returns 
            (float): count of confirmed covid cases  
    '''
    r.set_confirmed_cases(covid_data["Confirmed"].iloc[-1])

def recovered_cases_count(covid_data: DataFrame, r: reporting.Output):
    '''
    Function for getting count of Recovered covid cases globally by extracted_data
        Args
            covid_data(DataFrame): extracted input data
        Returns 
            (float): count of Recovered covid cases 
    '''
    r.set_recovered_cases(covid_data["Recovered"].iloc[-1])

def death_count(covid_data: DataFrame, r: reporting.Output):
    '''
    Function for getting count of covid deaths globally by extracted_data
        Args
            covid_data(DataFrame): extracted input data
        Returns 
            (float): count of covid deaths
    '''
    r.set_death_count(covid_data["Deaths"].iloc[-1])

def active_cases_count(covid_data: DataFrame, r: reporting.Output):
    '''
    Function for getting count of active cases globally by extracted_data
        Args
            covid_data(DataFrame): extracted input data
        Returns 
            (float): count of active cases 
    '''
    r.set_active_cases(covid_data["Confirmed"].iloc[-1] 
        - covid_data["Recovered"].iloc[-1]
        - covid_data["Deaths"].iloc[-1])

def closed_cases_count(covid_data: DataFrame, r: reporting.Output):
    '''
    Function for getting count of closed cases globally by extracted_data
        Args
            covid_data(DataFrame): extracted input covid data
        Returns 
            (float): count of closed cases
    '''
    r.set_closed_cases(covid_data["Recovered"].iloc[-1] + covid_data["Deaths"].iloc[-1])

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

def global_covid_status(r: reporting.Output):
    '''
    Function for printing basic information in the covid dataset
        Args
            covid_data(DataFrame): extracted input data
        Returns 
            None
    '''
    print("---Basic Information---")
    #print("First five raws as sample", covid_data.head(5))
    #print("Total number of countries with Disease Spread:", len(covid_data["Country"].unique()))
    print("Total number of Confirmed cases around the world", r.confirmed_cases)
    print("Total number of Recovered cases around the world", r.recovered_cases)
    print("Total number of Covid Death around the world", r.death_count)
    print("Total number of Active cases around the world", r.active_cases)
    print("Total number of Closed cases around the world", r.closed_cases)


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

    weekly_data ={
        'weekwise_confirmed': {
            'val': weekwise_confirmed,
            'label':str("weekly growth of confirmed case")
        },
        "weekwise_recovered": {
            'val' : weekwise_recovered,
            'label' :str("weekly growth of recovered case")
        },
        "weekwise_deaths": {
            'val' : weekwise_deaths,
            'label' : str("weekly growth of death case")
        }
    }
    return (weekly_data, week_num)


def top_n_country(n: int, reports_dir, covid_data: DataFrame):
    '''
    Function for plotting line graph
        Arg
            x,y(position) : label
            figsize(tuple) : width and height of the graph
            title(str) : title for bar charts
            xticks(sequence) : positioning of values ?
        Returns 
             distribution plot(line) : visualized data
def bar_plot_weekwise(file_name, x, y1, y2, subplot_title1, subplot_title2, figsize, x_label=None, y_label1=None, y_label2=None):
    '''
    countrywise = covid_data[covid_data["ObservationDate"]==covid_data["ObservationDate"].max()].groupby(["Country"]).agg({"Confirmed":"sum","Recovered":"sum","Deaths":"sum"}).sort_values(["Confirmed"],ascending=False)
    figsize = (25,10)
   
    top_20confirmed = countrywise.sort_values(["Confirmed"],ascending=False).head(20)
    top_20deaths = countrywise.sort_values(["Deaths"],ascending=False).head(20)
    x1 = top_20confirmed["Confirmed"]
    y = top_20confirmed.index
    subplot_title1= "Top 20 countries as per number of confirmed cases"
    x2=top_20deaths["Deaths"]
    subplot_title2="Top 20 countries as per number of death cases"
    bar_plot_weekwise(
        reports_dir + "/plot5",
        subplot_title1, 
        subplot_title2,
        (25,10),
        x1,
        x2,
        y, 
        y)
    

# def uk_covid_data(uk_data, datewise_uk):
#     '''
#     Function for printing basic information in the covid dataset
#         Args
#             covid_data(DataFrame): extracted input data
#         Returns 
#             None
#     '''
#     data = uk_data
#     datewise_data = datewise_uk
#     print(datewise_uk.iloc[-1])
#     print("Total Active Cases", total_active_cases)
#     print("Total Closed Cases", total_closed_cases)
#     print("Average increase in number of Confirmed Cases every day: ", mean_confirmed_cases)
#     print("Average increase in number of Recovered Cases every day: ", mean_recovered_cases)
#     print("Average increase in number of Deaths Cases every day: ", mean_death_count)    
    

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

    # initialising the 'reporting' object
    # this object stores important variable values will be used for final o/p display
    output = reporting.Output()
    reports_dir = reporting.create_reports_dir()

    csv_data = csv_reader(file_path=path)

    drop_list = ["Province/State"]
    cleaned_data = clean_data(covid_data=csv_data, tobe_dropped=drop_list)
    
    extracted_data = extract_data(covid_data = cleaned_data, start_date = start_date, end_date = end_date)
    datewise = group_by_date(covid_data=extracted_data)
    #print(extracted_data.head(5))

    global_covid_status(r = output)
    data_sanity_check_output(covid_data=extracted_data)
    
    # Distribution plot for Active cases
    x_axis = datewise.index.date
    activecases_y_axis = datewise["Confirmed"] - datewise["Recovered"] - datewise["Deaths"]
    activecases_plot_title = "Distribution plot for Active cases"
    bar_plot(file_name = reports_dir + "/plot1", 
        x = x_axis,
        y=activecases_y_axis,
        figsize = (15,5),
        title=activecases_plot_title,
        rotation=90)
   
    # Distribution plot for Closed cases
    closedcases_y_axis = datewise["Recovered"] + datewise["Deaths"]
    closedcases_plot_title = "Distribution plot for Closed cases"
    bar_plot(file_name = reports_dir + "/plot2",
        x = x_axis,
        y=closedcases_y_axis,
        figsize = (15,5),
        title=closedcases_plot_title,
        rotation=90)

    (weekly_data, week_num) = report_weekly_data(covid_data=datewise)
    
    line_graph(file_name = reports_dir + "/plot3", week_num=week_num,data = weekly_data)  

    # Distribution plot weekwise
    x = week_num
    y1 = pd.Series(weekly_data["weekwise_confirmed"]["val"]).diff().fillna(0)
    y2 = pd.Series(weekly_data["weekwise_recovered"]["val"]).diff().fillna(0)
    x_label = "Week Number"
    y_label1 = "Number of Confirmed cases"
    y_label2 = "Number of Recovered cases"
    subplot_title1 = "Weekly increase in number of confirmed case"
    subplot_title2 = "weekly increase in number of Death cases"
    bar_plot_weekwise(
        reports_dir + "/plot4",
        subplot_title1, 
        subplot_title2,
        (12,4),
        x,
        x,
        y1, 
        y2, 
        x_label, 
        y_label1, 
        y_label2)
        
    top_n_country(10, reports_dir, extracted_data)    

    # # countrywise analysis
    # countrywise_x1_axis = top_20confirmed["Confirmed"]
    # countrywise_y1_axis = top_20confirmed.index
    # Countrywise_title1 = "Top 20 countries as per number of confirmed cases"
    # ountrywise_x2_axis= top_20deaths["Deaths"]
    # ountrywise_y2_axis= top_20confirmed.index
    # Countrywise_title2 = "Top 20 countries as per number of death cases"

    # # uk analysis
    # uk_data = covid_data[covid_data["Country"]=="UK"]
    # datewise_uk = uk_data.groupby(["ObservationDate"]).agg({"Confirmed":"sum","Recovered":"sum","Deaths":"sum"})
    # total_active_cases = datewise_uk["Confirmed"].iloc[-1]- datewise_uk["Recovered"].iloc[-1] - datewise_uk["Deaths"].iloc[-1]
    # total_closed_cases = datewise_uk["Recovered"].iloc[-1] + datewise_uk["Deaths"].iloc[-1]
    # mean_confirmed_cases = np.round(datewise_uk["Confirmed"].diff().fillna(0).mean())
    # mean_recovered_cases = np.round(datewise_uk["Recovered"].diff().fillna(0).mean())
    # mean_death_count = np.round(datewise_uk["Deaths"].diff().fillna(0).mean())

    # x = datewise_uk["Confirmed"].diff().fillna(0)
    # y = datewise_uk["Recovered"].diff().fillna(0)
    # z = datewise_uk["Deaths"].diff().fillna(0)
    # x_label = "Timestamp"
    # y_label = "Daily increase"
    # uk_title = "Dialy increase in uk"


