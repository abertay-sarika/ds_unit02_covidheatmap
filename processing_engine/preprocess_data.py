import pandas as pd
from pandas.core.frame import DataFrame
from typing import List

def csv_reader(file_path):
    '''
    Function for reading csv file
        Args
            file_path(string):path of the csv file
        Returns 
            (dictionary):csv parsed data  or None  
    '''
    return pd.read_csv(file_path) 

def clean_data(covid_data: DataFrame, tobe_dropped: List[str]) -> DataFrame:
    '''
    Function for preprocessing the input data
        Args
            covid_data(DataFrame): raw input data as DataFrame
        Returns 
            covid_data(DataFrame): cleaned covid_data as Dataframe 
    '''
    #dropping column as "Province/State" contains too many missing values
    covid_data.drop(tobe_dropped, 1, inplace=True)
    covid_data["ObservationDate"] = pd.to_datetime(covid_data["ObservationDate"]) #coverting "ObservationDate" in to Datetime format
    covid_data.rename(columns={"Country/Region":"Country"}, inplace=True)
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
    return covid_data.loc[mask]

def  group_by_date(covid_data: DataFrame):
    '''
    Function for grouping data by date
        Args
            covid_data(DataFrame): raw input data as DataFrame
        Returns 
            (DataFrame): extracted data 
    '''   
    #grouping different types of cases as per the date
    covid_data = covid_data.groupby(["ObservationDate"]).agg({"Confirmed":"sum","Recovered":"sum","Deaths":"sum"})
    return covid_data     