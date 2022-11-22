#importing requered libraries
import warnings
warnings.filterwarnings('ignore')


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import seaborn as sns
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error

#reading the csv file
df = pd.read_csv("fullcovid_data.csv")
df["ObservationDate"] = pd.to_datetime(df["ObservationDate"]) #coverting "ObservationDate" in to Datetime format

#extrating five months data 
start_date = '2020-01-21'
end_date = '2020-04-30'
mask = (df['ObservationDate'] > start_date) & (df['ObservationDate'] <= end_date)
covid_data = df.loc[mask]

#print("size/shape of the dataset:", covid_data.shape)
#print("checking for null values:\n", covid_data.isnull().sum())
#print("checking Data-type of each column:\n", covid_data.dtypes)

#dropping column as "Province/State" contains too many missing values
covid_data.drop(["Province/State"],1,inplace=True)

#grouping different types of cases as per the date
datewise = covid_data.groupby(["ObservationDate"]).agg({"Confirmed":"sum","Recovered":"sum","Deaths":"sum"})
#active_cases = datewise["Confirmed"].iloc[-1] - datewise["Recovered"].iloc[-1] - datewise["Deaths"].iloc[-1]

print("Basic Information")
print("Total number of countries with Disease Spread:", len(covid_data["Country/Region"].unique()))
print("Total number of Confirmed cases around the world", datewise["Confirmed"].iloc[-1])
print("Total number of Recovered cases around the world", datewise["Recovered"].iloc[-1])
print("Total number of Death cases around the world", datewise["Deaths"].iloc[-1])
print("Total number of Active cases around the world",(datewise["Confirmed"].iloc[-1] - datewise["Recovered"].iloc[-1] - datewise["Deaths"].iloc[-1]))
print("Total number of Closed cases around the world", (datewise["Recovered"].iloc[-1] + datewise["Deaths"].iloc[-1]))

plt.figure(figsize=(15,5))
sns.barplot(x=datewise.index.date, y=datewise["Confirmed"] - datewise["Recovered"] - datewise["Deaths"])
plt.title("Distribution plot for Active cases")
plt.xticks(rotation=90)

plt.figure(figsize=(15,5))
sns.barplot(x=datewise.index.date, y=datewise["Recovered"] + datewise["Deaths"])
plt.title("Distribution plot for Closed cases")
plt.xticks(rotation=90)

# fig=px.bar(x=datewise.index, y=datewise["Confirmed"]-datewise["Recovered"]-datewise["Deaths"])
# fig.update_layout(title="Distribution of Number of Active Cases",
#                     xaxis_title="Date",yaxis_title="Number of Cases",)
# fig

datewise["WeekofYear"] = datewise.index.weekofyear



