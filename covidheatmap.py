#importing requered libraries
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import seaborn as sns
import geopandas as gpd
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score,silhouette_samples
from sklearn.metrics import mean_squared_error
from geopy.geocoders import Nominatim
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
std=StandardScaler()

#reading the csv file
df = pd.read_csv("fullcovid_data.csv")
df["ObservationDate"] = pd.to_datetime(df["ObservationDate"]) #coverting "ObservationDate" in to Datetime format
df.rename(columns={"Country/Region":"Country"}, inplace=True)

#extrating five months data 
start_date = '2020-01-21'
end_date = '2020-04-30'
mask = (df["ObservationDate"] > start_date) & (df["ObservationDate"] <= end_date)
covid_data = df.loc[mask]

#dropping column as "Province/State" contains too many missing values
covid_data.drop(["Province/State"],1,inplace=True)
covid_data.__setitem__("longitude",0.0)
covid_data.__setitem__("latitude",0.0)
#print(covid_data.dtypes)

# print("size/shape of the dataset:", covid_data.shape)
# print("checking for null values:\n", covid_data.isnull().sum())
# print("checking Data-type of each column:\n", covid_data.dtypes)

#grouping different types of cases as per the date
datewise = covid_data.groupby(["ObservationDate"]).agg({"Confirmed":"sum","Recovered":"sum","Deaths":"sum"})
active_cases = datewise["Confirmed"].iloc[-1] - datewise["Recovered"].iloc[-1] - datewise["Deaths"].iloc[-1]
closed_cases = (datewise["Recovered"].iloc[-1] + datewise["Deaths"].iloc[-1])

# #heatmap
# # geolocator = Nominatim(timeout=10, user_agent="MyApp")
# # location=geolocator.geocode(["Country"])


geolocator = Nominatim(user_agent="myApp")

for i in covid_data.index:
    try:
        #tries fetch address from geopy
        location = geolocator.geocode(covid_data["Country"][i])
        
        #append lat/long to column using dataframe location
        covid_data.loc[i,'location_lat'] = location.latitude
        covid_data.loc[i,'location_long'] = location.longitude
    except:
        #catches exception for the case where no value is returned
        #appends null value to column
        covid_data.loc[i,'location_lat'] = ""
        covid_data.loc[i,'location_long'] = ""


#print first rows as sample
covid_data.head(5)


#basic information
# print("Basic Information")
# print("Total number of countries with Disease Spread:", len(covid_data["Country/Region"].unique()))
# print("Total number of Confirmed cases around the world", datewise["Confirmed"].iloc[-1])
# print("Total number of Recovered cases around the world", datewise["Recovered"].iloc[-1])
# print("Total number of Death cases around the world", datewise["Deaths"].iloc[-1])
# #print("Total number of Active cases around the world", active_cases)
# print("Total number of Closed cases around the world", closed_cases)



# plt.figure(figsize=(15,5))
# sns.barplot(x=datewise.index.date, y=datewise["Confirmed"] - datewise["Recovered"] - datewise["Deaths"])
# plt.title("Distribution plot for Active cases")
# plt.xticks(rotation=90)

# plt.figure(figsize=(15,5))
# sns.barplot(x=datewise.index.date, y=datewise["Recovered"] + datewise["Deaths"])
# plt.title("Distribution plot for Closed cases")
# plt.xticks(rotation=90)


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

# plt.figure(figsize=(8,5))
# plt.plot(week_num,weekwise_confirmed,linewidth=3, label="weekly growth of confirmed case")
# plt.plot(week_num,weekwise_recovered,linewidth=3, label="weekly growth of recovered case")
# plt.plot(week_num,weekwise_deaths,linewidth=3, label="weekly growth of death case")
# plt.xlabel("Week Number")
# plt.ylabel("Number of cases")
# plt.title("weekly progress of different types of caese")
# plt.legend()

# #f
# fig,(ax1,ax2) = plt.subplots(1,2,figsize=(12,4))
# sns.barplot(x=week_num,y=pd.Series(weekwise_confirmed).diff().fillna(0),ax=ax1)
# sns.barplot(x=week_num,y=pd.Series(weekwise_recovered).diff().fillna(0),ax=ax2)
# ax1.set_xlabel("Week Number")
# ax2.set_xlabel("Week Number")
# ax1.set_ylabel("Number of Confirmed cases")
# ax2.set_ylabel("Number of Recovered cases")
# ax1.set_title("Weekly increase in number of confirmed case")
# ax2.set_title("weekly increase in number of Death cases")
# plt.show()


# # #countriwise analysis n mortality rate
# countrywise = covid_data[covid_data["ObservationDate"]==covid_data["ObservationDate"].max()].groupby(["Country/Region"]).agg({"Confirmed":"sum","Recovered":"sum","Deaths":"sum"}).sort_values(["Confirmed"],ascending=False)
# countrywise["Mortality Rate"]=(countrywise["Deaths"]/countrywise["Confirmed"])*100
# countrywise["Recovery Rate"]=(countrywise["Recovered"]/countrywise["Confirmed"])*100
# print (countrywise["Mortality Rate"].mean())
# print(countrywise["Recovery Rate"].mean())

# fig,(ax1,ax2)=plt.subplots(1,2,figsize=(25,10))
# top_15confirmed = countrywise.sort_values(["Confirmed"],ascending=False).head(15)
# top_15deaths = countrywise.sort_values(["Deaths"],ascending=False).head(15)
# sns.barplot(x=top_15confirmed["Confirmed"],y=top_15confirmed.index,ax=ax1)
# ax1.set_title("Top 15 countries as per number of confirmed cases")
# sns.barplot(x=top_15deaths["Deaths"],y=top_15confirmed.index,ax=ax2)
# ax2.set_title("Top 15 countries as per number of death cases")


#prediction n forcasting

#splitting data to features and target variables seperately
# datewise['Days Since'] = datewise.index - datewise.index[0]
# datewise['Days Since'] = datewise['Days Since'].dt.days
# datewise


# #1
# train_ml=datewise.iloc[:int(datewise.shape[0]*0.95)]
# valid_ml=datewise.iloc[int(datewise.shape[0]*0.95):]
# #Intializing SVR Model
# svm=SVR(C=1,degree=5,kernel='poly',epsilon=0.01)
# SVR(C=1, cache_size=200, coef0=0.0, degree=5, epsilon=0.01, gamma='scale',
#     kernel='poly', max_iter=-1, shrinking=True, tol=0.001, verbose=False)
# prediction_valid_svm=svm.predict(np.array(valid_ml["Days Since"]).reshape(-1,1))
# model_scores.append(np.sqrt(mean_squared_error(valid_ml["Confirmed"],prediction_valid_svm)))
# print("Root Mean Square Error for Support Vectore Machine: ",np.sqrt(mean_squared_error(valid_ml["Confirmed"],prediction_valid_svm)))

# # Split dataset into training set and test set
# X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, test_size=0.3,random_state=109) # 70% training and 30% test

# #Create a svm Classifier
# clf = svm.SVC(kernel='linear') # Linear Kernel
# clf.fit(X_train, y_train) #Train the model using the training sets
# y_pred = clf.predict(X_test) #Predict the response for test dataset
# print("Accuracy:",metrics.accuracy_score(y_test, y_pred)) # Model Accuracy: how often is the classifier correct?




# #data analysis for uk
# uk_data = covid_data[covid_data["Country/Region"]=="UK"]
# datewise_uk = uk_data.groupby(["ObservationDate"]).agg({"Confirmed":"sum","Recovered":"sum","Deaths":"sum"})
# print(datewise_uk.iloc[-1])
# print("Total Active Cases",datewise_uk["Confirmed"].iloc[-1]- datewise_uk["Recovered"].iloc[-1] - datewise_uk["Deaths"].iloc[-1])
# print("Total Closed Cases",datewise_uk["Recovered"].iloc[-1] + datewise_uk["Deaths"].iloc[-1])

# #reported cases by week
# datewise_uk["WeekofYear"] = datewise_uk.index.weekofyear
# week_num_uk = []
# uk_weekwise_confirmed = []
# uk_weekwise_recovered = []
# uk_weekwise_deaths = []
# w = 1
# for i in list(datewise_uk["WeekofYear"].unique()):
#     uk_weekwise_confirmed.append(datewise_uk[datewise_uk["WeekofYear"]==i]["Confirmed"].iloc[-1])
#     uk_weekwise_recovered.append(datewise_uk[datewise_uk["WeekofYear"]==i]["Recovered"].iloc[-1])
#     uk_weekwise_deaths.append(datewise_uk[datewise_uk["WeekofYear"]==i]["Deaths"].iloc[-1])
#     week_num_uk.append(w)
#     w=w+1

# plt.figure(figsize=(8,5))
# plt.plot(week_num_uk,uk_weekwise_confirmed,linewidth=3, label="weekly growth of confirmed case")
# plt.plot(week_num_uk,uk_weekwise_recovered,linewidth=3, label="weekly growth of recovered case")
# plt.plot(week_num_uk,uk_weekwise_deaths,linewidth=3, label="weekly growth of death case")
# plt.xlabel("Week Number")
# plt.ylabel("Number of cases")
# plt.title("weekly progress of different types of caese in UK")
# plt.legend()





