#importing requered libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import seaborn as sn
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error

fullcoviddata=pd.read_csv("/Users/sarikaraj/selfspace/mldata/covid_data.csv")
fullcoviddata.head()