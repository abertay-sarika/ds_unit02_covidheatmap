import pandas as pd
import numpy as np      
from datetime import timedelta
import datetime as dt
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn import metrics
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.api import Holt,SimpleExpSmoothing,ExponentialSmoothing
from sklearn.tree import DecisionTreeRegressor
from pandas.core.frame import DataFrame

html = """
    <!DOCTYPE html>
    <html>
    <body>
    <div id="header" style="height:15%;width:100%;">
    <div style='float:left'>
    <table border="1" width="44" style="margin-left:10%;float:top;">
    <tr>
    """

def prep_for_prediction(datewise : DataFrame):
    datewise['Days Since'] = datewise.index - datewise.index[0]
    datewise['Days Since'] = datewise['Days Since'].dt.days
    
    train=datewise.iloc[:int(datewise.shape[0]*0.95)]
    test=datewise.iloc[int(datewise.shape[0]*0.95):]

    x_train =train["Days Since"]
    y_train =train["Confirmed"]
    x_test = test["Days Since"]
    y_test = test["Confirmed"]

    return (x_train, y_train, x_test, y_test)

def decision_tree(datewise: DataFrame):
    strTable = html + "<th>R Squared Value</th><th>Mean Squared log error</th></tr>"

    (x_train, y_train, x_test, y_test) = prep_for_prediction(datewise=datewise)
    tree = DecisionTreeRegressor(max_depth = 3)
    tree.fit(np.array(x_train).reshape(-1,1),np.array(y_train).reshape(-1,1))
    y_pred_train = tree.predict(np.array(x_train).reshape(-1,1))
    y_pred_test = tree.predict(np.array(x_test).reshape(-1,1))
    prediction_tree=tree.predict(np.array(datewise["Days Since"]).reshape(-1,1))
    print("r squared score for decision tree: ", metrics.r2_score(y_train, prediction_tree[:int(datewise.shape[0]*0.95)]))
    print("Mean Squared log error for decision tree: ", metrics.mean_squared_log_error(y_train, prediction_tree[:int(datewise.shape[0]*0.95)]))
    strRW = "<tr><td>"+str(metrics.r2_score(y_train, prediction_tree[:int(datewise.shape[0]*0.95)]))+ "</td><td>"+str(metrics.mean_squared_log_error(y_train, prediction_tree[:int(datewise.shape[0]*0.95)]))+"</td></tr>"
    strTable = strTable+strRW
    strTable = strTable+"</table></div>"

    plt.figure(figsize=(11,6))
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=datewise.index, y=datewise["Confirmed"],
                        mode='lines+markers',name="Train Data for Confirmed Cases"))
    fig.add_trace(go.Scatter(x=datewise.index, y=prediction_tree,
                        mode='lines',name="Decision Tree Best fit Kernal",
                        line=dict(color='black', dash='dot')))
    fig.update_layout(title="Confirmed Cases Decision Tree Regression",
                    xaxis_title="Date",yaxis_title="Confirmed Cases",legend=dict(x=0,y=1,traceorder="normal"))
    fig.write_image('reports/pred1.png')

    strTable = strTable+"<div style='float:leftt'>"
    image = "<img src=\"pred1.png\" style=\"margin-left:10%;margin-top:0%\">"
    strTable = strTable+image
    strTable = strTable+"</div></div></body></html>"
    
    hs = open("reports/prediction_decision_tree.html", 'w')
    hs.write(strTable)

def svm(datewise: DataFrame):
    strTable = html + "<th>Root Mean Square Error</th></tr>"
    (x_train, y_train, x_test, y_test) = prep_for_prediction(datewise=datewise)

    svm=SVR(C=1,degree=5,kernel='poly',epsilon=0.01)
    svm.fit(np.array(x_train).reshape(-1,1),np.array(y_train).reshape(-1,1))
    prediction_svm=svm.predict(np.array(x_test).reshape(-1,1))
    # rmse_svm=np.sqrt(mean_squared_error(y_train, prediction_svm[:int(datewise.shape[0]*0.95)]))
    # print("Root Mean Square Error for SVR Model: ",rmse_svm)
    print("Root Mean Square Error for Support Vectore Machine: ",np.sqrt(mean_squared_error(y_test, prediction_svm)))
    strRW = "<tr><td>"+str(np.sqrt(mean_squared_error(y_test, prediction_svm)))+ "</td></tr>"
    strTable = strTable+strRW
    strTable = strTable+"</table></div>"

    new_date = []
    new_predict_svm = []
    for i in range(1,18):
        new_date.append(datewise.index[-1]+timedelta(days=i))
        new_predict_svm.append(svm.predict(np.array(datewise["Days Since"].max()+i).reshape(-1,1))[0])
    pd.set_option("display.float_format",lambda x: '%.f' % x)
    model_predict = pd.DataFrame(zip(new_date,new_predict_svm),columns = ["Dates","SVR"])
    ytest_pred =svm.predict(np.array(x_test).reshape(-1,1))
    model_predict.head()

    plt.figure(figsize=(11,6))
    prediction_svm=svm.predict(np.array(datewise["Days Since"]).reshape(-1,1))
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=datewise.index, y=datewise["Confirmed"],
                        mode='lines+markers',name="Train Data for Confirmed Cases"))
    fig.add_trace(go.Scatter(x=datewise.index, y=prediction_svm,
                        mode='lines',name="Support Vector Machine Best fit Kernel",
                        line=dict(color='black', dash='dot')))
    fig.update_layout(title="Confirmed Cases Support Vectore Machine Regressor Prediction",
                    xaxis_title="Date",yaxis_title="Confirmed Cases",legend=dict(x=0,y=1,traceorder="normal"))

    fig.write_image('reports/pred2.png')
    strTable = strTable+"<div style='float:leftt'>"
    image = "<img src=\"pred2.png\" style=\"margin-left:5%;margin-top:0%\">"
    strTable = strTable+image
    strTable = strTable+"</div></div></body></html>"
    
    hs = open("reports/prediction_svm.html", 'w')
    hs.write(strTable)

def test():
    html = """
    <!DOCTYPE html>
    <html>
    <body>
    <div id="header" style="height:15%;width:100%;">
    <div style='float:left'>
            <table border="1" width="44" style="margin-left:30%;float:top;">
                <tr>
    """
    strTable = html + "<th>Char</th><th>ASCII</th></tr>"

    for num in range(33, 35):
        symb = chr(num)
        strRW = "<tr><td>"+str(symb) + "</td><td>"+str(num)+"</td></tr>"
        strTable = strTable+strRW

    strTable = strTable+"</table></div>"

    strTable = strTable+"<div style='float:leftt'>"
    image = "<img src=\"African_Bush_Elephant.jpg\" style=\"margin-left:5%;margin-top:0%\">"
    strTable = strTable+image
    strTable = strTable+"</div></div></body></html>"

    hs = open("asciiCharHTMLTable.html", 'w')
    hs.write(strTable)

    print(strTable)


  

# def holt(datewise: DataFrame):
#     strTable = "<html><table><tr><th>Root Mean Square</th><th>mean_squared_log_error</th></tr>"
#     train=datewise.iloc[:int(datewise.shape[0]*0.95)]
#     test=datewise.iloc[int(datewise.shape[0]*0.95):]
#     (train, test) = prep_for_prediction(datewise=datewise)

#     model_scores = []
#     y_pred=test.copy()
#     holt=Holt(np.asarray(train["Confirmed"])).fit(smoothing_level=0.3, smoothing_slope=1.2)
#     y_pred["Holt"]=holt.forecast(len(test))
#     rmse_holt_linear=np.sqrt(mean_squared_error(y_pred["Confirmed"],y_pred["Holt"]))
#     model_scores.append(rmse_holt_linear)
#     print("Root Mean Square Error Holt's Linear Model: ",rmse_holt_linear)

#     fig=go.Figure()
#     fig.add_trace(go.Scatter(x=train.index, y=train["Confirmed"],
#                         mode='lines+markers',name="Train Data for Confirmed Cases"))
#     fig.add_trace(go.Scatter(x=test.index, y=test["Confirmed"],
#                         mode='lines+markers',name="Validation Data for Confirmed Cases",))
#     fig.add_trace(go.Scatter(x=test.index, y=y_pred["Holt"],
#                         mode='lines+markers',name="Prediction of Confirmed Cases",))
#     fig.update_layout(title="Confirmed Cases Holt's Linear Model Prediction",
#                     xaxis_title="Date",yaxis_title="Confirmed Cases",legend=dict(x=0,y=1,traceorder="normal"))

#     fig.write_image('reports/forecast.png')

#     image = '<img src="forecast.png">'
#     strTable= strTable+image
#     strTable = strTable+"</table></html>"
    
#     hs = open("reports/holt_forecast.html", 'w')
#     hs.write(strTable)
    
