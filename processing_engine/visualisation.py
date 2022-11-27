import matplotlib.pyplot as plt
import seaborn as sns

def bar_plot(file_name, x, y,figsize, title, rotation):
    '''
    Function for plotting barchart
        Arg
            covid_data(DataFrame) : extracted_data covid data
            x,y(position) : label
            figsize(tuple) : width and height of the graph
            title(str) : title for bar chart 
        Returns 
            distribution plot(bar) : visualized data
    '''
    plt.figure(figsize=figsize)
    sns.barplot(x=x,y=y)
    plt.title(title)
    plt.xticks(rotation = rotation)
    plt.savefig(file_name)

def line_graph(file_name, week_num, data):
    '''
    Function for plotting line graph
        Arg
            wwek_num(int) : week number
            weekwise_confirmed() : weekly confirmed cases count
            weekwise_recovered() : weekly recovered cases count
            weekwise_deaths() : weekly deaths count
        Returns
             distribution plot(line) : visualized data
    '''
    plt.figure(figsize=(10,5))
    for (key,item) in data.items():
        plt.plot(week_num, item['val'], linewidth=3, label=item["label"])

    plt.xlabel("Week Number")
    plt.ylabel("Number of cases")
    plt.title("weekly progress of different types of caese")
    plt.legend()
    plt.savefig(file_name)

def bar_plot_weekwise(file_name, subplot_title1, subplot_title2, figsize, x1=None, x2=None, y1=None, y2=None, x_label=None, y_label1=None, y_label2=None):
    '''
    Function for plotting two bar plots
        Arg
            x,y(position) : label
            (label) : subplots label
            title(str) : subplots title
        Returns 
            distribution plot(bar) : visualized data
    '''
    _fig, (ax1,ax2) = plt.subplots(1,2,figsize=figsize)
    sns.barplot(x=x1,y=y1,ax=ax1)
    sns.barplot(x=x2,y=y2,ax=ax2)
    if x_label is not None: 
        ax1.set_xlabel(x_label)
        ax2.set_xlabel(x_label)
    if y_label1 is not None:    
        ax1.set_ylabel(y_label1)
    if y_label2 is not None:    
        ax2.set_ylabel(y_label2)
    ax1.set_title(subplot_title1)
    ax2.set_title(subplot_title2)
    plt.savefig(file_name)

# def top20_countrywise(covid_data, countrywise_x1_axis,countrywise_y1_axis, Countrywise_title1, countrywise_x2_axis, countrywise_y2_axis, Countrywise_title2):
#      '''
#     Function for plotting line graph
#         Arg
            
#         Returns 
#              distribution plot(line) : visualized data
#     '''
#     countrywise = covid_data[covid_data["ObservationDate"]==covid_data["ObservationDate"].max()].groupby(["Country"]).agg(
#         {"Confirmed":"sum","Recovered":"sum","Deaths":"sum"}).sort_values(["Confirmed"],ascending=False)
#     fig,(ax1,ax2)=plt.subplots(1,2,figsize=(25,10))
#     top_20confirmed = countrywise.sort_values(["Confirmed"],ascending=False).head(20)
#     top_20deaths = countrywise.sort_values(["Deaths"],ascending=False).head(20)
#     sns.barplot(x=countrywise_x1_axis,y=countrywise_y1_axis,ax=ax1)
#     ax1.set_title(Countrywise_title1)
#     sns.barplot(x=countrywise_x2_axis,y=countrywise_y2_axis,ax=ax2)
#     ax2.set_title(Countrywise_title2) 
#     plt.show()

# #def uk_plot(x, y, z, x_label, y_label, uk_title):
#     '''
#     Function for plotting line graph
#         Arg
            
#         Returns 
#              distribution plot(line) : visualized data
#     '''
#     plt.plot(x,label="Daily increase in confirmed cases")
#     plt.plot(y,label="Daily increase in recovered cases")
#     plt.z,label="Daily increase in deaths"
#     plt.xlabel(x_label)
#     plt.ylabel(y_label)
#     plt.title(uk_title)
#     plt.xticks(rotation = 90)
#     plt.legend()


