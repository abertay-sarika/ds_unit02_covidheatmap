import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from pandas.core.frame import DataFrame
import plotly.io as pio
from plotly.io import write_html
    
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
            covida_data
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

def plot_global_heatmap(file_name, covid_data, title):
    '''
    Function for generating heatmap
        Arg
           
        Returns 
            distribution plot(bar) : visualized data
    ''' 
    fig = px.choropleth(data_frame = covid_data,
                    locations= "iso_alpha",
                    color= "Confirmed",  # value in column 'Confirmed' determines color
                    hover_name= "Country",
                    color_continuous_scale= 'RdYlGn',  #  color scale red, yellow green
                    animation_frame= "ObservationDate")
    fig.update_layout(title_text = title)
    write_html(fig, file_name)

