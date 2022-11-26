import matplotlib.pyplot as plt
import seaborn as sns

def bar_plot(x, y,figsize, title, rotation):
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
    plt.show()

def line_graph(week_num, data, linewidth=3):    
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
    for item in data:
        print(item['val'],  item['label'])
        # plt.plot(week_num, item['val'], linewidth, item['label'])
    
    # plt.xlabel("Week Number")
    # plt.ylabel("Number of cases")
    # plt.title("weekly progress of different types of caese")
    # plt.legend()
    # plt.show()
