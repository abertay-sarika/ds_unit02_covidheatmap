import os
     
    
def create_reports_dir():
    # this will create reports directory in the current working directory
    # the programm will be using this directory to save the results 
    current_directory = os.getcwd()
    reports_dir = os.path.join(current_directory, r'reports')
    if not os.path.exists(reports_dir):
        os.makedirs(reports_dir)
    return reports_dir    