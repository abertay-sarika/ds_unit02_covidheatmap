import os


class Output:
    def __init__(self):
        self.confirmed_cases = 0
        self.recovered_cases = 0
        self.death_count = 0
        self.active_cases = 0
        self.closed_cases = 0



    def get_confirmed_cases(self):
        return self.confirmed_cases
    def get_recovered_cases(self):
        return self.recovered_cases
    def get_death_count(self):
        return self.death_count
    def get_active_cases(self):
        return self.active_cases
    def get_closed_cases(self):
        return self.closed_cases  

    def set_confirmed_cases(self, confirmed):
        self.confirmed_cases = confirmed
    def set_recovered_cases(self, recovered):
        self.recovered_cases = recovered
    def set_death_count(self, deaths):
        self.death_count = deaths
    def set_active_cases(self,active_cases):
        self.active_cases = active_cases
    def set_closed_cases(self, closed_cases):
        self.closed_cases = closed_cases      
    
def create_reports_dir():
    # this will create reports directory in the current working directory
    # the programm will be using this directory to save the results 
    current_directory = os.getcwd()
    reports_dir = os.path.join(current_directory, r'reports')
    if not os.path.exists(reports_dir):
        os.makedirs(reports_dir)
    return reports_dir    