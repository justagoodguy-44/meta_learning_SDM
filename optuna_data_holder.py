
class OptunaDataHolder:
    """Helper class for storing data used with optuna in the hyperparameter search"""

    def __init__(self, num_trials):
        self.num_trials = num_trials
        self.intervals = {}


    def add_new_interval(self, name, lower_val, higher_val):
        self.intervals[name] = (lower_val, higher_val)


    def get_interval(self, name):
        return self.intervals[name]
    

    def get_num_trials(self):
        return self.num_trials
