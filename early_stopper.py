import copy

class EarlyStopper:
    """ Class to help handle early stopping during training"""

    def __init__(self, increasing:bool, patience:int):
        """_summary_

        Args:
            initial_score (float): The initial score for initialization, typically 0 or inf or -inf
            increasing (bool): _description_
            patience (int): _description_
        """        
        self.increasing = increasing
        self.patience = patience
        self.impatience = 0
        self.best_score = float("-inf") if increasing else float("inf")
        self.best_model = None


    def add_new_score(self, score:float, model):
        """_summary_
        Update with a new score
        Args:
            score (float): the score
            model (nn.Module): the model that generated this score 
        """
        if self.increasing:
            if score > self.best_score:
                self._update_with_better_score(score, model)
            else:
                self._update_with_worse_score()
        else:
            if score < self.best_score:
                self._update_with_better_score(score, model)
            else:
                self._update_with_worse_score()


    def should_stop(self):
        return self.impatience > self.patience
    

    def get_best_model(self):
        return self.best_model
    

    def _update_with_better_score(self, score, model):
            self.best_score = score
            self.impatience = 0
            self.best_model = copy.deepcopy(model)

    
    def _update_with_worse_score(self):
        self.impatience += 1