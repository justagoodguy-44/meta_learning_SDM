import torch

class DataHolder:
    """
    Convenience class to hold data that needs to be passed around and moved from cpu to gpu
    """

    def __init__(self, x_train, y_train, bg_train, x_test, y_test, x_val=None, y_val=None, bg_val=None):
        self.data={
            "x_train": x_train,
            "y_train": y_train,
            "bg_train": bg_train,
            "x_test": x_test,
            "y_test": y_test,
            "x_val": x_val,
            "y_val": y_val,
            "bg_val": bg_val
        }
    
    
    def to_tensor(self, dtype, device=None):
        for key in self.data:
            val = self.data[key]
            if val is not None:
                self.data[key] = torch.tensor(val, dtype=dtype)
                if device is not None:
                    self.data[key] = self.data[key].to(device)


    def get(self, name:str):
        return self.data[name]