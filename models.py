from torch import nn

class MLP(nn.Module):
    
    def __init__(self, input_size, output_size, num_layers, width, dropout=0.0):
            super(MLP, self).__init__()

            self.output_size = output_size
            self.width = width

            layers = []
            # Store the indexes of dropout layers to be able to change them subsequently
            self.dropout_layers_idx = []

            linear = nn.Linear(input_size, width)
            layers.append(linear)
            layers.append(nn.ReLU())

            for _ in range(num_layers - 1):
                layers.append(nn.BatchNorm1d(width))
                linear = nn.Linear(width, width)
                layers.append(linear)
                layers.append(nn.ReLU())
                self.dropout_layers_idx.append(len(layers))
                layers.append(nn.Dropout(p=dropout))
        
            linear = nn.Linear(width, output_size)
            layers.append(linear)

            self.layers = nn.Sequential(*layers)


    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


    def change_output_size(self, output_size:int):
         new_layers = self.layers[:-1]
         new_layers.append(nn.Linear(self.width, output_size))
         self.layers = new_layers

    
    def freeze_all_but_last_layer(self):
        params = self.parameters()
        param_list = list(params)
        all_except_last_layer_params = param_list[:-2]

        for param in all_except_last_layer_params:
            param.requires_grad = False


    def change_dropout(self, dropout:float):
        for layer_idx in self.dropout_layers_idx:
            self.layers[layer_idx] = nn.Dropout(p=dropout)