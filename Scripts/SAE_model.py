import torch
import torch.nn as nn
import torch.optim as optim

class StackedAutoencoder(nn.Module):
    def __init__(self, 
                 input_size=784, 
                 layer_sizes=[800, 200, 50], 
                 activation_functions=[nn.ReLU(), nn.ReLU(), nn.ReLU()], 
                 dropout_rates=[0.0, 0.0, 0.0],
                 weight_init=None):
        super(StackedAutoencoder, self).__init__()
        
        assert len(layer_sizes) >= 1, "There must be at least one hidden layer."
        assert len(activation_functions) == len(layer_sizes), "Activation functions list must match layer sizes."
        assert len(dropout_rates) == len(layer_sizes), "Dropout rates list must match layer sizes."
        
        # Encoder
        encoder_layers = []
        prev_size = input_size
        for idx, (size, activation_fn, dropout_rate) in enumerate(zip(layer_sizes, activation_functions, dropout_rates)):
            encoder_layers.append(nn.Linear(prev_size, size))
            if activation_fn:
                encoder_layers.append(activation_fn)
            if dropout_rate > 0.0:
                encoder_layers.append(nn.Dropout(dropout_rate))
            prev_size = size
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder
        decoder_layers = []
        reversed_layer_sizes = list(reversed(layer_sizes[:-1])) + [input_size]
        for idx, (size, activation_fn, dropout_rate) in enumerate(
            zip(
                reversed_layer_sizes,
                reversed(activation_functions[:-1] + [nn.Sigmoid()]),
                reversed(dropout_rates[:-1] + [0.0])
            )
        ):
            decoder_layers.append(nn.Linear(prev_size, size))
            if activation_fn:
                decoder_layers.append(activation_fn)
            if dropout_rate > 0.0:
                decoder_layers.append(nn.Dropout(dropout_rate))
            prev_size = size
        self.decoder = nn.Sequential(*decoder_layers)
        
        # Weight Initialization
        if weight_init:
            self.apply(weight_init)
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def get_encoded_features(self, x):
        return self.encoder(x)
