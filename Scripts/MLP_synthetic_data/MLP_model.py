import torch
import torch.nn as nn

class MLPClassifier(nn.Module):
    def __init__(self, 
                 input_size=784, 
                 layer_sizes=[256, 128, 64], 
                 output_size=10,
                 activation_functions=None, 
                 dropout_rates=None,
                 batch_norm=None,
                 weight_init=None):
        super(MLPClassifier, self).__init__()

        assert len(layer_sizes) >= 1, "There must be at least one hidden layer."

        # Default activation functions to ReLU if not provided
        if activation_functions is None:
            activation_functions = [nn.ReLU() for _ in layer_sizes]
        else:
            assert len(activation_functions) == len(layer_sizes), "Activation functions list must match layer sizes."

        # Default dropout rates to 0.0 if not provided
        if dropout_rates is None:
            dropout_rates = [0.0 for _ in layer_sizes]
        else:
            assert len(dropout_rates) == len(layer_sizes), "Dropout rates list must match layer sizes."

        # Default batch normalization to False if not provided
        if batch_norm is None:
            batch_norm = [False for _ in layer_sizes]
        else:
            assert len(batch_norm) == len(layer_sizes), "Batch norm list must match layer sizes."

        layers = []
        prev_size = input_size

        for idx, (size, activation_fn, dropout_rate, bn) in enumerate(zip(layer_sizes, activation_functions, dropout_rates, batch_norm)):
            layers.append(nn.Linear(prev_size, size))
            if bn:
                layers.append(nn.BatchNorm1d(size))
            if activation_fn:
                layers.append(activation_fn)
            if dropout_rate > 0.0:
                layers.append(nn.Dropout(dropout_rate))
            prev_size = size

        # Output layer
        layers.append(nn.Linear(prev_size, output_size))
        # No activation function here; use CrossEntropyLoss which includes LogSoftmax

        self.model = nn.Sequential(*layers)

        # Apply weight initialization if provided
        if weight_init:
            self.apply(weight_init)

    def forward(self, x):
        return self.model(x)
