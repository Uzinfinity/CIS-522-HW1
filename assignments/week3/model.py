import torch
from typing import Callable


class MLP(torch.nn.Module):
    """
    initialization of multi layer perceptrons
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_classes: int,
        hidden_count: int = 1,
        activation: Callable = torch.nn.ReLU,
        initializer: Callable = torch.nn.init.ones_,
    ) -> None:
        """
        Initialize the MLP.

        Arguments:
            input_size: The dimension D of the input data.
            hidden_size: The number of neurons H in the hidden layer.
            num_classes: The number of classes C.
            activation: The activation function to use in the hidden layer.
            initializer: The initializer to use for the weights.
        """

        super().__init__()

        self.hidden_count = hidden_count
        self.hidden_layers = torch.nn.ModuleList()
        self.activation = activation
        self.output_layer = torch.nn.Linear(hidden_size, num_classes)
        self.initializer = initializer

        for i in range(hidden_count):
            self.hidden_layers.append(torch.nn.Linear(input_size, hidden_size))
            input_size = hidden_size

        self.initializer(self.output_layer.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network.

        Arguments:
            x: The input data.

        Returns:
            The output of the network.
        """
        for hidden_layer in self.hidden_layers:
            x = self.activation()(hidden_layer(x))
        return self.output_layer(x)
