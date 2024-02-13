import numpy as np


class MLP(object):

    """A Multilayer Perceptron class.
    """

    def __init__(self, num_inputs=3, hidden_layers=[3, 5], num_outputs=2):
        """Constructor for the MLP. Takes the number of inputs,
            a variable number of hidden layers, and number of outputs

        Args:
            num_inputs (int): Number of inputs
            hidden_layers (list): A list of ints for the hidden layers
            num_outputs (int): Number of outputs
        """

        self.num_inputs = num_inputs
        self.hidden_layers = hidden_layers
        self.num_outputs = num_outputs

        # create a generic representation of the layers
        layers = [num_inputs] + hidden_layers + [num_outputs]
        print(layers)
        # create random connection weights for the layers
        weights = []
        for i in range(len(layers)-1):
            w = np.random.rand(layers[i], layers[i+1])
            print(w)
            weights.append(w)
        self.weights = weights

    def forward_propogate(self,inputs):
          activation=inputs
          
          for x in self.weights:
            
            net_inputs=np.dot(activation,x)
            
            activation=self.sigmoid(net_inputs)

          return activation

    def sigmoid(self,x):
          y=1/(1+np.exp(-x))
          return y


mlp=MLP()
inputs=np.random.rand(mlp.num_inputs)
output=mlp.forward_propogate(inputs)
print("input for nural network:{}".format(inputs))
print("output for nural network:{}".format(output))