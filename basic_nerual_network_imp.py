import numpy as np
#Save activations and derivatives
#implement backpropogation
#Implement gredint decent
#implement train
#train our net with some dummy data
#make some prediction

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
        #print(layers)
        # create random connection weights for the layers
        weights = []
        for i in range(len(layers)-1):
            w = np.random.rand(layers[i], layers[i + 1])
            weights.append(w)
        self.weights = weights


        activations=[]
        for i in range(len(layers)):
          a=np.zeros(layers[i])
          activations.append(a)

        self.activations=activations

        derivatives=[]
        for i in range(len(layers) - 1):
            d = np.zeros((layers[i], layers[i + 1]))
            derivatives.append(d)
        self.derivatives = derivatives

    def forward_propogate(self,inputs):
          # the input layer activation is just the input itself
          activation = inputs

          # save the activations for backpropogation
          self.activations[0] = activation
          for i,w in enumerate(self.weights):
            net_inputs=np.dot(activation,w)
            #update
            activation=self.sigmoid(net_inputs)
            # save the activations for backpropogation
            self.activations[i+1]=activation
          return activation
    def back_backpropogate(self,error,flag=False):

      """Backpropogates an error signal.
        Args:
            error (ndarray): The error to backprop.
        Returns:
            error (ndarray): The final error of the input
        """
         # iterate backwards through the network layers
         #dw_i=(a[i+1]-y)*s'(h[i+1])*a[i]
         #s'(h[i+1])->1/1+np.exp(-h[i+1])=  s(h[i+1])*(i-s(h[i+1]))
         #s(h[i+1])=a[i+1]

         #dw_[i-1]=(a[i+1]-y)*s'(h[i+1])*w[i]*s'(h[i])*a[i-1]
      for i in reversed(range(len(self.derivatives))):

        # get activation for previous layer
        activation=self.activations[i+1]
        #get the derrivative of activation
        delta=error * self.sigmoid_derivative(activation)
  
        delta_reshaped=delta.reshape(delta.shape[0],-1).T
        current_derevatives=self.activations[i] #activation is 2d so current is 1d so make current 2d using reshape [1]-->[[1]]
        current_derevatives_reshaped=current_derevatives.reshape(current_derevatives.shape[0],-1)
        #cal base derivative
        self.derivatives[i]=np.dot(current_derevatives_reshaped,delta_reshaped)
        error=np.dot(delta,self.weights[i].T)

        if(flag):
          print("Derivative for W{}:{}".format(i,self.derivatives[i]))
      return error


    def sigmoid_derivative(self,x):
      return x* (1.0 - x)
    def sigmoid(self,x):
          y=1/(1+np.exp(-x))
          return y
