import numpy as np

class neuralNetwork:
    # Create and initialise the neural network (constructor).
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Set number of nodes in the input, hidden, and output layer.
        self.inodes = input_nodes
        self.hnodes = hidden_nodes
        self.onodes = output_nodes
        
        # Create the weight matrices (wih and who).
        # wih is the weights for the links between nodes in the input layer and the hidden layer.
        # who is the weights for the links between nodes in the hidden layer and the output layer.
        self.wih = np.random.rand(self.hnodes, self.inodes)-0.5
        self.who = np.random.rand(self.onodes, self.hnodes)-0.5

        # Set the learning rate.
        self.lr = learning_rate
        
        # Define the activation function (sigmoid function)
        def sigmoid(x):
            sig = 1 / (1 + np.exp(-x))
            return sig
        
        # Set the activiation function (sigmoid function)
        self.activation_function = lambda x: sigmoid(x)

    # Train the neural network.
    def train(self, inputs_list, targets_list):
        # Convert a list into an array (e.g. 1 x 3 then apply transpose to make it 3 x 1).
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T
        
        # Calculate signals into hidden layer.
        hidden_inputs = np.dot(self.wih, inputs)
        # Calculate the output signals from the hidden layer.
        hidden_outputs = self.activation_function(hidden_inputs)
        
        # Calculate signals into final output layer.
        final_inputs = np.dot(self.who, hidden_outputs)
        # Calculate the output signals from final output layer.
        final_outputs = self.activation_function(final_inputs)
        
        # Calcualte output layer error (target - actual).
        output_errors = targets - final_outputs
        # Calculate hidden layer error based on output layer error (split by weights and recombined at hidden nodes).
        hidden_errors = np.dot(self.who.T, output_errors) 
        
        # Update the weights for the links between the hidden layer and the output layer.
        self.who += self.lr * np.dot((output_errors * final_outputs * (1.0 - final_outputs)), np.transpose(hidden_outputs))
        
        # update the weights for the links between the input layer and the hidden layer.
        self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), np.transpose(inputs))
              
    # Query (test) the neural network.
    def query(self, inputs_list):
        # Convert a list into an array (e.g. 1 x 3 then apply transpose to make it 3 x 1).
        inputs = np.array(inputs_list, ndmin=2).T
        
        # Calculate signals into hidden layer.
        hidden_inputs = np.dot(self.wih, inputs)
        # Calculate the output signals from the hidden layer.
        hidden_outputs = self.activation_function(hidden_inputs)
        
        # Calculate signals into final output layer.
        final_inputs = np.dot(self.who, hidden_outputs)
        # Calculate the output signals from final output layer.
        final_outputs = self.activation_function(final_inputs)
        
        return final_outputs