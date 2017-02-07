
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class NeuralNetwork(object):
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Set number of nodes in input, hidden and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Initialize weights
        self.weights_input_to_hidden = np.random.normal(0.0, self.hidden_nodes ** -0.5,
                                                        (self.hidden_nodes, self.input_nodes))

        self.weights_hidden_to_output = np.random.normal(0.0, self.output_nodes ** -0.5,
                                                         (self.output_nodes, self.hidden_nodes))
        self.lr = learning_rate

        #### Set this to your implemented sigmoid function ####
        # Activation function is the sigmoid function
        self.activation_function = self.sigmoid



    def train(self, inputs_list, targets_list):
        # Convert inputs list to 2d array
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T
        #### Implement the forward pass here ####
        ### Forward pass ###
        # TODO: Hidden layer

        hidden_inputs = np.dot(self.weights_input_to_hidden,inputs) # signals into hidden layer
        hidden_outputs = self.activation_function(hidden_inputs) # signals from hidden layer

        # TODO: Output layer
        final_inputs = np.dot(self.weights_hidden_to_output,hidden_outputs) # signals into final output layer
        final_outputs = final_inputs # signals from final output layer

        #### Implement the backward pass here ####
        ### Backward pass ###

        # TODO: Output error
        output_errors = targets - final_outputs # Output layer error is the difference between desired target and actual output.

        # TODO: Backpropagated error
        hidden_errors = np.dot(self.weights_hidden_to_output.T, output_errors) # errors propagated to the hidden layer
        hidden_grad = hidden_errors * hidden_outputs * (1 - hidden_outputs)  # hidden layer gradients

        # TODO: Update the weights
        self.weights_input_to_hidden += self.lr * np.dot(hidden_grad , inputs.T)
        self.weights_hidden_to_output += self.lr * output_errors*hidden_outputs.T



    def run(self, inputs_list):
        #Run a forward pass through the network
        inputs = np.array(inputs_list, ndmin=2)

        #### Implement the forward pass here ####
        # TODO: Hidden layer
        hidden_inputs = np.dot(inputs,self.weights_input_to_hidden.T) # signals into hidden layer
        hidden_outputs = self.activation_function(hidden_inputs) # signals from hidden layer

        # TODO: Output layer
        final_inputs = np.dot(hidden_outputs,self.weights_hidden_to_output.T) # signals into final output layer
        final_outputs = final_inputs # signals from final output layer

        return final_outputs
