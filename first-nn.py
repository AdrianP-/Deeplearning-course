
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data_path = '/home/aportabales/udacity/DLND-your-first-network/Bike-Sharing-Dataset/hour.csv'

rides = pd.read_csv(data_path)

rides.head()

rides[:24*10].plot(x='dteday', y='cnt')

dummy_fields = ['season', 'weathersit', 'mnth', 'hr', 'weekday']
for each in dummy_fields:
    dummies = pd.get_dummies(rides[each], prefix=each, drop_first=False)
    rides = pd.concat([rides, dummies], axis=1)

fields_to_drop = ['instant', 'dteday', 'season', 'weathersit',
                  'weekday', 'atemp', 'mnth', 'workingday', 'hr']
data = rides.drop(fields_to_drop, axis=1)
data.head()

quant_features = ['casual', 'registered', 'cnt', 'temp', 'hum', 'windspeed']
# Store scalings in a dictionary so we can convert back later
scaled_features = {}
for each in quant_features:
    mean, std = data[each].mean(), data[each].std()
    scaled_features[each] = [mean, std]
    data.loc[:, each] = (data[each] - mean)/std


# Save the last 21 days
test_data = data[-21*24:]
data = data[:-21*24]

# Separate the data into features and targets
target_fields = ['cnt', 'casual', 'registered']
features, targets = data.drop(target_fields, axis=1), data[target_fields]
test_features, test_targets = test_data.drop(target_fields, axis=1), test_data[target_fields]

# Hold out the last 60 days of the remaining data as a validation set
train_features, train_targets = features[:-60*24], targets[:-60*24]
val_features, val_targets = features[-60*24:], targets[-60*24:]


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
        for x, y in zip(inputs,targets):
            hidden_inputs = np.dot(x,self.weights_input_hidden)
            hidden_outputs = self.activation_function(hidden_inputs)

            # TODO: Output layer
            final_inputs =  np.dot(hidden_outputs,self.weights_hidden_to_output)
            final_outputs = final_inputs

            #### Implement the backward pass here ####
            ### Backward pass ###

            # TODO: Output error
            output_errors = y - final_outputs[:,None] # Output layer error is the difference between desired target and actual output.

            # TODO: Backpropagated error
            hidden_errors = np.dot(output_errors, self.weights_hidden_to_output) * hidden_outputs * (1 - hidden_outputs) # errors propagated to the hidden layer
            hidden_grad =  self.activation_function(hidden_errors) # hidden layer gradients

            # TODO: Update the weights
            self.weights_hidden_to_output += output_errors * hidden_errors # update hidden-to-output weights with gradient descent step
            self.weights_input_to_hidden +=  hidden_grad * x[:, None] # update input-to-hidden weights with gradient descent step

    def run(self, inputs_list):
        # Run a forward pass through the network
        # inputs = np.array(inputs_list, ndmin=2).T
        #
        # #### Implement the forward pass here ####
        # # TODO: Hidden layer
        # hidden_inputs =  # signals into hidden layer
        # hidden_outputs =  # signals from hidden layer
        #
        # # TODO: Output layer
        # final_inputs =  # signals into final output layer
        # final_outputs =  # signals from final output layer

        # return final_outputs
        return None




import unittest

inputs = [0.5, -0.2, 0.1]
targets = [0.4]
test_w_i_h = np.array([[0.1, 0.4, -0.3],
                       [-0.2, 0.5, 0.2]])
test_w_h_o = np.array([[0.3, -0.1]])

class TestMethods(unittest.TestCase):
    ##########
    # Unit tests for data loading
    ##########

    def test_data_path(self):
        # Test that file path to dataset has been unaltered
        self.assertTrue(data_path.lower() == '/home/aportabales/udacity/DLND-your-first-network/Bike-Sharing-Dataset/hour.csv')

    def test_data_loaded(self):
        # Test that data frame loaded
        self.assertTrue(isinstance(rides, pd.DataFrame))

    ##########
    # Unit tests for network functionality
    ##########

    def test_activation(self):
        network = NeuralNetwork(3, 2, 1, 0.5)
        # Test that the activation function is a sigmoid
        self.assertTrue(np.all(network.activation_function(0.5) == 1 / (1 + np.exp(-0.5))))

    def test_train(self):
        # Test that weights are updated correctly on training
        network = NeuralNetwork(3, 2, 1, 0.5)
        network.weights_input_to_hidden = test_w_i_h.copy()
        network.weights_hidden_to_output = test_w_h_o.copy()

        network.train(inputs, targets)
        self.assertTrue(np.allclose(network.weights_hidden_to_output,
                                    np.array([[0.37275328, -0.03172939]])))
        self.assertTrue(np.allclose(network.weights_input_to_hidden,
                                    np.array([[0.10562014, 0.39775194, -0.29887597],
                                              [-0.20185996, 0.50074398, 0.19962801]])))

    def test_run(self):
        # Test correctness of run method
        network = NeuralNetwork(3, 2, 1, 0.5)
        network.weights_input_to_hidden = test_w_i_h.copy()
        network.weights_hidden_to_output = test_w_h_o.copy()

        self.assertTrue(np.allclose(network.run(inputs), 0.09998924))

suite = unittest.TestLoader().loadTestsFromModule(TestMethods())
unittest.TextTestRunner().run(suite)