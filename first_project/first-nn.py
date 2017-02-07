import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data_path = 'DLND-your-first-network/Bike-Sharing-Dataset/hour.csv'

rides = pd.read_csv(data_path)

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

from NeuralNetwork import NeuralNetwork
import sys

def MSE(y, Y):
    return np.mean((y-Y)**2)


import sys

### Set the hyperparameters here ###
epochs = 100
#learning_rate = 0.0001
#hidden_nodes = 100
output_nodes = 1

for learning_rate in [0.00001]:
    for hidden_nodes in range(1,200,5):
        N_i = train_features.shape[1]
        network = NeuralNetwork(N_i, hidden_nodes, output_nodes, learning_rate)

        losses = {'train': [], 'validation': []}
        for e in range(epochs):
            # Go through a random batch of 128 records from the training data set
            batch = np.random.choice(train_features.index, size=128)
            for record, target in zip(train_features.ix[batch].values,
                                      train_targets.ix[batch]['cnt']):
                network.train(record, target)

            # Printing out the training progress
            train_loss = MSE(network.run(train_features), train_targets['cnt'].values)
            val_loss = MSE(network.run(val_features), val_targets['cnt'].values)
            sys.stdout.write("\rProgress: " + str(100 * e / float(epochs))[:4] \
                             + "% ... Training loss: " + str(train_loss)[:5] \
                             + " ... Validation loss: " + str(val_loss)[:5])

            losses['train'].append(train_loss)
            losses['validation'].append(val_loss)

        predictions = network.run(test_features)*std + mean
        print("\nlearning_rate: ",learning_rate, " hidden_nodes: ", hidden_nodes)
        print("MSE: ",MSE(predictions.T[0],(test_targets['cnt']*std + mean).values))

