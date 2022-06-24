from NeuralNetwork import neuralNetwork
import numpy as np

# Set the number of input, hidden and output nodes
input_nodes = 784
hidden_nodes = 200
output_nodes = 10

# Set the learning rate.
learning_rate = 0.01

# Create a neural network based on the settings given above.
n = neuralNetwork(input_nodes,hidden_nodes,output_nodes, learning_rate)

# Load the MNIST training data stored in the CSV file into a list.
training_data_file = open("mnist_train.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

# Train the neural network.
# Epochs is the number of times the training data set is used for training.
epochs = 5

for e in range(epochs):
    # Go through all samples in the training data set.
    for sample in training_data_list:
        # Split the intensity values based on commas (,) and store the values into a list.
        all_values = sample.split(',')
        
        # Convert each intensity value to an integer (from string) and store in an array.
        # Scale the intensity values to the range of 0.01 to 1.00.
        inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        
        # Create the expected output values and set all to 0.01.
        targets = np.zeros(output_nodes) + 0.01
        
        # The first value in a sample (all_values[0]) is the correct label.
        # Set the excepted output value of the correct class to 0.99.
        targets[int(all_values[0])] = 0.99
       
        # Start training.
        n.train(inputs, targets)


# Load the MNIST test data stored in the CSV file into a list.
test_data_file = open("mnist_test.csv", 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

# Test the neural network.
# Use to store matching result (1 for correct match, else 0), initially empty.
accuracy_list = []

# Go through all the samples in the test data set
for sample in test_data_list:
    # Split the intensity values based on commas (,) and store the values into a list.
    all_values = sample.split(',')
    
    # The first value in a sample (all_values[0]) is the correct label.
    correct_label = int(all_values[0])
    
    # Convert each intensity value to an integer (from string) and store in an array.
    # Scale the intensity values to the range of 0.01 to 1.00.
    inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    
    # Query the network network.
    outputs = n.query(inputs)
    
    # Get the index of highest output.
    # For example, if outputs[5] has the higest output, then the index returned is 5.
    # The index corresponds to the predicted label.
    label = np.argmax(outputs)
    
    # Append the matching result to the list.
    if (label == correct_label):
        # Add 1 to indicata a correct match.
        accuracy_list.append(1)
    else:
        # Add 1 to indicata an incorrect match.
        accuracy_list.append(0)

# Calculate the overall acurracy.
accuracy_array = np.asarray(accuracy_list)
print ("Accuracy = ", accuracy_array.sum() / accuracy_array.size)
