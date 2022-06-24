import cv2
import numpy as np

#Create the artificial neural network.
ann = cv2.ml.ANN_MLP_create()

#Set the number of nodes in the input layer, hidden layer and output layer.
ann.setLayerSizes(np.array([784,200,10]))

#Set the activation function (sigmoid function is selected in this case).
ann.setActivationFunction(cv2.ml.ANN_MLP_SIGMOID_SYM, 2.5, 1.0)

#Set the training method and learning rate.
ann.setTrainMethod(cv2.ml.ANN_MLP_BACKPROP, 0.01)

# Load the MNIST training data stored in the CSV file into a list.
training_data_file = open("mnist_train.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

# Prepare an array to store all the training data.
training_data_array = np.zeros((60000,784),np.float32)
training_data_class_label_array = np.zeros((60000,10),np.float32)

for x in range(len(training_data_list)):
    # Split the intensity values based on commas (,) and store the values into a list.
    all_values = training_data_list[x].split(',')
    
    # Convert each intensity value to an integer (from string) and store in an array.
    # Scale the intensity values to the range of 0.01 to 1.00.
    inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    
    # Reshape and put all the intensity values into the array.
    inputs = np.reshape(inputs,(1,784))
    training_data_array[x,:] = inputs
    
    # Create the expected output values and set all to 0.01.
    targets = np.zeros(10) + 0.01
    
    # The first value in a sample (all_values[0]) is the correct label.
    # Set the excepted output value of the correct class to 0.99.
    targets[int(all_values[0])] = 0.99
    
    # Reshape and put the expected value of each class into the array.
    training_data_class_label_array[x,:] = targets

# Train the artificial neural network using the training data.
ann.train(training_data_array,cv2.ml.ROW_SAMPLE,training_data_class_label_array)

# Load the MNIST testing data stored in the CSV file into a list.
test_data_file = open("mnist_test.csv", 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

# Prepare an array to store all the testing data.
test_data_array = np.zeros((10000,784),np.float32)
test_data_answer_array = np.zeros((10000,1),np.float32)

for x in range(len(test_data_list)):
    # Split the intensity values based on commas (,) and store the values into a list.
    all_values = test_data_list[x].split(',')
    
    # Convert each intensity value to an integer (from string) and store in an array.
    # Scale the intensity values to the range of 0.01 to 1.00.
    inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    
    # Reshape and put all the intensity values into the array.
    inputs = np.reshape(inputs,(1,784))
    test_data_array[x,:] = inputs
    
    # The first value represents the correct answer (i.e. 0 - 9).
    # It is not an intensity value.
# Store the first values (the correct answers) into the array.
# The answers will
    test_data_answer_array[x,:] = int(all_values[0])
    
# Use the trained artificial neural network for prediction.
_, all_prediction_results = ann.predict(test_data_array)

# To evalutate the neural network.
# Use to store matching result (1 for correct match, else 0), initially empty.
accuracy_list = []

for x in range(len(all_prediction_results)):
    # Get the index of highest output.
    # For example, if outputs[5] has the higest output, then the index returned is 5.
    # The index corresponds to the predicted label.
    prediction = np.argmax(all_prediction_results[x,:])
    
    # Get the correct answer.
    answer = test_data_answer_array[x,:]
    
    # Append the matching result to the list.
    if (prediction == answer):
        # Add 1 to indicata a correct match.
        accuracy_list.append(1)
    else:
        # Add 1 to indicata an incorrect match.
        accuracy_list.append(0)

# Calculate the overall acurracy.
accuracy_array = np.asarray(accuracy_list)
print ("Accuracy = ", accuracy_array.sum() / accuracy_array.size)
