import numpy as np
from scipy.optimize import fmin_cg
import matplotlib.pyplot as plt
import pandas as pd
from mnist_nn import *

# Read dataset + bit of preprocessing
train_images = np.fromfile("train-images.idx3-ubyte", dtype=np.ubyte)
train_images = train_images[16:]

train_labels = np.fromfile("train-labels.idx1-ubyte", dtype=np.ubyte)
train_labels = train_labels[8:]

test_images = np.fromfile("t10k-images.idx3-ubyte", dtype=np.ubyte)
test_images = test_images[16:]

test_labels = np.fromfile("t10k-labels.idx1-ubyte", dtype=np.ubyte)
test_labels = test_labels[8:]

# Reshape into a matrix having dimensions (60000, 28 * 28)
train_images = np.reshape(train_images, (60000, 784))

# Reshape into a matrix having dimensions (10000, 28 * 28)
test_images = np.reshape(test_images, (10000, 784))

# Initialise the network
m = len(train_images)
input_layer_size = 784
num_labels = 10
hidden_layer_size = 25
reg_lambda = 1

r_w = input("Read parameters from csv? (y/n): ")

if r_w == "y":
    theta0 = np.loadtxt("theta0.csv", delimiter=',')
    theta1 = np.loadtxt("theta1.csv", delimiter=',')
else:
    init_epsilon_0 = np.sqrt(6) / \
        (np.sqrt(input_layer_size) + np.sqrt(hidden_layer_size))
    init_epsilon_1 = np.sqrt(6) / \
        (np.sqrt(hidden_layer_size) + np.sqrt(num_labels))
    theta0 = np.random.random((hidden_layer_size, input_layer_size + 1)) \
        * 2 * init_epsilon_0 - init_epsilon_0
    theta1 = np.random.random((num_labels, hidden_layer_size + 1)) \
        * 2 * init_epsilon_1 - init_epsilon_1

# One-hot encoding
train_labels_vector = np.zeros((num_labels, m))
for i in range(m):
    train_labels_vector[train_labels[i], i] = 1

test_labels_vector = np.zeros((num_labels, len(test_images)))
for i in range(len(test_images)):
    test_labels_vector[test_labels[i], i] = 1

# Print inital cost and accuracy
print("Initial cost and accuracy:")
print_cost_accuracy(train_images, train_labels, train_labels_vector,
                        test_images, test_labels, test_labels_vector,
                        theta0, theta1)

# Some useful initialisations
theta = np.append(theta0.flatten(), theta1.flatten())
my_cost = lambda theta: back_propagation(theta, train_images, train_labels_vector,
                                   reg_lambda, input_layer_size,
                                   hidden_layer_size, num_labels, 0)
my_gradient = lambda theta: back_propagation(theta, train_images, train_labels_vector,
                                       reg_lambda, input_layer_size,
                                       hidden_layer_size, num_labels, 1)

# Train the network
periods = 10
cost_history = np.zeros(periods + 1)
for i in range(periods):
    cost_history[i] = cost(train_images, train_labels_vector, theta0, theta1, reg_lambda)
    theta = fmin_cg(f=my_cost, x0=theta, fprime=my_gradient, maxiter=50/periods)
    
    theta0 = np.reshape(theta[0 : hidden_layer_size * (input_layer_size + 1)],
        (hidden_layer_size, input_layer_size + 1))
    theta1 = np.reshape(theta[hidden_layer_size * (input_layer_size + 1) :],
        (num_labels, hidden_layer_size + 1))
    print("\n\nPeriod {}.".format(i + 1))
    print_cost_accuracy(train_images, train_labels, train_labels_vector,
                        test_images, test_labels, test_labels_vector,
                        theta0, theta1)

cost_history[periods] = cost(train_images, train_labels_vector, theta0, theta1, reg_lambda)

plt.title("Cost vs Periods")
plt.xlabel("Number of periods")
plt.ylabel("Cost")
plt.plot(cost_history)
plt.show()

print("Final cost and accuracy:")
print_cost_accuracy(train_images, train_labels, train_labels_vector,
                        test_images, test_labels, test_labels_vector,
                        theta0, theta1)

r_w = input("Write parameters to csv? (y/n): ")
if r_w == "y":
    pd.DataFrame(theta0).to_csv("theta0.csv", header=False, index=False)
    pd.DataFrame(theta1).to_csv("theta1.csv", header=False, index=False)
