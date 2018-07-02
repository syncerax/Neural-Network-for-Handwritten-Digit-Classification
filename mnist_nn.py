import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_gradient(z):
    g = sigmoid(z)
    return g * (1 - g)

def feedforward_propagation(X, theta0, theta1):
    m = len(X)

    a0 = np.hstack((np.ones((m, 1)), X)).T
    z1 = theta0 @ a0
    a1 = sigmoid(z1)
    a1 = np.vstack((np.ones((1, m)), a1))
    z2 = theta1 @ a1
    a2 = sigmoid(z2)
    return a2


def cost(X, y, theta0, theta1, reg_lambda):
    m = len(X)
    h = feedforward_propagation(X, theta0, theta1)
    J = (-1 / m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
    reg = (reg_lambda / (2 * m)) * \
        (np.sum(theta0[:, 2:] ** 2) + np.sum(theta1[:, 2:] ** 2))
    return J + reg


def back_propagation(theta, X, y, reg_lambda, input_layer_size,
                     hidden_layer_size, num_labels, cost_or_grad):
    m = len(X)

    theta0 = np.reshape(theta[0 : hidden_layer_size * (input_layer_size + 1)],
        (hidden_layer_size, input_layer_size + 1))
    theta1 = np.reshape(theta[hidden_layer_size * (input_layer_size + 1) :],
        (num_labels, hidden_layer_size + 1))

    a0 = np.hstack((np.ones((m, 1)), X)).T
    z1 = theta0 @ a0
    a1 = sigmoid(z1)
    a1 = np.vstack((np.ones((1, m)), a1))
    z2 = theta1 @ a1
    h = sigmoid(z2)

    J = (-1 / m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
    reg = (reg_lambda / (2 * m)) * \
        (np.sum(theta0[:, 2:] ** 2) + np.sum(theta1[:, 2:] ** 2))
    
    if cost_or_grad == 0:
        return J + reg

    d2 = h - y
    d1 = theta1[:, 1:].T @ d2 * sigmoid_gradient(z1)
    delta0 = d1 @ a0.T
    delta1 = d2 @ a1.T
    theta0[:, 0] = 0
    theta1[:, 0] = 0

    theta0_grad = (1 / m) * delta0 + (reg_lambda / m) * theta0
    theta1_grad = (1 / m) * delta1 + (reg_lambda / m) * theta1

    grad = np.append(theta0_grad.flatten(), theta1_grad.flatten())
    return grad

def print_cost_accuracy(X_train, y_train, y_train_vector,
                        X_test, y_test, y_test_vector,
                        theta0, theta1):
    predictions = feedforward_propagation(X_train, theta0, theta1)
    predictions = np.argmax(predictions, axis=0)
    J = cost(X_train, y_train_vector, theta0, theta1, 0)
    print("Training set cost = {}".format(J))
    print("Training set accuracy = {}".format(np.mean(predictions == y_train) * 100))
    predictions = feedforward_propagation(X_test, theta0, theta1)
    predictions = np.argmax(predictions, axis=0)
    J = cost(X_test, y_test_vector, theta0, theta1, 0)
    print("Testing set cost = {}".format(J))
    print("Testing set accuracy = {}\n\n".format(np.mean(predictions == y_test) * 100))