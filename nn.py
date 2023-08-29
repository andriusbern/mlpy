from abc import abstractmethod
import random
import numpy as np


#######################
## Activation functions
def relu(x, deriv=False):
    if deriv:
        return (x > 0).astype(np.uint8).astype(np.float32)
    else:
        return np.maximum(x, 0)

def sigmoid(x, deriv=False):
    sigm = 1 / (1 + np.exp(-x))
    if deriv:
        return sigm * (1 - sigm)
    else:
        return sigm

def softmax(x, deriv=False):
    e = np.exp(x)
    s = np.sum(e, axis=1)
    s = np.expand_dims(s, axis=1)
    return e / s

#######################
## Data, labels
def one_hot(labels, n_classes=None):
    
    if n_classes is None:
        n_classes = int(np.max(labels) + 1) 
    try:
        n_samples = len(labels)
    except:
        n_samples = 1
    one_hot_labels = np.zeros([n_samples, n_classes])

    for label in range(n_classes):
        one_hot_labels[np.where(labels==label), label] = 1
    return one_hot_labels

def get_label(one_hot_labels):
    return np.argmax(np.squeeze(one_hot_labels), axis=1)

def split_data(data, labels, ratio=0.25):
    n_samples = len(labels)
    test_ids = random.sample(range(n_samples), int((n_samples-1)*ratio))
    train_ids = list(set(range(n_samples)).difference(test_ids))
    train_data = data[:, np.array(train_ids)]
    test_data = data[:, np.array(test_ids)]
    train_labels = labels[np.array(train_ids)].astype(np.uint8)
    test_labels = labels[np.array(test_ids)]

    return train_data, test_data, train_labels, test_labels

def split_data3d(data, labels, ratio=0.25):
    n_samples = len(labels)
    test_ids = random.sample(range(n_samples), int((n_samples-1)*ratio))
    train_ids = list(set(range(n_samples)).difference(test_ids))
    train_data = data[np.array(train_ids), :, :, :]
    test_data = data[np.array(test_ids), :, :, :]
    train_labels = labels[np.array(train_ids)].astype(np.uint8)
    test_labels = labels[np.array(test_ids)]

    return train_data, test_data, train_labels, test_labels

class Layer:
    def __init__(self, input_vector):
        self.shape = input_vector.shape
        zeros = np.zeros(self.shape)
        self.w  = zeros
        self.dw = zeros
        self.b  = zeros
        self.a = zeros
        self.z = zeros
        self.activation_fn = sigmoid
        self.summary()

    def __call__(self, input):
        return self.forward_pass(input)

    @abstractmethod
    def forward_pass(self, input):
        return None

    def summary(self):
        print('{}, nodes: {}, activation: {}'.format(self.__class__.__name__, self.shape, self.activation_fn))

    def __repr__(self):
        return '{}, nodes: {}, activation: {}'.format(self.__class__.__name__, self.shape, self.activation_fn)


class InputLayer(Layer):
    def __init__(self, input_tensor):
        Layer.__init__(self, input_tensor)
        self.shape = input_tensor.shape[-1]
        self.a = input_tensor

    def forward_pass(self, input_tensor):
        self.a = input_tensor.T
        return self.a


class FCLayer(Layer):
    def __init__(self, nodes, prev_layer, activation=sigmoid, w_init='randn'):
        Layer.__init__(self, prev_layer)
        print(prev_layer)
        self.nodes = nodes
        self.init_fn = getattr(np.random, w_init)
        self.shape = nodes
        self.prev_layer = prev_layer
        self.activation_fn = activation
        self.init_weights()
        self.dw = np.zeros_like(self.w)
        self.error = []
    
    def init_weights(self):
        self.w = (self.init_fn(self.prev_layer.shape, self.nodes)) * np.sqrt(1 / self.prev_layer.shape)
        self.b = np.zeros(self.nodes)
        self.mask = np.ones_like(self.b) # Neuron dropout mask

    def forward_pass(self, input_tensor):
        self.z = input_tensor @ self.w + self.b
        self.a = self.activation_fn(self.z) * self.mask
        return self.a

class MLP:
    def __init__(self, layers=[], input=None, output=None):
        self.layers = []
        self.input = []
        self.output = []
        self.optimizer = SGD(self)

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        self.output = x

        return x

    def build(self, input_vector, nodes_hidden, n_outputs):
        self.input = InputLayer(input_vector)
        self.layers.append(self.input)
        for node_count in nodes_hidden:
            layer = FCLayer(node_count, self.layers[-1], activation=sigmoid)
            self.layers.append(layer)
        output = FCLayer(n_outputs, self.layers[-1], activation=sigmoid)
        self.layers.append(output)

class SGD:
    def __init__(self, model):
        self.model = model
        self.lr = .01

    def MSE(self, labels):
        return np.mean((self.model.output - labels)**2) 
    
    def classify(self, labels):
        a = get_label(self.model.output)
        return np.sum(a == labels) 

    def backprop(self, data, labels):
        self.model(data)
        error = self.MSE(labels)

        dA = self.model.layers[-1].error = 2 * (self.model.output - labels).T
        for layer in reversed(self.model.layers[1:]):
            layer.error = dA
            pA = layer.prev_layer.a

            dz = dA * layer.activation_fn(layer.z.T, deriv=True)
            layer.dw = dw = (dz @ pA).T

            dA = layer.w @ dz
            layer.w = layer.w - layer.dw * self.lr
            layer.b = layer.b - np.mean(dz, axis=1) * self.lr

        return error

    def eval(self, data, labels):
        preds = self.model(data).argmax(axis=1)
        accuracy = (1 - np.count_nonzero(preds - labels.argmax(axis=1)) / labels.shape[0]) * 100
        error = self.MSE(labels)
        print('Error: ', error)

        return error, accuracy
