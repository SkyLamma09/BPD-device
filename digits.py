from keras.datasets import mnist
import numpy as np
from scipy.special import softmax
class Network:
    def __init__(self):
        (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
        self.weights1 = np.random.randn(784, 128) * 0.01
        self.weights2 = np.random.randn(128, 10) * 0.01
        self.inputs = None
        self.biases1 = np.zeros((1,128))
        self.biases2 = np.zeros((1,10))
        self.learning_rate = 0.01
        indices = np.random.permutation(len(train_images))
        self.train_images = train_images[indices].astype(np.float32) / 255.0
        self.train_labels = train_labels[indices]
        self.test_images = test_images.astype(np.float32) / 255.0
        self.test_labels = test_labels
    def forward(self):
        self.hidden = np.maximum(0,(self.inputs @ self.weights1 + self.biases1))
        self.logits = self.hidden @ self.weights2 + self.biases2
        self.probabilities = softmax(self.logits, axis=1)
        self.num = np.argmax(self.probabilities)
    def softmax(self):
        for e,i in enumerate(self.probabilities):
            print(e, "is", i)
    def backpropogate(self,groundtruth):
        correct = np.zeros((1,10))
        correct[0,groundtruth] += 1
        diff2 = self.probabilities-correct
        gradient2 = self.hidden.T @ diff2
        diff1 = diff2 @ self.weights2.T
        gradient1 = self.inputs.T @ ((self.hidden > 0)*diff1)
        self.weights2 = self.weights2 - (gradient2*self.learning_rate)
        self.weights1 = self.weights1 - (gradient1*self.learning_rate)
        self.biases2 -= diff2*self.learning_rate
        self.biases1 -= diff1*self.learning_rate
network = Network()
predictions = []
print("________TRAINING__________")
for epoch in range(3):
    print(f"___________Doing epoch {epoch+1}___________")
    for image, label in zip(network.train_images, network.train_labels):
        network.inputs = image.reshape(1,784)
        network.forward()
        network.backpropogate(label)

print("________TESTING__________")
for image, label in zip(network.test_images, network.test_labels):
        network.inputs = image.reshape(1,784)
        network.forward()
        predictions.append(network.num)
print(predictions)
counter = 0
total = 0
groundtruths = network.test_labels
print(groundtruths)
for i in range(len(groundtruths)):
    if predictions[i] == groundtruths[i]:
        counter += 1
    total += 1
print(f"The accuracy is {100*(counter/total)}%")
