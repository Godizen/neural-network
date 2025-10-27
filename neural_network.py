import numpy as np
import random
import math

# ReLU for most common hidden layers
def ReLU(x):
    return np.maximum(0, x)

def softmax(z):
    z = np.array(z).reshape(-1, 1)            # ensure a column vector
    z = z - np.max(z)
    exp_z = np.exp(z)
    sum_exp = np.sum(exp_z)
    return exp_z / sum_exp

class neuralNetwork:
    def __init__(self, layerSizes, stepSize=0.001, optimiser="lion") -> None:
        self.optimiser = optimiser
        self.nLayers = len(layerSizes)
        self.stepSize = stepSize
        self.adamTimeStep = 0

        # He initialization
        self.weights = []
        self.biases = []
        for i in range(0, self.nLayers -1):
            self.weights.append(np.random.randn(layerSizes[i+1], layerSizes[i]) * np.sqrt(2 / layerSizes[i]))
            self.biases.append(np.zeros((layerSizes[i+1], 1)))

        self.m = [np.zeros_like(w) for w in self.weights]
        self.v = [np.zeros_like(w) for w in self.weights]
        self.mb = [np.zeros_like(b) for b in self.biases]
        self.vb = [np.zeros_like(b) for b in self.biases]
            
    def use(self, x) -> None:
        a = x.reshape(-1, 1)
        activations = [a]
        preacts = []

        for i in range(self.nLayers - 2):
            z = self.weights[i] @ a + self.biases[i]
            a = ReLU(z)
            preacts.append(z)
            activations.append(a)
        
        z = self.weights[-1] @ a + self.biases[-1]
        a = softmax(z)
        preacts.append(z)
        activations.append(a)

        return activations, preacts

    def calculateDeltas(self, activations, preacts, y):
        deltas = [None] * (self.nLayers - 1)
        y = y.reshape(-1, 1)

        deltas[-1] = activations[-1] - y # Softmax Derivative

        # Hidden Layers
        for l in range(self.nLayers - 3, -1, -1):
            da = self.weights[l+1].T @ deltas[l+1]
            dz = da * (preacts[l] > 0).astype(float)
            deltas[l] = dz

        return deltas

    def backpropagate(self, activations, preacts, expected_op) -> None:
        deltas = self.calculateDeltas(activations, preacts, expected_op)
        beta1, beta2 = 0.9, 0.999
        eps = 0.00000001
        weight_decay = 0.01
        self.adamTimeStep += 1

        for l in range(self.nLayers - 1):
            grad_w = deltas[l] @ activations[l].T
            grad_b = deltas[l]

            grad_w = np.clip(grad_w, -1.0, 1.0)
            grad_b = np.clip(grad_b, -1.0, 1.0)
        
            if self.optimiser == "lion":
                self.m[l] = beta1 * self.m[l] + (1 - beta1) * grad_w
                update = beta2 * self.m[l] + (1 - beta2) * grad_w
                self.weights[l] -= self.stepSize * (np.sign(update) + weight_decay * self.weights[l])

                self.mb[l] = beta1 * self.mb[l] + (1 - beta1) * grad_b
                update_b = beta2 * self.mb[l] + (1 - beta2) * grad_b
                self.biases[l] -= self.stepSize * np.sign(update_b)

            if self.optimiser == "adamw":
                self.m[l] = beta1 * self.m[l] + (1 - beta1) * grad_w
                self.v[l] = beta2 * self.v[l] + (1 - beta2) * (grad_w ** 2)
                mhat = self.m[l] / (1 - beta1 ** self.adamTimeStep)
                vhat = self.v[l] / (1 - beta2 ** self.adamTimeStep)
                self.weights[l] -= self.stepSize * (mhat / (np.sqrt(vhat) + eps) + weight_decay * self.weights[l])

                self.mb[l] = beta1 * self.mb[l] + (1 - beta1) * grad_b
                self.vb[l] = beta2 * self.vb[l] + (1 - beta2) * (grad_b ** 2)
                mhatb = self.mb[l] / (1 - beta1 ** self.adamTimeStep)
                vhatb = self.vb[l] / (1 - beta2 ** self.adamTimeStep)
                self.biases[l] -= self.stepSize * mhatb / (np.sqrt(vhatb) + eps)

    def train(self, X, Y, epochs) -> None:
        for epoch in range(1, epochs + 1):
            indices = np.arange(len(X))
            np.random.shuffle(indices)
            if epoch != 1:
                prev_loss = loss
            loss = 0.0

            for idx in indices:
                x, y = np.array(X[idx]), np.array(Y[idx])
                activations, preacts = self.use(x)
                self.backpropagate(activations, preacts, y)
                pred = activations[-1].flatten()
                eps = 1e-12
                pred = np.clip(pred, eps, 1 - eps)
                loss += -np.sum(y * np.log(pred))
            
            loss /= len(X)

            print(f"Epoch {epoch} - Loss: {loss:.6f}")
            if epoch != 1:
                print(f"Change in Loss: {loss - prev_loss:.6f}")