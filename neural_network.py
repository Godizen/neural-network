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
            
    def use(self, x):
        a = x.reshape(-1, 1)
        self.activations = [a]
        self.preacts = []

        for i in range(self.nLayers - 2):
            z = self.weights[i] @ a + self.biases[i]
            a = ReLU(z)
            self.preacts.append(z)
            self.activations.append(a)
        
        z = self.weights[-1] @ a + self.biases[-1]
        a = softmax(z)
        self.preacts.append(z)
        self.activations.append(a)


    def calculateDeltas(self, y):
        deltas = [None] * (self.nLayers - 1)
        y = y.reshape(-1, 1)

        deltas[-1] = self.activations[-1] - y # Softmax Derivative

        # Hidden Layers
        for l in range(self.nLayers - 3, -1, -1):
            da = self.weights[l+1].T @ deltas[l+1]
            dz = da * (self.preacts[l] > 0).astype(float)
            deltas[l] = dz

        return deltas

    def backpropagate(self, expected_op) -> None:
        deltas = self.calculateDeltas(expected_op)
        beta1, beta2 = 0.9, 0.999
        eps = 0.00000001
        weight_decay = 0.01
        self.adamTimeStep += 1

        for l in range(self.nLayers - 1):
            grad_w = deltas[l] @ self.activations[l].T
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

    def train(self, X, Y, epochs, doTillCutoff=False, cutoff=0, diverge_cutoff=math.inf, log_interval=1) -> None:
        eps = 1e-12
        if not doTillCutoff:
            diverged_epochs = 0
            prev_loss = float('inf')
            for epoch in range(1, epochs + 1):
                indices = np.arange(len(X))
                np.random.shuffle(indices)
                loss = 0.0

                for idx in indices:
                    x, y = np.array(X[idx]), np.array(Y[idx])
                    self.use(x)
                    self.backpropagate(y)
                    pred = self.activations[-1].flatten()
                    pred = np.clip(pred, eps, 1 - eps)
                    loss += -np.sum(y * np.log(pred))
                
                loss /= len(X)

                if epoch % log_interval == 0 or epoch == 1:
                    print(f"Epoch {epoch} - Loss: {loss:.6f} | Change in Loss: {loss - prev_loss:.6f}")
                
                if prev_loss - loss < cutoff and epoch != 1:
                    print("Early stopping criteria met.")
                    break
                
                if loss - prev_loss > 0.0 and epoch != 1:
                    diverged_epochs += 1
                else:
                    diverged_epochs = 0
                
                if diverged_epochs >= diverge_cutoff and epoch != 1:
                    print("Loss diverged for 5 epochs, stopping training.")
                    break
                prev_loss = loss
        else:
            diverged_epochs = 0
            epoch = 1
            prev_loss = float('inf')
            while True:
                indices = np.arange(len(X))
                np.random.shuffle(indices)
                loss = 0.0

                for idx in indices:
                    x, y = np.array(X[idx]), np.array(Y[idx])
                    self.use(x)
                    self.backpropagate(y)
                    pred = self.activations[-1].flatten()
                    pred = np.clip(pred, eps, 1 - eps)
                    loss += -np.sum(y * np.log(pred))
                
                loss /= len(X)

                if epoch % log_interval == 0 or epoch == 1:
                    print(f"Epoch {epoch} - Loss: {loss:.6f} | Change in Loss: {loss - prev_loss:.6f}")
                
                if prev_loss - loss < cutoff and epoch != 1:
                    print("Early stopping criteria met.")
                    break
                
                if loss - prev_loss > 0.0 and epoch != 1:
                    diverged_epochs += 1
                else:
                    diverged_epochs = 0
                
                if diverged_epochs >= diverge_cutoff and epoch != 1:
                    print("Loss diverged for 5 epochs, stopping training.")
                    break
                
                epoch += 1
                prev_loss = loss


class recurrentNeuralNetwork:
    def __init__(self, layerSizes, stepSize=0.001, optimiser="lion") -> None:
        self.optimiser = optimiser
        self.nLayers = len(layerSizes)
        self.stepSize = stepSize
        self.adamTimeStep = 0

        # He initialization
        self.weights = []
        self.hiddenStateWeights = []
        self.biases = []
        for i in range(0, self.nLayers -1):
            self.weights.append(np.random.randn(layerSizes[i+1], layerSizes[i]) * np.sqrt(2 / layerSizes[i]))
            if i+1 != self.nLayers -1:  # No hidden state weights for output layer
                self.hiddenStateWeights.append(np.random.randn(layerSizes[i+1], layerSizes[i+1]) * np.sqrt(2 / layerSizes[i+1]))
            self.biases.append(np.zeros((layerSizes[i+1], 1)))
        
        self.m = [np.zeros_like(w) for w in self.weights]
        self.v = [np.zeros_like(w) for w in self.weights]
        self.mh = [np.zeros_like(wh) for wh in self.hiddenStateWeights]
        self.vh = [np.zeros_like(wh) for wh in self.hiddenStateWeights]
        self.mb = [np.zeros_like(b) for b in self.biases]
        self.vb = [np.zeros_like(b) for b in self.biases]

        self.h_prev = [np.zeros((layerSizes[i+1], 1)) for i in range(self.nLayers -2)]  # Previous hidden states for each hidden layer
            
    def use(self, x):
        a = x.reshape(-1, 1)
        self.activations = [a]
        self.preacts = []

        for i in range(self.nLayers - 2):
            z = self.weights[i] @ a + self.biases[i] + self.hiddenStateWeights[i] @ self.h_prev[i]
            a = ReLU(z)

            self.h_prev[i] = a.copy()  # Update hidden state

            self.preacts.append(z)
            self.activations.append(a)
        
        z = self.weights[-1] @ a + self.biases[-1]
        a = softmax(z)
        self.preacts.append(z)
        self.activations.append(a)

    def calculateDeltas(self, Y_sequence):
        T = len(Y_sequence)
        deltas_time = [None] * T
        d_h_next = [np.zeros_like(h) for h in self.h_prev]  # For recurrent gradient

        for t in reversed(range(T)):
            y_true = Y_sequence[t].reshape(-1, 1)
            activations = self.time_activations[t]
            preacts = self.time_preacts[t]

            deltas = [None] * (self.nLayers - 1)

            # Output delta (softmax)
            deltas[-1] = activations[-1] - y_true

            # Hidden layers
            da = self.weights[-1].T @ deltas[-1]

            for l in reversed(range(self.nLayers - 2)):
                # recurrent contribution (from next time step)
                recurrent_term = self.hiddenStateWeights[l].T @ d_h_next[l]
                
                dz = (da + recurrent_term) * (preacts[l] > 0).astype(float)
                deltas[l] = dz

                # store this for next timestep
                d_h_next[l] = dz.copy()

                da = self.weights[l].T @ dz

            deltas_time[t] = deltas

        return deltas_time

    def backpropagate(self, expected_sequence) -> None:
        deltas_time = self.calculateDeltas(expected_sequence)
        beta1, beta2 = 0.9, 0.999
        eps = 1e-8
        weight_decay = 0.01
        self.adamTimeStep += 1

        # Initialize accumulated gradients
        grad_W = [np.zeros_like(w) for w in self.weights]
        grad_b = [np.zeros_like(b) for b in self.biases]
        grad_Wh = [np.zeros_like(Wh) for Wh in self.hiddenStateWeights]

        # ----- Accumulate gradients across all timesteps -----
        T = len(self.time_activations)
        for t in range(T):
            activations_t = self.time_activations[t]
            deltas_t = deltas_time[t]

            # Regular weights and biases
            for l in range(self.nLayers - 1):
                grad_W[l] += deltas_t[l] @ activations_t[l].T
                grad_b[l] += deltas_t[l]

            # Hidden-to-hidden weights (recurrent)
            for l in range(len(self.hiddenStateWeights)):
                if t > 0:
                    h_prev_t = self.time_activations[t - 1][l + 1]
                else:
                    h_prev_t = np.zeros((self.hiddenStateWeights[l].shape[1], 1))
                grad_Wh[l] += deltas_t[l] @ h_prev_t.T

        # ----- Normalize by sequence length -----
        grad_W = [g / T for g in grad_W]
        grad_b = [g / T for g in grad_b]
        grad_Wh = [g / T for g in grad_Wh]

        # ----- Gradient clipping -----
        grad_W = [np.clip(g, -1.0, 1.0) for g in grad_W]
        grad_b = [np.clip(g, -1.0, 1.0) for g in grad_b]
        grad_Wh = [np.clip(g, -1.0, 1.0) for g in grad_Wh]

        # ----- Apply updates -----
        for l in range(self.nLayers - 1):
            if self.optimiser == "lion":
                # --- Standard weights ---
                self.m[l] = beta1 * self.m[l] + (1 - beta1) * grad_W[l]
                update = beta2 * self.m[l] + (1 - beta2) * grad_W[l]
                self.weights[l] -= self.stepSize * (np.sign(update) + weight_decay * self.weights[l])

                # --- Biases ---
                self.mb[l] = beta1 * self.mb[l] + (1 - beta1) * grad_b[l]
                update_b = beta2 * self.mb[l] + (1 - beta2) * grad_b[l]
                self.biases[l] -= self.stepSize * np.sign(update_b)

            elif self.optimiser == "adamw":
                # --- Standard weights ---
                self.m[l] = beta1 * self.m[l] + (1 - beta1) * grad_W[l]
                self.v[l] = beta2 * self.v[l] + (1 - beta2) * (grad_W[l] ** 2)
                mhat = self.m[l] / (1 - beta1 ** self.adamTimeStep)
                vhat = self.v[l] / (1 - beta2 ** self.adamTimeStep)
                self.weights[l] -= self.stepSize * (mhat / (np.sqrt(vhat) + eps) + weight_decay * self.weights[l])

                # --- Biases ---
                self.mb[l] = beta1 * self.mb[l] + (1 - beta1) * grad_b[l]
                self.vb[l] = beta2 * self.vb[l] + (1 - beta2) * (grad_b[l] ** 2)
                mhatb = self.mb[l] / (1 - beta1 ** self.adamTimeStep)
                vhatb = self.vb[l] / (1 - beta2 ** self.adamTimeStep)
                self.biases[l] -= self.stepSize * mhatb / (np.sqrt(vhatb) + eps)

        # ----- Update recurrent hidden weights (Wh) -----
        for l in range(len(self.hiddenStateWeights)):
            if self.optimiser == "lion":
                self.mh[l] = beta1 * self.mh[l] + (1 - beta1) * grad_Wh[l]
                update = beta2 * self.mh[l] + (1 - beta2) * grad_Wh[l]
                self.hiddenStateWeights[l] -= self.stepSize * (np.sign(update) + weight_decay * self.hiddenStateWeights[l])

            elif self.optimiser == "adamw":
                self.mh[l] = beta1 * self.mh[l] + (1 - beta1) * grad_Wh[l]
                self.vh[l] = beta2 * self.vh[l] + (1 - beta2) * (grad_Wh[l] ** 2)
                mhat = self.mh[l] / (1 - beta1 ** self.adamTimeStep)
                vhat = self.vh[l] / (1 - beta2 ** self.adamTimeStep)
                self.hiddenStateWeights[l] -= self.stepSize * (mhat / (np.sqrt(vhat) + eps) + weight_decay * self.hiddenStateWeights[l])

    def train(self, X, Y, epochs) -> None:
        eps = 1e-12
        prev_loss = float('inf')

        for epoch in range(1, epochs + 1):
            indices = np.arange(len(X))
            np.random.shuffle(indices)
            total_loss = 0.0
            total_steps = 0  # total timesteps for average loss

            for idx in indices:
                sequence_X, sequence_Y = X[idx], Y[idx]

                # Reset hidden states for each new sequence
                self.h_prev = [np.zeros((Wh.shape[0], 1)) for Wh in self.hiddenStateWeights]

                # store activations across timesteps
                self.time_activations = []
                self.time_preacts = []

                # ---------- Forward pass through the whole sequence ----------
                for t in range(len(sequence_X)):
                    x_t = np.array(sequence_X[t])
                    self.use(x_t)

                    # store activations for BPTT
                    self.time_activations.append([a.copy() for a in self.activations])
                    self.time_preacts.append([p.copy() for p in self.preacts])

                    # accumulate loss
                    y_t = np.array(sequence_Y[t])
                    pred = self.activations[-1].flatten()
                    pred = np.clip(pred, eps, 1 - eps)
                    total_loss += -np.sum(y_t * np.log(pred))
                    total_steps += 1

                # ---------- Backpropagation Through Time (once per sequence) ----------
                self.backpropagate(sequence_Y)

            # ---------- Epoch stats ----------
            avg_loss = total_loss / total_steps
            print(f"Epoch {epoch} - Loss: {avg_loss:.6f}")
            if epoch != 1:
                print(f"Change in Loss: {avg_loss - prev_loss:.6f}")
            prev_loss = avg_loss