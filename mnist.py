import os
import gzip
import struct
from neural_network import neuralNetwork

# === CONFIG ===
FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images":  "t10k-images-idx3-ubyte.gz",
    "test_labels":  "t10k-labels-idx1-ubyte.gz",
}
DATA_DIR = "mnist_data"

# === STEP 2: Helper functions ===
def load_images(path):
    with gzip.open(path, 'rb') as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        buf = f.read(rows * cols * num)
        data = [list(buf[i*rows*cols:(i+1)*rows*cols]) for i in range(num)]
    # Normalize to 0–1
    data = [[px / 255.0 for px in img] for img in data]
    return data

def load_labels(path):
    with gzip.open(path, 'rb') as f:
        magic, num = struct.unpack(">II", f.read(8))
        labels = list(f.read(num))
    return labels

# === STEP 3: Load all data ===
train_images = load_images(os.path.join(DATA_DIR, FILES["train_images"]))
train_labels = load_labels(os.path.join(DATA_DIR, FILES["train_labels"]))
test_images = load_images(os.path.join(DATA_DIR, FILES["test_images"]))
test_labels = load_labels(os.path.join(DATA_DIR, FILES["test_labels"]))

# === STEP 4: Check shapes ===
print("Train samples:", len(train_images))
print("Test samples:", len(test_images))
print("Image size:", len(train_images[0]), "pixels (28x28)")

actual_labels = []

for i in range(0, len(train_labels)):
    actual_labels.append([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
    actual_labels[i][train_labels[i]] = 1.0
    

NN = neuralNetwork([784, 128, 64, 10], .0003, "adamw")
#NN.initialise()
print("starting training")
NN.train(train_images, actual_labels, 25)