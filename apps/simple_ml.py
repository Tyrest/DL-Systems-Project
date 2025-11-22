"""hw1/apps/simple_ml.py"""

import struct
import gzip
import numpy as np
from tqdm import tqdm

import sys

sys.path.append("python/")
import needle as ndl

import needle.nn as nn
from apps.models import *
import time
device = ndl.cpu()

def parse_mnist(image_filesname, label_filename):
    """Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded
                data.  The dimensionality of the data should be
                (num_examples x input_dim) where 'input_dim' is the full
                dimension of the data, e.g., since MNIST images are 28x28, it
                will be 784.  Values should be of type np.float32, and the data
                should be normalized to have a minimum value of 0.0 and a
                maximum value of 1.0.

            y (numpy.ndarray[dypte=np.int8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.int8 and
                for MNIST will contain the values 0-9.
    """
    ### BEGIN YOUR SOLUTION
    with gzip.open(image_filesname, "rb") as f:
        f.read(4)
        num_images = struct.unpack(">I", f.read(4))[0]
        rows = struct.unpack(">I", f.read(4))[0]
        cols = struct.unpack(">I", f.read(4))[0]
        X = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, rows * cols)
        X = X.astype(np.float32) / 255.0

    with gzip.open(label_filename, "rb") as f:
        f.read(8)
        y = np.frombuffer(f.read(), dtype=np.uint8)

    return X, y
    ### END YOUR SOLUTION


def softmax_loss(Z, y_one_hot):
    """Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (ndl.Tensor[np.float32]): 2D Tensor of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (ndl.Tensor[np.int8]): 2D Tensor of shape (batch_size, num_classes)
            containing a 1 at the index of the true label of each example and
            zeros elsewhere.

    Returns:
        Average softmax loss over the sample. (ndl.Tensor[np.float32])
    """
    ### BEGIN YOUR SOLUTION
    log_sum_exp = ndl.ops.log(ndl.ops.summation(ndl.ops.exp(Z), axes=(1,)))
    correct_class_scores = ndl.ops.summation(Z * y_one_hot, axes=(1,))
    error = log_sum_exp - correct_class_scores
    return ndl.ops.summation(error) / Z.shape[0]
    ### END YOUR SOLUTION


def nn_epoch(X, y, W1, W2, lr=0.1, batch=100):
    """Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W2
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (ndl.Tensor[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (ndl.Tensor[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD mini-batch

    Returns:
        Tuple: (W1, W2)
            W1: ndl.Tensor[np.float32]
            W2: ndl.Tensor[np.float32]
    """

    ### BEGIN YOUR SOLUTION
    num_batches = X.shape[0] // batch
    for i in range(num_batches):
        X_batch = X[i * batch : (i + 1) * batch]
        y_batch = y[i * batch : (i + 1) * batch]
        X_tensor = ndl.Tensor(X_batch, requires_grad=False)
        y_one_hot = np.zeros((y_batch.shape[0], W2.shape[1]))
        y_one_hot[np.arange(y_batch.size), y_batch] = 1
        Z = X_tensor @ W1
        Z = ndl.ops.relu(Z)
        Z = Z @ W2
        loss = softmax_loss(Z, ndl.Tensor(y_one_hot, requires_grad=False))
        loss.backward()
        W1 -= lr * W1.grad
        W2 -= lr * W2.grad
    return W1, W2
    ### END YOUR SOLUTION

### CIFAR-10 training ###
def epoch_general_cifar10(dataloader, model, loss_fn=nn.SoftmaxLoss(), opt=None):
    """
    Iterates over the dataloader. If optimizer is not None, sets the
    model to train mode, and for each batch updates the model parameters.
    If optimizer is None, sets the model to eval mode, and simply computes
    the loss/accuracy.

    Args:
        dataloader: Dataloader instance
        model: nn.Module instance
        loss_fn: nn.Module instance
        opt: Optimizer instance (optional)

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    if opt is None:
        model.eval()
    else:
        model.train()
    accs = []
    losses = []
    for X, y in tqdm(dataloader, desc="Processing Batches", total=len(dataloader)):
        X_tensor = ndl.Tensor(X, device=model.device)
        y_pred = model(X_tensor)
        loss = loss_fn(y_pred, y)
        acc = (y_pred.detach().numpy().argmax(axis=1) == y.numpy()).mean()
        loss_val = loss.detach().numpy()
        if opt is not None:
            opt.reset_grad()
            loss.backward()
            opt.step()
        accs.append(acc)
        losses.append(loss_val)
    
    return np.mean(accs), np.mean(losses)
    ### END YOUR SOLUTION


def train_cifar10(model, dataloader, n_epochs=1, optimizer=ndl.optim.Adam,
          lr=0.001, weight_decay=0.001, loss_fn=nn.SoftmaxLoss()):
    """
    Performs {n_epochs} epochs of training.

    Args:
        dataloader: Dataloader instance
        model: nn.Module instance
        n_epochs: number of epochs (int)
        optimizer: Optimizer class
        lr: learning rate (float)
        weight_decay: weight decay (float)
        loss_fn: nn.Module class

    Returns:
        avg_acc: average accuracy over dataset from last epoch of training
        avg_loss: average loss over dataset from last epoch of training
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)
    avg_acc, avg_loss = 0, 0
    for epoch in range(n_epochs):
        avg_acc, avg_loss = epoch_general_cifar10(dataloader, model, loss_fn, opt)
        print(f"Epoch {epoch + 1}/{n_epochs}, Accuracy: {avg_acc}, Loss: {avg_loss}")
    return avg_acc, avg_loss
    ### END YOUR SOLUTION


def evaluate_cifar10(model, dataloader, loss_fn=nn.SoftmaxLoss()):
    """
    Computes the test accuracy and loss of the model.

    Args:
        dataloader: Dataloader instance
        model: nn.Module instance
        loss_fn: nn.Module class

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    return epoch_general_cifar10(dataloader, model, loss_fn, opt=None)
    ### END YOUR SOLUTION


### PTB training ###
def epoch_general_ptb(data, model, seq_len=40, loss_fn=nn.SoftmaxLoss(), opt=None,
        clip=None, device=None, dtype="float32"):
    """
    Iterates over the data. If optimizer is not None, sets the
    model to train mode, and for each batch updates the model parameters.
    If optimizer is None, sets the model to eval mode, and simply computes
    the loss/accuracy.

    Args:
        data: data of shape (nbatch, batch_size) given from batchify function
        model: LanguageModel instance
        seq_len: i.e. bptt, sequence length
        loss_fn: nn.Module instance
        opt: Optimizer instance (optional)
        clip: max norm of gradients (optional)

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    if opt is None:
        model.eval()
    else:
        model.train()
    num_batches = (data.shape[0] - 1) // seq_len
    h = None
    accs = []
    losses = []
    for i in tqdm(range(num_batches), desc="Processing Batches", total=num_batches):
        X_batch, y_batch = ndl.data.get_batch(data, i * seq_len, seq_len,
                                             device=device, dtype=dtype)
        y_pred, h = model(X_batch, h)
        if model.seq_model == 'lstm':
            h = (h[0].detach(), h[1].detach())
        else:
            h = h.detach()
        loss = loss_fn(y_pred, y_batch)
        acc = (y_pred.detach().numpy().argmax(axis=1) == y_batch.numpy()).mean()
        loss_val = loss.detach().numpy()
        if opt is not None:
            opt.reset_grad()
            loss.backward()
            opt.step()
        accs.append(acc)
        losses.append(loss_val)
    avg_acc = np.mean(accs)
    avg_loss = np.mean(losses)
    return avg_acc, avg_loss
    ### END YOUR SOLUTION


def train_ptb(model, data, seq_len=40, n_epochs=1, optimizer=ndl.optim.SGD,
          lr=4.0, weight_decay=0.0, loss_fn=nn.SoftmaxLoss(), clip=None,
          device=None, dtype="float32"):
    """
    Performs {n_epochs} epochs of training.

    Args:
        model: LanguageModel instance
        data: data of shape (nbatch, batch_size) given from batchify function
        seq_len: i.e. bptt, sequence length
        n_epochs: number of epochs (int)
        optimizer: Optimizer class
        lr: learning rate (float)
        weight_decay: weight decay (float)
        loss_fn: nn.Module class
        clip: max norm of gradients (optional)

    Returns:
        avg_acc: average accuracy over dataset from last epoch of training
        avg_loss: average loss over dataset from last epoch of training
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)
    avg_acc, avg_loss = 0, 0
    for epoch in range(n_epochs):
        avg_acc, avg_loss = epoch_general_ptb(
            data, model, seq_len, loss_fn, opt, clip, device, dtype
        )
        print(f"Epoch {epoch + 1}/{n_epochs}, Accuracy: {avg_acc}, Loss: {avg_loss}")
    return avg_acc, avg_loss
    ### END YOUR SOLUTION

def evaluate_ptb(model, data, seq_len=40, loss_fn=nn.SoftmaxLoss(),
        device=None, dtype="float32"):
    """
    Computes the test accuracy and loss of the model.

    Args:
        model: LanguageModel instance
        data: data of shape (nbatch, batch_size) given from batchify function
        seq_len: i.e. bptt, sequence length
        loss_fn: nn.Module class

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    return epoch_general_ptb(data, model, seq_len, loss_fn, opt=None,
                             device=device, dtype=dtype)
    ### END YOUR SOLUTION

### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT


def loss_err(h, y):
    """Helper function to compute both loss and error"""
    y_one_hot = np.zeros((y.shape[0], h.shape[-1]))
    y_one_hot[np.arange(y.size), y] = 1
    y_ = ndl.Tensor(y_one_hot)
    return softmax_loss(h, y_).numpy(), np.mean(h.numpy().argmax(axis=1) != y)
