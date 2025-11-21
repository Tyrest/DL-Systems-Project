"""The module.
"""
from typing import List
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np
from .nn_basic import Parameter, Module


class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return (1 + ops.exp(-x)) ** -1
        ### END YOUR SOLUTION

class RNNCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies an RNN cell with tanh or ReLU nonlinearity.

        Parameters:
        input_size: The number of expected features in the input X
        hidden_size: The number of features in the hidden state h
        bias: If False, then the layer does not use bias weights
        nonlinearity: The non-linearity to use. Can be either 'tanh' or 'relu'.

        Variables:
        W_ih: The learnable input-hidden weights of shape (input_size, hidden_size).
        W_hh: The learnable hidden-hidden weights of shape (hidden_size, hidden_size).
        bias_ih: The learnable input-hidden bias of shape (hidden_size,).
        bias_hh: The learnable hidden-hidden bias of shape (hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.nonlinearity = nonlinearity
        bound = 1 / np.sqrt(hidden_size)
        self.W_ih = Parameter(
            init.rand(
                input_size,
                hidden_size,
                low=-bound,
                high=bound,
                device=device,
                dtype=dtype
            )
        )
        self.W_hh = Parameter(
            init.rand(
                hidden_size,
                hidden_size,
                low=-bound,
                high=bound,
                device=device,
                dtype=dtype
            )
        )
        if bias:
            bound = 1 / np.sqrt(hidden_size)
            self.bias_ih = Parameter(
                init.rand(
                    hidden_size,
                    low=-bound,
                    high=bound,
                    device=device,
                    dtype=dtype
                )
            )
            self.bias_hh = Parameter(
                init.rand(
                    hidden_size,
                    low=-bound,
                    high=bound,
                    device=device,
                    dtype=dtype
                )
            )
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs:
        X of shape (bs, input_size): Tensor containing input features
        h of shape (bs, hidden_size): Tensor containing the initial hidden state
            for each element in the batch. Defaults to zero if not provided.

        Outputs:
        h' of shape (bs, hidden_size): Tensor contianing the next hidden state
            for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        batch_size = X.shape[0]
        if h is None:
            h = Tensor(np.zeros((batch_size, self.hidden_size), dtype=X.dtype), device=X.device)
        h_next = ops.matmul(X, self.W_ih) + ops.matmul(h, self.W_hh)
        if self.bias:
            h_next = h_next + ops.restoredims(self.bias_ih, h_next.shape, 0)
            h_next = h_next + ops.restoredims(self.bias_hh, h_next.shape, 0)
        if self.nonlinearity == 'tanh':
            h_next = ops.tanh(h_next)
        elif self.nonlinearity == 'relu':
            h_next = ops.relu(h_next)
        else:
            raise ValueError("Only tanh and relu supported currently")
        return h_next
        ### END YOUR SOLUTION


class RNN(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies a multi-layer RNN with tanh or ReLU non-linearity to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        nonlinearity - The non-linearity to use. Can be either 'tanh' or 'relu'.
        bias - If False, then the layer does not use bias weights.

        Variables:
        rnn_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, hidden_size) for k=0. Otherwise the shape is
            (hidden_size, hidden_size).
        rnn_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, hidden_size).
        rnn_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (hidden_size,).
        rnn_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (hidden_size,).
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.num_layers = num_layers
        self.rnn_cells = tuple([
            RNNCell(
                input_size if layer == 0 else hidden_size,
                hidden_size,
                bias=bias,
                nonlinearity=nonlinearity,
                device=device,
                dtype=dtype
            )
            for layer in range(num_layers)
        ])
        ### END YOUR SOLUTION

    def forward(self, X, h0=None):
        """
        Inputs:
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h_0 of shape (num_layers, bs, hidden_size) containing the initial
            hidden state for each element in the batch. Defaults to zeros if not provided.

        Outputs
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the RNN, for each t.
        h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        seq_len = X.shape[0]
        if h0 is not None:
            h0 = ops.split(h0, axis=0)
        h_n = []
        layer_input = X
        for layer in range(self.num_layers):
            rnn_cell = self.rnn_cells[layer]
            layer_input = ops.split(layer_input, axis=0)
            h_prev = h0[layer] if h0 is not None else None
            outputs = []
            for t in range(seq_len):
                h_prev = rnn_cell(layer_input[t], h_prev)
                outputs.append(h_prev)
            layer_input = ops.stack(outputs, axis=0)
            h_n.append(h_prev)
        h_n = ops.stack(h_n, axis=0)
        return layer_input, h_n
        ### END YOUR SOLUTION


class LSTMCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, device=None, dtype="float32"):
        """
        A long short-term memory (LSTM) cell.

        Parameters:
        input_size - The number of expected features in the input X
        hidden_size - The number of features in the hidden state h
        bias - If False, then the layer does not use bias weights

        Variables:
        W_ih - The learnable input-hidden weights, of shape (input_size, 4*hidden_size).
        W_hh - The learnable hidden-hidden weights, of shape (hidden_size, 4*hidden_size).
        bias_ih - The learnable input-hidden bias, of shape (4*hidden_size,).
        bias_hh - The learnable hidden-hidden bias, of shape (4*hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        bound = 1 / np.sqrt(hidden_size)
        self.sigmoid = Sigmoid()
        self.W_ih = Parameter(
            init.rand(
                input_size,
                4 * hidden_size,
                low=-bound,
                high=bound,
                device=device,
                dtype=dtype
            )
        )
        self.W_hh = Parameter(
            init.rand(
                hidden_size,
                4 * hidden_size,
                low=-bound,
                high=bound,
                device=device,
                dtype=dtype
            )
        )
        if bias:
            bound = 1 / np.sqrt(hidden_size)
            self.bias_ih = Parameter(
                init.rand(
                    4 * hidden_size,
                    low=-bound,
                    high=bound,
                    device=device,
                    dtype=dtype
                )
            )
            self.bias_hh = Parameter(
                init.rand(
                    4 * hidden_size,
                    low=-bound,
                    high=bound,
                    device=device,
                    dtype=dtype
                )
            )
        ### END YOUR SOLUTION


    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (batch, input_size): Tensor containing input features
        h, tuple of (h0, c0), with
            h0 of shape (bs, hidden_size): Tensor containing the initial hidden state
                for each element in the batch. Defaults to zero if not provided.
            c0 of shape (bs, hidden_size): Tensor containing the initial cell state
                for each element in the batch. Defaults to zero if not provided.

        Outputs: (h', c')
        h' of shape (bs, hidden_size): Tensor containing the next hidden state for each
            element in the batch.
        c' of shape (bs, hidden_size): Tensor containing the next cell state for each
            element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        batch_size = X.shape[0]
        h, c = (None, None) if h is None else h
        if h is None:
            h = Tensor(np.zeros((batch_size, self.hidden_size), dtype=X.dtype), device=X.device)
        if c is None:
            c = Tensor(np.zeros((batch_size, self.hidden_size), dtype=X.dtype), device=X.device)
        combined = ops.matmul(X, self.W_ih) + ops.matmul(h, self.W_hh)
        if self.bias:
            combined = combined + ops.restoredims(self.bias_ih, combined.shape, 0)
            combined = combined + ops.restoredims(self.bias_hh, combined.shape, 0)
        combined = combined.reshape((batch_size, 4, self.hidden_size))
        combined = ops.split(combined, axis=1)
        i = self.sigmoid(combined[0])
        f = self.sigmoid(combined[1])
        g = ops.tanh(combined[2])
        o = self.sigmoid(combined[3])
        c_prime = f * c + i * g
        h_prime = o * ops.tanh(c_prime)
        return h_prime, c_prime
        ### END YOUR SOLUTION


class LSTM(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        """
        Applies a multi-layer long short-term memory (LSTM) RNN to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        bias - If False, then the layer does not use bias weights.

        Variables:
        lstm_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, 4*hidden_size) for k=0. Otherwise the shape is
            (hidden_size, 4*hidden_size).
        lstm_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, 4*hidden_size).
        lstm_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        lstm_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        """
        ### BEGIN YOUR SOLUTION
        self.num_layers = num_layers
        self.lstm_cells = tuple([
            LSTMCell(
                input_size if layer == 0 else hidden_size,
                hidden_size,
                bias=bias,
                device=device,
                dtype=dtype
            )
            for layer in range(num_layers)
        ])
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h, tuple of (h0, c0) with
            h_0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden state for each element in the batch. Defaults to zeros if not provided.
            c0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden cell state for each element in the batch. Defaults to zeros if not provided.

        Outputs: (output, (h_n, c_n))
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the LSTM, for each t.
        tuple of (h_n, c_n) with
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden cell state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        seq_len = X.shape[0]
        if h is not None:
            h0, c0 = h
            h0 = ops.split(h0, axis=0)
            c0 = ops.split(c0, axis=0)
        h_n = []
        c_n = []
        layer_input = X
        for layer in range(self.num_layers):
            lstm_cell = self.lstm_cells[layer]
            layer_input = ops.split(layer_input, axis=0)
            h_prev = h0[layer] if h is not None else None
            c_prev = c0[layer] if h is not None else None
            outputs = []
            for t in range(seq_len):
                h_prev, c_prev = lstm_cell(layer_input[t], (h_prev, c_prev))
                outputs.append(h_prev)
            layer_input = ops.stack(outputs, axis=0)
            h_n.append(h_prev)
            c_n.append(c_prev)
        h_n = ops.stack(h_n, axis=0)
        c_n = ops.stack(c_n, axis=0)
        return layer_input, (h_n, c_n)
        ### END YOUR SOLUTION

class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype="float32"):
        super().__init__()
        """
        Maps one-hot word vectors from a dictionary of fixed size to embeddings.

        Parameters:
        num_embeddings (int) - Size of the dictionary
        embedding_dim (int) - The size of each embedding vector

        Variables:
        weight - The learnable weights of shape (num_embeddings, embedding_dim)
            initialized from N(0, 1).
        """
        ### BEGIN YOUR SOLUTION
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = Parameter(
            init.randn(
                num_embeddings,
                embedding_dim,
                device=device,
                dtype=dtype
            )
        )
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        """
        Maps word indices to one-hot vectors, and projects to embedding vectors

        Input:
        x of shape (seq_len, bs)

        Output:
        output of shape (seq_len, bs, embedding_dim)
        """
        ### BEGIN YOUR SOLUTION
        total_length = np.prod(x.shape)
        one_hot = init.one_hot(
            self.num_embeddings,
            x.reshape((total_length,)),
            device=x.device,
            dtype=self.weight.dtype,
        )
        embedded = ops.matmul(one_hot, self.weight)
        embedded = embedded.reshape((*x.shape, self.embedding_dim))
        return embedded
        ### END YOUR SOLUTION