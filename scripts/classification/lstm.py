import tensorflow as tf
from tensorflow.python.keras.layers import Dense


class LSTMcell(tf.keras.Model):
    """
    Creates LSTM cell
    """

    def __init__(self, units):
        """
        Initializes my LSTM cell.
            Args:
                units <int>: units for my weight matrices == seq_len
        """
        super(LSTMcell, self).__init__()

        self.units = units

        # forget gate components
        # bias of forget gate is initialized with ones instead of zeros
        self.forget_gate_x = Dense(
            self.units, bias_initializer='ones', use_bias=True)
        self.forget_gate_h = Dense(self.units, use_bias=False)

        # input gate components
        self.input_gate_x = Dense(self.units, use_bias=True)
        self.input_gate_h = Dense(self.units, use_bias=False)

        # cell memory components
        self.memory_gate_x = Dense(self.units, use_bias=True)
        self.memory_gate_h = Dense(self.units, use_bias=False)

        # out gate components
        self.out_gate_x = Dense(self.units, use_bias=True)
        self.out_gate_h = Dense(self.units, use_bias=False)

    @tf.function
    def call(self, x, states):
        """
        Executes forward pass in my LSTM cell.
            Args:
                x <tensor>: input for the current timestep
                states <tuple<tensor,tensor>>: hidden_output and cell_state from last timestep
            Returns:
                (h_next,c_next) <tensor,tensor>: hidden_output and cell_state for current timestep
        """
        (h, c_prev) = states

        # input gate
        x_i = self.input_gate_x(x)
        h_i = self.input_gate_h(h)
        i = tf.nn.sigmoid(x_i+h_i)

        # forget gate
        x_f = self.forget_gate_x(x)
        h_f = self.forget_gate_h(h)
        f = tf.nn.sigmoid(x_f+h_f)

        # forget old context/cell info
        c_temp = f * c_prev

        # updating cell memory
        x_c = self.memory_gate_x(x)
        h_c = self.memory_gate_h(h)
        c = tf.nn.tanh(x_c+h_c)

        m = c * i
        c_next = m + c_temp

        # output gate
        x_o = self.out_gate_x(x)
        h_o = self.out_gate_h(h)
        o = tf.nn.sigmoid(x_o+h_o)

        # hidden output
        h_next = o * tf.nn.tanh(c_next)

        return (h_next, c_next)


class LSTMlayer(tf.keras.Model):
    """
     Creates an LSTM layer.
    """

    def __init__(self, cell):
        """
        Initializes my LSTM layer.
            Args:
                cell <LSTMcell-object>: the LSTM cell of the layer
        """

        super(LSTMlayer, self).__init__()

        self.cell = cell

        # saving units of my cell
        self.cell_units = self.cell.units

    def zero_states(self, batch_size):
        """
        Creates two tensors filled with zeros
            Args:
                batch_size <int>: size for tensor
            Returns:
                (h,c) <tuple<tensor>>: created tensors
        """
        h = tf.zeros((batch_size, self.cell_units), tf.float32)
        c = tf.zeros((batch_size, self.cell_units), tf.float32)
        return(h, c)

    @tf.function
    def call(self, x):
        """
        Performs a forward pass in my LSTM layer.
            Args:
                x <tensor>: input-batch for all timesteps
            Returns:
            outputs <tensor>: output tensor with outputs from all timesteps
        """

        # get sequence length  ~ time-steps
        seq_len = x.shape[1]

        # initial states from h and c to 0
        states = self.zero_states(x.shape[0])

        # array for hidden states
        hidden_states = tf.TensorArray(dtype=tf.float32, size=seq_len)

        # interating through all timesteps
        for t in tf.range(seq_len):

            x_t = x[:, t, :]

            states = self.cell(x_t, states)

            # only saving the hidden-output not the cell-state
            (h, c) = states
            hidden_states = hidden_states.write(t, h)

        # transpose hidden_states accordingly (batch and time steps switched)
        outputs = tf.transpose(hidden_states.stack(), [1, 0, 2])

        return outputs


class LSTMmodel(tf.keras.Model):
    """
    Creates LSTM model.
    """

    def __init__(self, num_layer=1):
        """
        Initializes my LSTM model.
            Args:
                num_layer <int>: number of LSTM layers: either 1 or 2
        """
        super(LSTMmodel, self).__init__()

        self.num_layer = num_layer

        # my LSTM layer(s)
        self.first_LSTM_layer = LSTMlayer(LSTMcell(units=25))
        if self.num_layer == 2:
            self.second_LSTM_layer = LSTMlayer(LSTMcell(units=25))

        # classification
        self.out = Dense(1, activation="sigmoid")

    @tf.function
    def call(self, x):
        """
        Performs a forward pass in my LSTM model.
            Args:
                x <tensor>: input-batch for all timesteps
            Returns:
                x <tensor>: output tensor with outputs from all timesteps
        """
        x = self.first_LSTM_layer(x)

        # if multi-layer LSTM model
        if self.num_layer == 2:
            x = self.second_LSTM_layer(x)

        x = self.out(x)
        return x
