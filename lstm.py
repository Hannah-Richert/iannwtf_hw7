import tensorflow as tf
from tensorflow.python.keras.layers import Dense

class LSTMcell(tf.keras.Model):


    def __init__(self,units):

        super(LSTMcell,self).__init__()
        self.units= units
        # forget gate components
        self.forget_gate_x = Dense(self.units,bias_initializer='ones')
        self.forget_gate_h = Dense(self.units,use_bias=False)

        # input gate components
        self.input_gate_x = Dense(self.units)
        self.input_gate_h = Dense(self.units,use_bias=False)

        # cell memory components
        self.memory_gate_x = Dense(self.units)
        self.memory_gate_h = Dense(self.units,use_bias=False)

        # out gate components
        self.out_gate_x= Dense(self.units)
        self.out_gate_h= Dense(self.units,use_bias=False)

    @tf.function
    def call(self,x, states):
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


    def __init__(self,cell):

        super(LSTMlayer,self).__init__()
        self.cell = cell
        self.cell_units = self.cell.units


    def zero_states(self,batch_size):
        h = tf.zeros((batch_size,self.cell_units),tf.float32)
        c = tf.zeros((batch_size,self.cell_units),tf.float32)
        return(h,c)

    @tf.function
    def call(self,x,states=None):

        # get sequence length  ~ time-steps
        seq_len = x.shape[1]

        # initial state
        states = self.zero_states(x.shape[0])

        # array for hidden states
        hidden_states = tf.TensorArray(dtype=tf.float32, size=seq_len)

        for t in tf.range(seq_len):

            x_t = x[:,t,:]

            states = self.cell(x_t, states)

            # only saving the hidden-output not the cell-state
            (h,c) = states
            hidden_states = hidden_states.write(t, h)

        outputs = tf.transpose(hidden_states.stack(), [1,0,2])

        return outputs



class LSTMmodel(tf.keras.Model):

    def __init__(self,num_layer=1):

        super(LSTMmodel,self).__init__()
        # two diiferent
        self.num_layer = num_layer
        self.first_LSTM_layer = LSTMlayer(LSTMcell(units = 25))
        if self.num_layer == 2:
            self.second_LSTM_layer = LSTMlayer(LSTMcell(units = 25))
        self.out = Dense(1,activation="sigmoid")

    @tf.function
    def call(self,x):

        x = self.first_LSTM_layer(x)
        if self.num_layer == 2:
            x = self.second_LSTM_layer(x)
        x = self.out(x)
        return x
