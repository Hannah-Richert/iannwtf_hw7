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
        comb_xh = tf.concat([x,h],axis=-1)

        # forget gate (x(t), h(t-1)) = f(t)
        x_f = self.forget_gate_x(x)
        h_f = self.forget_gate_h(h)
        f = tf.nn.sigmoid(x_f+h_f)

        # forget old context/cell info
        c_temp = f * c_prev

        # input gate (x(t),h(t-1)) = i(t)
        x_i = self.input_gate_x(x)
        h_i = self.input_gate_h(h)
        i = tf.nn.sigmoid(x_i+h_i)

        # updating cell memory
        x_m = self.memory_gate_x(x)
        h_m = self.memory_gate_h(h)
        m = tf.nn.tanh(x_m+h_m)

        g = m * i
        c_next = g + c_temp

        # main output gate
        x_o = self.out_gate_x(x)
        h_o = self.out_gate_h(h)
        o = tf.nn.sigmoid(x_o+h_o)

        # hidden output
        h_next = o * tf.nn.tanh(c_next)

        return (h_next, c_next)

class LSTMlayer(tf.keras.Model):


    def __init__(self,cells):

        super(LSTMlayer,self).__init__()
        # multiple cell layer
        self.cell_one = cells[0]
        self.cell_units = self.cell_one.units
        #self.cell_two = cells[1]

    def zero_states(self,batch_size):
        h = tf.zeros((batch_size,self.cell_units),tf.float32)
        c = tf.zeros((batch_size,self.cell_units),tf.float32)
        return(h,c)

    @tf.function
    def call(self,x,states=None):

        seq_length = x.shape[1]

        states = self.zero_states(x.shape[0])

        for t in tf.range(seq_length):

            x_t = x[:,t,:]

            states = self.cell_one(x_t, states)

        outputs = states

        return outputs



class LSTMmodel(tf.keras.Model):

    def __init__(self):

        super(LSTMmodel,self).__init__()
        # multiple cell layer
        self.first_LSTM_layer = LSTMlayer([LSTMcell(units = 25)])
        #self.second_LSTM_layer = LSTMlayer([LSTMcell(units = 25)])
        #self.dense = Dense(128, activation="sigmoid")
        self.out = Dense(1,activation="sigmoid")

    @tf.function
    def call(self,x):

        (x,y) = self.first_LSTM_layer(x)
        #(x,y) = self.second_LSTM_layer(x)
        #x = self.dense(a)
        x = self.out(x)
        return x
