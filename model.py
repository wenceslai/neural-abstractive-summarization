import tensorflow as tf
#from tensorflow.keras.layers import Dense#, Input, LSTM, Bidirectional, Embedding, Concatenate, RepeatVector, Activation, Dot
#from keras.models import Model
#import keras
#from tensorflow.keras.backend as K

import time
from helper_funcs import print_status_bar

class Encoder(tf.keras.layers.Layer):
    def __init__(self, units, vocab_size, embedding_dim):
        super(Encoder, self).__init__()
        self.units = units
        self.bilstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units, return_sequences=True))
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)

    def call(self, X):
        X = self.embedding(X)
        a = self.bilstm(X)
        #
        return a 


class AttentionModule(tf.keras.layers.Layer):
    def __init__(self, units, Tx):
        super(AttentionModule, self).__init__()
        self.dense_linear1 = tf.keras.layers.Dense(units) #Dense is defaultly set to linear activation (None activ)
        self.dense_linear2 = tf.keras.layers.Dense(units)
        self.dense_energy = tf.keras.layers.Dense(1, activation='relu')
        self.repeat_vector = tf.keras.layers.RepeatVector(Tx) #can you specify later?
        self.dot = tf.keras.layers.Dot(axes=1)

    def call(self, a, h):
        h = self.repeat_vector(h)
        
        ah_concat = tf.concat([a, h], axis=-1)
        
        e = self.dense_linear1(ah_concat)
        e = self.dense_linear2(e)
        e = self.dense_energy(e)
        
        alpha_weights = tf.nn.softmax(e, axis=-1) #scalars!
    
        #alpha_weights = tf.transpose(alpha_weights, perm=[0, 2, 1])
        #context = tf.tensordot(alpha_weights, a, axes=[1, 2])
        
        context = alpha_weights * a
        context = tf.reduce_sum(context, axis=1)
        context = tf.expand_dims(context, axis=1)
        
        return context


class Decoder(tf.keras.layers.Layer):
    def __init__(self, units, vocab_size, embedding_dim, Tx):
        super(Decoder, self).__init__()
        self.units = units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.attention_module = AttentionModule(10, Tx) #number of units !!!!!!!
        self.lstm = tf.keras.layers.LSTM(units=units, return_state=True)
        self.dense_softmax = tf.keras.layers.Dense(vocab_size, activation='softmax')

    def call(self, X, a, h, c):  
        X = self.embedding(X)

        context = self.attention_module(a, h)

        #print("context ", context.shape, X.shape)
        X = tf.concat([context, X], axis=-1)

        h, _, c = self.lstm(inputs=X, initial_state=[h, c])

        y_pred = self.dense_softmax(h)

        return y_pred, h, c


class TextSummarizer:
    def __init__(self, Tx, Ty, batch_size, vocab_size, embedding_dim, a_units, h_units):
        self.Tx = Tx
        self.Ty = Ty
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        
        self.a_units = a_units #encoder hidden units
        self.h_units = h_units #decoder hidden and cell state units
        
        self.encoder = Encoder(a_units, vocab_size, embedding_dim)
        self.decoder = Decoder(h_units, vocab_size, embedding_dim, Tx)

        #self.loss_function = tf.keras.losses.CategoricalCrossentropy() # CE = - tf.reduce_sum(y_true * log(y_pred))
        self.loss_function = tf.keras.losses.SparseCategoricalCrossentropy()
        self.optimizer = tf.keras.optimizers.Adam()

    @tf.function
    def train_step(self, X, y):
        #encoder forward -> a, create initial hidden state
        loss = 0
        h = tf.zeros([self.batch_size, self.h_units])
        c = tf.zeros([self.batch_size, self.h_units])
        
        #y = tf.one_hot(y, self.vocab_size)
        
        with tf.GradientTape() as tape:
            
            a = self.encoder(X)
            
            #decoder_input = tf.zeros([batch_size, self.vocab_size]) #first input for decoder
            #decoder_input = decoder_input[:, 1].assign(1) #token value 1 defined in data_preprocessing
            decoder_input = tf.expand_dims([1] * self.batch_size, 1) #?????
            
            for t in range(self.Ty): #decoder loop
                #print("decoder input ", decoder_input.shape)
                y_pred, h, c = self.decoder(decoder_input, a, h, c)
                #y_pred = tf.expand_dims(y_pred, axis=1)
                
                y_true = y[:, t] 
                y_true = tf.reshape(y_true, [-1, 1])

                #print("y_pred", y_pred.shape)
                #print("slice", y_true.shape)

                loss += self.loss_function(y_true, y_pred)
                
                decoder_input = y_true #teacher forcing
        
        avg_batch_loss = loss / y.shape[0]

        variables = self.encoder.trainable_weights + self.decoder.trainable_weights

        gradients = tape.gradient(loss, variables)

        self.optimizer.apply_gradients(zip(gradients, variables)) #gradient descent
    
        return avg_batch_loss
    

    def fit(self, batches, epochs):
        total_batches = len(batches)
        
        for epoch in range(epochs):
            epoch_loss = 0
            t0 = time.time()
            for batch_i, (X_batch, y_batch) in enumerate(batches):
                
                batch_loss = self.train_step(X_batch, y_batch)
                epoch_loss += batch_loss
                
                print_status_bar(epoch, batch_i, total_batches, batch_loss)

            print(f"epoch: {epoch} done\t avg_loss: {epoch_loss}\texec_time: {time.time() - t0}")
    
    #def infer():

    #def plot_attention():

    #def beam_infer():

    #print_stats():

    
"""
class AttentionModel:
    def __init__(self, Tx, Ty, vocab_size, a_units=32, h_units=32):
        self.a_units = a_units #size of the activation of Pre Attention bilstm
        self.h_units = h_units #size of the activation of Post Attention LSTM
        self.Tx = Tx #input length
        self.Ty = Ty #output max_length
        self.vocab_size = vocab_size

    def attention_module(self, a, h):

        h = RepeatVector(self.Tx)
        #concatenation
        v = Concatenate(axis=-1)([h, a])
        #energies
        e_1 = Dense(10, activation="tanh")(v) 

        e_2 = Dense(1, activation="relu")(e_1)
        #alpha weights
        alphas = Activation("sigmoid")(e_2)

        context = Dot(axes=1)(alphas, a)

        return context

    def encoder_decoder(self):
        #encoder
        X = Input(shape=(self.Tx, self.vocab_size))
        y_shift = Input(shape=(self.Ty, self.vocab_size))

        c0 = K.zeros((self.h_units, ))
        h0 = K.zeros((self.h_units, ))
        h = h0
        c = c0
        outputs = []
        #EMBEDDING?
        a = Bidirectional(LSTM(self.a_units, return_sequence=True))(X)
        #decoder
        for t in range(self.Ty):

            context = self.attention_module(a, h)  
            #EMBEDDING?
            decoder_input = Concatenate(axis=-1)([context, y_shift[t]])

            h, _, c = LSTM(units=self.h_units, return_state=True)(inputs=decoder_input, initial_state=[h, c])

            y_pred = Dense(self.vocab_size, activation="softmax")
            outputs.append(y_pred)

        return Model(inputs=[X, y_shift], outputs=[outputs])
"""