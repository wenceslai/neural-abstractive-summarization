import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

#from tensorflow.keras.layers import Dense#, Input, LSTM, Bidirectional, Embedding, Concatenate, RepeatVector, Activation, Dot
import time
from helper_funcs import print_status_bar
import random
import numpy as np

class Encoder(tf.keras.layers.Layer):
    def __init__(self, units, vocab_size, embedding_dim):
        super(Encoder, self).__init__()
        self.units = units
        self.bilstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units, return_sequences=True))
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)

    def call(self, X):
        X = self.embedding(X)
        a = self.bilstm(X)
 
        return a 

class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units, Tx):
        super(BahdanauAttention, self).__init__()
        self.dense_linear1 = tf.keras.layers.Dense(units) #Dense is defaultly set to linear activation (None activ)
        self.dense_linear2 = tf.keras.layers.Dense(units)
        self.dense_energy = tf.keras.layers.Dense(1, activation='relu')
        self.repeat_vector = tf.keras.layers.RepeatVector(Tx) #can you specify later?
        self.dot = tf.keras.layers.Dot(axes=1)

    def call(self, a, h):
        h = self.repeat_vector(h)
        
        ah_concat = tf.concat([a, h], -1)
        
        e = self.dense_linear1(ah_concat)
        e = self.dense_linear2(e)
        e = self.dense_energy(e)
        
        alpha_weights = tf.nn.softmax(e, -1) #scalars!
        
        context = alpha_weights * a
        context = tf.reduce_sum(context, 1)
        context = tf.expand_dims(context, 1)
        
        return context, tf.squeeze(alpha_weights)


class PointerGenerator(tf.keras.layers.Layer):
    def __init__(self):
        super(PointerGenerator, self).__init__()
        self.dense_sigmoid = tf.keras.layers.Dense(1, activation='sigmoid') #choice of units?

    def call(self, context, h, decoder_input):
        
        concat = tf.concat([tf.squeeze(context), h, decoder_input], -1)
        p_gen = self.dense_sigmoid(concat)

        return p_gen #soft switch for wheter to extract or abstract word


class Decoder(tf.keras.layers.Layer):
    def __init__(self, units, vocab_size, embedding_dim, Tx):
        super(Decoder, self).__init__()
        self.units = units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.attention_module = BahdanauAttention(10, Tx) #number of units !!!!!!!
        self.lstm = tf.keras.layers.LSTM(units=units, return_state=True)
        self.dense_softmax = tf.keras.layers.Dense(vocab_size, activation='softmax')

    def call(self, X, a, h, c):  
        X = self.embedding(X)

        context, alpha_weights = self.attention_module(a, h)
        #print("context ", context.shape, X.shape)
        X = tf.concat([context, X], -1)

        h, _, c = self.lstm(inputs=X, initial_state=[h, c])

        y_pred = self.dense_softmax(h)

        return y_pred, context, alpha_weights, h, c,


class TextSummarizer:
    def __init__(self, Tx, Ty, batch_size, vocab_size, embedding_dim, a_units, h_units, word_dict, pointer_extract=True):
        self.Tx = Tx
        self.Ty = Ty
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        
        self.a_units = a_units #encoder hidden units
        self.h_units = h_units #decoder hidden and cell state units
        
        self.pointer_extract = pointer_extract
        self.lmbda = 1
        self.max_oov = 0
        self.word_dict = word_dict

        self.encoder = Encoder(a_units, vocab_size, embedding_dim)
        self.decoder = Decoder(h_units, vocab_size, embedding_dim, Tx)

        self.pointer_generator = PointerGenerator()
        
        self.batch_indeces = np.array([i for i in range(self.batch_size) for _ in range(self.Tx)]) # used final dist function 
        self.X_batch_indeces_ext = None
        
        self.loss_function = tf.keras.losses.SparseCategoricalCrossentropy() #CE = - tf.reduce_sum(y_true * log(y_pred))
        self.optimizer = tf.keras.optimizers.Adam()


    @tf.function
    def _compute_attention_dist(self, alpha_weights):
        #word_indeces = tf.reshape(self.X_batch_indeces_ext, shape=[-1]) #flattening
        alpha_weights = tf.reshape(alpha_weights, [-1])
       # print(type(self.batch_indeces))
        indeces = [[i, j] for i, j in zip(self.batch_indeces, self.X_batch_indeces_ext)]

        attention_dist = tf.zeros([self.batch_size, self.vocab_size + self.max_oov], tf.float32) #add oov zeros
        attention_dist = tf.tensor_scatter_nd_update(attention_dist, indeces, updates=alpha_weights)
        #attention_dist = tf.expand_dims(attention_dist, axis=1)
        return attention_dist
        
        
    @tf.function
    def _compute_final_dist(self, p_gen, vocab_dist, attention_dist):
        #print(vocab_dist.shape, attention_dist.shape)
        final_dist = p_gen * vocab_dist + (1 - p_gen) * attention_dist
        return final_dist
        

    @tf.function
    def train_step(self, X, y, y_teacher_force):
        """
        compute max_oov from example in batch
        init coverage vec to 0s
        coverage distinguishing between oov in ext vocab
        """
        coverage = tf.zeros([self.batch_size, self.Tx])
        #encoder forward -> a, create initial hidden state
        loss = 0
        h = tf.zeros([self.batch_size, self.h_units])
        c = tf.zeros([self.batch_size, self.h_units])
        #y = tf.one_hot(y, self.vocab_size)
        with tf.GradientTape() as tape:
            
            a = self.encoder(X)
            
            decoder_input = tf.expand_dims([1] * self.batch_size, 1) #?????
            
            for t in range(self.Ty): #decoder loop
                #print("decoder input ", decoder_input.shape)
                vocab_dist, context, alpha_weights, h, c = self.decoder(decoder_input, a, h, c) #attention_dist = alpha_weights
                #y_pred = tf.expand_dims(y_pred, axis=1)
                
                vocab_dist = tf.concat([vocab_dist, tf.zeros([self.batch_size, self.max_oov])], -1) 

                p_gen = self.pointer_generator(context, h, tf.cast(decoder_input, tf.float32)) #computes soft switch between copyting and abstracting
                #print("pgenshape", p_gen.shape)
                attention_dist = self._compute_attention_dist(alpha_weights) #computes dis across attention
                y_pred = self._compute_final_dist(p_gen, vocab_dist, attention_dist) #combines both
                
                coverage += alpha_weights #which words were attended already

                y_true = y[:, t] 
                y_true = tf.reshape(y_true, [-1, 1])
                #print("ytrue", y_true.shape)
                coverage_loss = self.lmbda * tf.reduce_sum(tf.math.minimum(alpha_weights, coverage), [0, 1]) #??????// axiss summing over time step and batch dim
                #print("covloss", coverage_loss.numpy())
                loss += self.loss_function(y_true, y_pred)
                #print("loss", loss.numpy())
                loss += coverage_loss
                #print("y_teacherforce", y_teacher_force.shape)
                decoder_input = tf.reshape(y_teacher_force[:, t], [-1, 1]) #teacher forcing
                #print("decoder input", decoder_input.shape)
        avg_batch_loss = loss / y.shape[0]

        variables = self.encoder.trainable_weights + self.decoder.trainable_weights

        gradients = tape.gradient(loss, variables)

        self.optimizer.apply_gradients(zip(gradients, variables)) #gradient descent
    
        return avg_batch_loss
    

    def fit(self, epochs, train_data, val_data=None):
        print("...")
        
        total_train_examples = len(train_data) #TEMRPORARY FIXXX
        total_train_batches = total_train_examples // self.batch_size
        
        if val_data is not None:
            total_val_examples = len(val_data) #TEMRPORARY FIXXX
            total_val_batches = total_val_examples // self.batch_size

        for epoch in range(epochs): #training loop
            #shuffle dataset
            #random.shuffle(train_data)
            if val_data is not None: random.shuffle(val_data) 

            t0 = time.time()
            
            train_epoch_loss = 0; val_epoch_loss = 0

            
            #for batch_i, (X_batch, y_batch) in enumerate(train_batches):
            for batch_i in range(0, total_train_batches):
                X_batch_ext = []; y_batch = []; oov_cnts = []; oov_vocabs = [];
                X_batch = []; y_batch_teacher_force = []
                for i in range(batch_i * self.batch_size, (batch_i + 1) * self.batch_size):
                     X_ext, y_ext, oov_cnt, oov_vocab = train_data[i]
                     X_batch.append(list(map(self.ext_to_unk, X_ext)))
                     y_batch_teacher_force.append(list(map(self.ext_to_unk, y_ext)))
                     X_batch_ext.append(X_ext); y_batch.append(y_ext); oov_cnts.append(oov_cnt); oov_vocabs.append(oov_vocab)
                    
                X_batch = np.array(X_batch)
                #print("maximum" ,max([max(x) for x in X_batch]))
                #X_batch_ext = np.array(X_batch_ext)
                self.X_batch_indeces_ext = np.array(X_batch_ext).flatten()
                y_batch_teacher_force = np.array(y_batch_teacher_force)
                y_batch = np.array(y_batch)
                #self.max_oov = max(oov_cnts)
                
                loss = self.train_step(X_batch, y_batch, y_batch_teacher_force)
                train_epoch_loss += loss
                
                print_status_bar(epoch, "tra", batch_i, total_train_batches, loss, t0)
            """
            if val_batches is not None: #validation
                total_val_batches = len(val_batches)
                print("\nvalidating...")
                val_epoch_loss = self.evaluate(val_batches, self.batch_size)
            """
            print(f"\repoch: {epoch+1} done\
                \tavg_train_loss: {train_epoch_loss / total_train_batches}\
                \tavg_val_loss: {val_epoch_loss / total_val_batches}\
                \texec_time: {(time.time() - t0)/60:.2f}min")
    

    @tf.function
    def evaluate(self, batches, batch_size): #inference time, varible batch_size for few preds
        t0 = time.time()
        val_epoch_loss = 0
        for batch_i, (X, y) in enumerate(batches):
        
            loss = 0
            h = tf.zeros([batch_size, self.h_units])
            c = tf.zeros([batch_size, self.h_units])
            
            decoder_input = tf.expand_dims([1] * batch_size, 1) #creating <sos> token
            
            a = self.encoder(X)
            
            for t in range(self.Ty): #decoder loop
                
                y_pred, h, c = self.decoder(decoder_input, a, h, c) #return alphas for attention plotting???
                
                y_true = y[:, t] 
                y_true = tf.reshape(y_true, [-1, 1])

                loss += self.loss_function(y_true, y_pred)
                
                y_pred = tf.expand_dims(tf.argmax(y_pred, 1), 1) #decoding one hot vectors to indexes
                decoder_input = y_pred #passing last prediction
            
            avg_batch_loss = loss / batch_size
            val_epoch_loss += avg_batch_loss

            #print_status_bar(0, "val", batch_i, len(batches), avg_batch_loss, t0)

        return val_epoch_loss


    def beam_search(self):
        pass


    def plot_attention(self):
        pass


    def ext_to_unk(self, index):
        if index >= self.vocab_size: #############..greater tha equals??
            return self.word_dict["<unk>"]
        else: return index
    
    

    """
    converted dataset can be read from the csv
    """



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