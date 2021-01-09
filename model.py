import tensorflow as tf
#from tensorflow.keras.layers import Dense#, Input, LSTM, Bidirectional, Embedding, Concatenate, RepeatVector, Activation, Dot
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
        
        ah_concat = tf.concat([a, h], axis=-1)
        
        e = self.dense_linear1(ah_concat)
        e = self.dense_linear2(e)
        e = self.dense_energy(e)
        
        alpha_weights = tf.nn.softmax(e, axis=-1) #scalars!
        
        context = alpha_weights * a
        context = tf.reduce_sum(context, axis=1)
        context = tf.expand_dims(context, axis=1)
        
        return context, alpha_weights


class PointerGenerator(tf.keras.layers.Layer):
    def __init__(self):
        super(PointerGenerator, self).__init__()
        self.dense_sigmoid = tf.keras.layers.Dense(32, activation='sigmoid') #choice of units?

    def call(self, context, h, decoder_input):
        
        concat = tf.concat([tf.squeeze(context), h, decoder_input], axis=-1)
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
        X = tf.concat([context, X], axis=-1)

        h, _, c = self.lstm(inputs=X, initial_state=[h, c])

        y_pred = self.dense_softmax(h)

        return y_pred, context, alpha_weights, h, c,


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

        self.pointer_generator = PointerGenerator()
        self.batch_indexes = [i for i in range(self.batch_size) for _ in range(self.Tx)] # used final dist function 
        self.loss_function = tf.keras.losses.SparseCategoricalCrossentropy() #CE = - tf.reduce_sum(y_true * log(y_pred))
        self.optimizer = tf.keras.optimizers.Adam()


    @tf.function
    def _compute_attention_dist(self, alpha_weights, X):
        attention_dist = tf.zeros([self.batch_size, 1, self.vocab_size])
        
        word_indexes = tf.reshape(X, [-1]) #flattening
        attention_dist[self.batch_indexes, :, word_indexes] += alpha_weights
        
        return attention_dist
        
        
    @tf.function
    def _compute_final_dist(self, p_gen, vocab_dist, attention_dist):
        final_dist = p_gen * vocab_dist + (1 - p_gen) * attention_dist
        return final_dist
        

    @tf.function
    def train_step(self, X, y):
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
                
                p_gen = self.pointer_generator(context, h, tf.cast(decoder_input, dtype=tf.float32))
                attention_dist = self._compute_attention_dist(alpha_weights, X)
                y_pred = self._compute_final_dist(p_gen, vocab_dist, attention_dist)

                y_true = y[:, t] 
                y_true = tf.reshape(y_true, [-1, 1])

                loss += self.loss_function(y_true, y_pred)
                
                decoder_input = y_true #teacher forcing
        
        avg_batch_loss = loss / y.shape[0]

        variables = self.encoder.trainable_weights + self.decoder.trainable_weights

        gradients = tape.gradient(loss, variables)

        self.optimizer.apply_gradients(zip(gradients, variables)) #gradient descent
    
        return avg_batch_loss
    

    def fit(self, epochs, train_batches, val_batches=None):
        print("starting training...")
        total_train_batches = len(train_batches); 

        for epoch in range(epochs): #training loop
            
            t0 = time.time()
            train_epoch_loss = 0; val_epoch_loss = 0

            for batch_i, (X_batch, y_batch) in enumerate(train_batches):
                
                loss = self.train_step(X_batch, y_batch)
                train_epoch_loss += loss
                
                print_status_bar(epoch, "tra", batch_i, total_train_batches, loss, t0)
                
            if val_batches is not None: #validation
                total_val_batches = len(val_batches)
                print("\nvalidating...")
                val_epoch_loss = self.evaluate(val_batches, self.batch_size)
            
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
            
            decoder_input = tf.expand_dims([1] * batch_size, axis=1) #creating <sos> token
            
            a = self.encoder(X)
            
            for t in range(self.Ty): #decoder loop
                
                y_pred, h, c = self.decoder(decoder_input, a, h, c) #return alphas for attention plotting???
                
                y_true = y[:, t] 
                y_true = tf.reshape(y_true, [-1, 1])

                loss += self.loss_function(y_true, y_pred)
                
                y_pred = tf.expand_dims(tf.argmax(y_pred, axis=1), axis=1) #decoding one hot vectors to indexes
                decoder_input = y_pred #passing last prediction
            
            avg_batch_loss = loss / batch_size
            val_epoch_loss += avg_batch_loss

            #print_status_bar(0, "val", batch_i, len(batches), avg_batch_loss, t0)

        return val_epoch_loss


    def beam_search(self):
        pass


    def plot_attention(self):
        pass
    

    def _preprocess(self, line, word_dict):
        indexes = tf.map_fn(lambda word: word_dict[word], elems=line)
        return indexes


    def csv_reader_dataset(self, file_path, word_dict, repeat=1, n_reader=5, n_read_threads=None, shuffle_buffer_size=10000, n_parse_threads=5):
        
        dataset = tf.data.TextLineDataset(file_path).skip(1) #skipping the header row
        dataset = dataset.shuffle(shuffle_buffer_size).repeat(repeat) 
        dataset = dataset.map(self._preprocess, num_parallel_calls=n_parse_threads)

        return dataset.batch(self.batch_size).prefetch(1)



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