import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')

for physical_device in physical_devices:
    tf.config.experimental.set_memory_growth(physical_device, enable=True)

#from tensorflow.keras.layers import Dense#, Input, LSTM, Bidirectional, Embedding, Concatenate, RepeatVector, Activation, Dot
import time
from helper_funcs import print_status_bar, read_csv_dataset
import random
import numpy as np
import os

class Encoder(tf.keras.Model):
    def __init__(self, units, vocab_size, embedding_dim):
        super(Encoder, self).__init__()
        self.units = units
        self.bilstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units, return_sequences=True))
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)

    def call(self, X):
        tf.debugging.check_numerics(tf.cast(X, tf.float16), "nan pred embeddingemPOZOR nan KAMO")
        X = self.embedding(X)
        #tf.debugging.check_numerics(tf.cast(X, tf.float16), "POZOR nan KAMO")
        a = self.bilstm(X)
 
        return a 

class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units, Tx):
        super(BahdanauAttention, self).__init__()
        self.Tx = Tx

        self.W = tf.keras.layers.Dense(1, activation=None, use_bias=False)
        
        self.repeat_vector = tf.keras.layers.RepeatVector(Tx) #can you specify later?

        self.v = self.add_weight(shape=(self.Tx,), initializer='random_normal', trainable=True)
        self.w = self.add_weight(shape=(self.Tx,), initializer='random_normal', trainable=True)
        self.b = self.add_weight(shape=(1), initializer='zeros', trainable=True)

        #def build(self):
        #self.W = self.add_weight(shape=(), initializer='random_normal', trainable=True)
        
        #b_init = tf.zeros_initializer()
        #self.b = tf.Variable(initial_value=b_init(shape=(1), dtype='float32'), trainable=True)

    def call(self, a, h, coverage):
        h = self.repeat_vector(h)
        
        ah_concat = tf.concat([a, h], -1)
        
        #e = self.dense_linear1(ah_concat)
        #e = self.dense_linear2(e)
        #e = self.dense_energy(e)
        weights = self.W(ah_concat)
        #print(weights.shape)
        #print("coverage:", coverage.shape)
        cov_w = coverage * self.w
        cov_w = tf.expand_dims(cov_w, -1)
        #print(cov_w.shape)

        #print(self.v.shape)

        e = tf.nn.tanh(weights + cov_w + self.b)
        #print(e.shape)
        e = tf.squeeze(e) * self.v
        e = tf.expand_dims(e, -1)
        #print(e.shape)

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
        
        concat = tf.concat([tf.squeeze(context), h, tf.squeeze(decoder_input)], -1)
        p_gen = self.dense_sigmoid(concat)

        return p_gen # soft switch for wheter to extract or abstract word


class Decoder(tf.keras.Model):
    def __init__(self, units, vocab_size, embedding_dim, Tx, batch_size):
        super(Decoder, self).__init__()
        self.units = units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.attention_module = BahdanauAttention(10, Tx) #number of units !!!!!!!
        self.lstm = tf.keras.layers.LSTM(units=units, return_state=True)
        self.dense_softmax = tf.keras.layers.Dense(vocab_size, activation='softmax')
        
        self.pointer_generator = PointerGenerator()

        self.batch_size = batch_size

    def call(self, X, a, h, c, context, max_oov):  
        
        X = self.embedding(X)

        context, alpha_weights = self.attention_module(a, h, context)
        #print("context ", context.shape, X.shape)
        X = tf.concat([context, X], -1)

        h, _, c = self.lstm(inputs=X, initial_state=[h, c])

        y_pred = self.dense_softmax(h)

        y_pred = tf.concat([y_pred, tf.zeros([self.batch_size, max_oov])], -1) #appending zeros accounting for oovs
        
        #pointer generator
        p_gen = self.pointer_generator(context, h, tf.cast(X, tf.float32))

        return y_pred, context, alpha_weights, h, c, p_gen


class TextSummarizer:
    def __init__(self, Tx, Ty, batch_size, vocab_size, embedding_dim, a_units, h_units, word_dict, pointer_extract=True):
        self.Tx = Tx
        self.Ty = Ty
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        
        self.a_units = a_units # encoder hidden units
        self.h_units = h_units # decoder hidden and cell state units
        
        self.pointer_extract = pointer_extract
        self.lmbda = 1
        self.gradient_norm = 2
        #self.max_oov = 0
        self.word_dict = word_dict

        # models
        self.encoder = Encoder(a_units, vocab_size, embedding_dim)
        self.decoder = Decoder(h_units, vocab_size, embedding_dim, Tx, batch_size)

        self.batch_indeces = np.array([i for i in range(self.batch_size) for _ in range(self.Tx)]) # used final dist function 
        
        self.loss_function = tf.keras.losses.SparseCategoricalCrossentropy() #CE = - tf.reduce_sum(y_true * log(y_pred))
        self.optimizer = tf.keras.optimizers.Adam()

        self.checkpoint_dir = "/checkpoints"
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")
        self.checkpoint = tf.train.Checkpoint(step=tf.Variable(1), encoder=self.encoder, decoder=self.decoder)
        self.checkpoint_manager = tf.train.CheckpointManager(self.checkpoint, self.checkpoint_dir, max_to_keep=3)


    @tf.function
    def _compute_attention_dist(self, alpha_weights, max_oov):
        #word_indeces = tf.reshape(self.X_batch_indeces_ext, shape=[-1]) #flattening
        alpha_weights = tf.reshape(alpha_weights, [-1])
       # print(type(self.batch_indeces))
        indeces = [[i, j] for i, j in zip(self.batch_indeces, self.X_batch_indeces_ext)]

        attention_dist = tf.zeros([self.batch_size, self.vocab_size + max_oov], tf.float32) #add oov zeros
    
        attention_dist = tf.tensor_scatter_nd_update(attention_dist, indeces, updates=alpha_weights)
        #attention_dist = tf.expand_dims(attention_dist, axis=1)
        return attention_dist
        
        
    @tf.function
    def _compute_final_dist(self, p_gen, vocab_dist, attention_dist):
        #print(vocab_dist.shape, attention_dist.shape)
        final_dist = p_gen * vocab_dist + (1 - p_gen) * attention_dist
        #print(final_dist.numpy())
        return final_dist
        

    @tf.function
    def train_step(self, X, y, y_teacher_force, max_oov):
        """
        compute max_oov from example in batch
        init coverage vec to 0s
        coverage distinguishing between oov in ext vocab
        """
        #tf.print("xx", max_oov)
        
        coverage = tf.zeros([self.batch_size, self.Tx])
        #encoder forward -> a, create initial hidden state
        loss = 0
        h = tf.zeros([self.batch_size, self.h_units])
        c = tf.zeros([self.batch_size, self.h_units])
        #y = tf.one_hot(y, self.vocab_size)
        with tf.GradientTape() as tape:
            
            a = self.encoder(X)
            
            decoder_input = tf.expand_dims([1] * self.batch_size, 1) #?????
            #ADDD OcVER TO FORWARD PASS YOU IDIOTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT
            for t in range(self.Ty): #decoder loop
                
                #print("decoder input ", decoder_input.shape)
                vocab_dist, context, alpha_weights, h, c, p_gen = self.decoder(decoder_input, a, h, c, coverage, max_oov) #attention_dist = alpha_weights
                #y_pred = tf.expand_dims(y_pred, axis=1)
                
                #print("pgenshape", p_gen.shape)
                attention_dist = self._compute_attention_dist(alpha_weights, max_oov) #computes dis across attention
                
                #print("vocab_dist", vocab_dist.numpy())
                y_pred = self._compute_final_dist(p_gen, vocab_dist, attention_dist) #combines both
                #print("y_pred", y_pred.shape)
                coverage += alpha_weights #which words were attended already
                
                y_true = y[:, t] 
                y_true = tf.reshape(y_true, [-1, 1])

                #if tf.math.is_nan(y_pred): 
                #    tf.print("nan value in y_pred in step ", t)
                
                #print("ytrue", y_true.shape)
                coverage_loss = self.lmbda * tf.reduce_sum(tf.math.minimum(alpha_weights, coverage), [0, 1]) #??????// axiss summing over time step and batch dim
                
                #loss += self.loss_function(y_true, y_pred)
                l = self.loss_function(y_true, y_pred)
                #print(l.numpy())
                loss += l
                #print(loss.numpy())
                #print(loss.numpy())
                #loss += coverage_loss
                #print("y_teacherforce", y_teacher_force.shape)
                decoder_input = tf.reshape(y_teacher_force[:, t], [-1, 1]) #teacher forcing
                #print("decoder input", decoder_input.shape)
        
        avg_batch_loss = loss / y.shape[0]

        variables = self.encoder.trainable_weights + self.decoder.trainable_weights

        gradients = tape.gradient(loss, variables)

        #gradients = [(tf.clip_by_norm(grad, self.gradient_norm)) for grad in gradients]

        self.optimizer.apply_gradients(zip(gradients, variables)) #gradient descent
    
        return avg_batch_loss
    

    def fit(self, epochs, train_data, val_data=None, early_stopping_patience=None):
        print("...")
        
        #computing number of val and train batches
        total_train_examples = len(train_data) #TEMRPORARY FIXXX
        total_train_batches = total_train_examples // self.batch_size
        
        if val_data is not None:
            total_val_examples = len(val_data) #TEMRPORARY FIXXX
            total_val_batches = total_val_examples // self.batch_size

        for epoch in range(epochs): #training loop
           
            #shuffling val and train data after each before each epoch
            #random.shuffle(train_data)
            if val_data is not None: random.shuffle(val_data) 

            t0 = time.time()
            
            train_epoch_loss = 0; val_epoch_loss = 0

            #for batch_i, (X_batch, y_batch) in enumerate(train_batches):
            for batch_i in range(0, total_train_batches):
                #initializng empty list for creating batches
                X_batch_ext = []; y_batch = []; oov_cnts = []; oov_vocabs = []; X_batch = []; y_batch_teacher_force = []
                
                #appending data to batches
                for i in range(batch_i * self.batch_size, (batch_i + 1) * self.batch_size):
                     X_ext, y_ext, oov_cnt, oov_vocab = train_data[i]

                     X_batch.append(list(map(self.ext_to_unk, X_ext)))
                     y_batch_teacher_force.append(list(map(self.ext_to_unk, y_ext)))
                     X_batch_ext.append(X_ext)
                     y_batch.append(y_ext)
                     oov_cnts.append(oov_cnt) 
                     oov_vocabs.append(oov_vocab)

                
                #converting lists to array  - NEEEDED TO CONVert???     
                X_batch = np.array(X_batch, np.uint16)
            
                #X_batch_ext = np.array(X_batch_ext)
                self.X_batch_indeces_ext = np.array(X_batch_ext, np.uint16).flatten()
                y_batch_teacher_force = np.array(y_batch_teacher_force, np.uint16)
                y_batch = np.array(y_batch, np.uint16)
                
                max_oov = max(oov_cnts)
                #print(self.max_oov)
                #running one step of mini batch gradient descent
                loss = self.train_step(X_batch, y_batch, y_batch_teacher_force, max_oov)
                
                train_epoch_loss += loss
                
                print_status_bar(epoch, "tra", batch_i, total_train_batches, loss, t0)

            #running validation    
            """
            if val_batches is not None: #validation
                total_val_batches = len(val_batches)
                print("\nvalidating...")
                val_epoch_loss = self.evaluate(val_batches, self.batch_size)
            """
            print(f"\repoch: {epoch+1} DONE!\
                \tavg_train_loss: {train_epoch_loss / total_train_batches}\
                \texec_time: {(time.time() - t0)/60:.2f}min")
                #\tavg_val_loss: {val_epoch_loss / total_val_batches}\
            
            #running callbacks
            if early_stopping_patience is not None:
                pass
                #saving the model
                #self.save_model()


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
    def save_model(self):
        self.checkpoint.step.assign_add(1)
        save_path = self.checkpoint_manager.save()
        print(f"Saved checkpoint for epoch {int(self.checkpoint.step)}: filepath: {save_path}")


    def load_model(self):
        self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
        
        if self.checkpoint_manager.latest_checkpoint:
            print(f"Restored from {self.checkpoint_manager.latest_checkpoint}")
        else:
            print("Initializing from scratch.")
    """



    """
    converted dataset can be read from the csv
    """