from numpy.core.numeric import indices
import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU') # usually just one
for physical_device in physical_devices:
    tf.config.experimental.set_memory_growth(physical_device, enable=True)

import time
from datetime import datetime
import random
import numpy as np
import os
import pickle
from distutils.dir_util import copy_tree

from helper_funcs import print_status_bar, read_csv_dataset
from submodels_defs import *
from rouge import tf_rouge_l

class Encoder(tf.keras.Model):
    def __init__(self, units, vocab_size, embedding_dim):
        super(Encoder, self).__init__()
   
        self.bilstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units, return_sequences=True, return_state=True))
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)

        self.reduce_h = tf.keras.layers.Dense(units, activation='relu')
        self.reduce_c = tf.keras.layers.Dense(units, activation='relu')

    def call(self, X, X_mask):
        
        X = self.embedding(X) # input embedding [batch_size, embed_dim]
        
        a, forward_h, forward_c, backward_h, backward_c = self.bilstm(X, mask=X_mask) # a is concat of a-> and a<- shape=[batch_size, Tx, 2 * a_units]

        #tf.print(tf.reduce_sum(a[:, -1]))
        h = tf.concat([forward_h, backward_h], axis=-1) # [batch_size, 2 * hidden_dim]
        c = tf.concat([forward_c, backward_c], axis=-1) 

        h_reduced = self.reduce_h(h) # [batch_size, hidden_dim]
        c_reduced = self.reduce_c(c)
        
        return a, h_reduced, c_reduced 


class Attention(tf.keras.layers.Layer):
    def __init__(self, Tx, units):
        super(Attention, self).__init__()
        self.Tx = Tx

        self.Wa = tf.keras.layers.Dense(2 * units, use_bias=False)
        self.Wh = tf.keras.layers.Dense(2 * units, use_bias=True)
        self.Wc = tf.keras.layers.Dense(2 * units, use_bias=False)
        self.v = tf.keras.layers.Dense(1, use_bias=False)
        #self.repeat_vector = tf.keras.layers.RepeatVector(Tx) #can you specify later?

    
    def call(self, a, h, coverage, X_mask, use_coverage, use_masking=True):

        encoder_features = self.Wa(a) # [batch_size, Tx, 2 * hidd_dim]
        
        decoder_features = self.Wh(h) # [batch_size, 2 * hidd_dim]
        decoder_features =  tf.expand_dims(decoder_features, axis=1) # adding time axis for broadcasting (alternative for repeat vector)
        #decoder_featurNotice also that these zeros are going directly to the dense layer, which will also eliminate the gradients for a lot of the dense weights. This might overfit longer sequences though if they are few compared to shorter sequences. s = self.repeat_vector(decoder_features)     # [batch_size, 1, 2 * hidd_dim]
        
        if use_coverage:
            coverage = tf.expand_dims(coverage, axis=-1) # ???
            coverage_features = self.Wc(coverage)

            features = encoder_features + decoder_features + coverage_features
        else:
            features = encoder_features + decoder_features
        
        e = self.v(tf.nn.tanh(features)) 
        
        alpha_weights = tf.nn.softmax(e, axis=1) # [batch_size, Tx, 1] each alpha weight is a scalar 
       
        if use_masking:
            alpha_weights = tf.where(X_mask, alpha_weights, tf.zeros_like(alpha_weights)) # masking
            alpha_weights_sum = tf.reduce_sum(alpha_weights, axis=1)
            alpha_weights = tf.squeeze(alpha_weights)
            alpha_weights /= alpha_weights_sum # renormalization
            alpha_weights = tf.expand_dims(alpha_weights, axis=-1)

        context = alpha_weights * a # [batch_size, Tx, 2 * hidden_dim]
        context = tf.reduce_sum(context, 1) 
        context = tf.expand_dims(context, 1) # [batch_size, 1, 2 * hidden_dim]
       
        return context, tf.squeeze(alpha_weights)
        

class PointerGenerator(tf.keras.layers.Layer):
    def __init__(self):
        super(PointerGenerator, self).__init__()
        self.dense_sigmoid = tf.keras.layers.Dense(1, activation='sigmoid') # each p_gen is scalar


    def call(self, context, hidden_states, decoder_input):
        
        concat = tf.concat([tf.squeeze(context), hidden_states, tf.squeeze(decoder_input)], -1) # [batch_size, 3 * hidden_dim + embedding_dim]
        
        p_gen = self.dense_sigmoid(concat) # equivalent to sig(w*h + w*s + w*X + b)

        return p_gen # soft switch for wheter to extract or abstract word in next timestep


class Decoder(tf.keras.Model):
    def __init__(self, units, vocab_size, embedding_dim, Tx, batch_size, encoder_embedding_layer):
        super(Decoder, self).__init__()
        self.units = units
        
        #self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        #sharing weights with encoder embedding layer
        self.embedding = encoder_embedding_layer

        self.attention_module = Attention(Tx, units)
        
        self.lstm = tf.keras.layers.LSTM(units=units, return_state=True)
        
        self.output_linear_1 = tf.keras.layers.Dense(units, activation=None)
        self.output_linear_softmax_2 = tf.keras.layers.Dense(vocab_size, activation='softmax')
        
        self.pointer_generator = PointerGenerator()

        #self.batch_size = batch_size

        self.W_merge = tf.keras.layers.Dense(embedding_dim, activation=None)
    
  
    def call(self, X, a, h, c, coverage, context, max_oov, X_mask, use_coverage):
        # context [batch_size, 1, 2 * hidden_dim]

        X = self.embedding(X) # [batch_size, 1, embedding_dim]
     
        X = tf.concat([context, X], -1) # [batch_size, 1, 2 * hidden_dim + embedding_dim]
        X = self.W_merge(X) # [batch_size, 1, embedding_dim]

        #h, _, c = self.lstm(inputs=X, initial_state=[h, c]) # lstm returns seq, h, c - when return_sequences=False seq and h are equivalent
        _, h, c = self.lstm(inputs=X, initial_state=[h, c]) # ????

        hidden_states = tf.concat([h, c], axis=-1) # [batch_size, 2 * hidden_dim]
   
        context, alpha_weights = self.attention_module(a, hidden_states, coverage, X_mask, use_coverage)

        # github implementation uses c in addtion to h as input to p gen layer why?
        p_gen = self.pointer_generator(context, hidden_states, tf.cast(X, tf.float32)) # shape [batches, 1]

        output = tf.concat([h, tf.squeeze(context)], -1)
        output = self.output_linear_1(output)
        vocab_dist = self.output_linear_softmax_2(output)
        
        return vocab_dist, alpha_weights, context, h, c, p_gen


class TextSummarizer:
    def __init__(self, Tx, Ty, batch_size, vocab_size, embedding_dim, a_units, h_units, word_dict, index_dict, max_global_oov, save_dir_path="", pointer_extract=True):
        # hyperparameters
        self.Tx = Tx # length of input seq
        self.Ty = Ty# length of output seq (inclucding <eos>)
        self.batch_size = batch_size
        self.vocab_size = vocab_size # number of words in vocabulary
        self.max_global_oov = max_global_oov

        self.a_units = a_units # encoder hidden units
        self.h_units = h_units # decoder hidden and cell state units
        
        self.pointer_extract = pointer_extract 
        self.lmbda = 1.0 # regularization parameter for coverage
        self.gradient_norm = 2.0 # for gradient clipping
        
        self.word_dict = word_dict
        self.index_dict = index_dict
        # models
        self.encoder = Encoder(a_units, vocab_size, embedding_dim)
        self.decoder = Decoder(h_units, vocab_size, embedding_dim, Tx, batch_size, self.encoder.embedding)

        self.batch_indeces = np.array([i for i in range(self.batch_size) for _ in range(self.Tx)]) # used final dist function 
        
        #self.loss_function = tf.keras.losses.SparseCategoricalCrossentropy() #CE = - tf.reduce_sum(y_true * log(y_pred))
        self.optimizer = tf.keras.optimizers.Adagrad()
        self.accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
        self.top_k_accuracy = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5)
        
        self.save_dir_path = save_dir_path


    @tf.function
    def _compute_attention_dist(self, alpha_weights, X_batch_indeces_ext, max_oov):
        #print(alpha_weights[0, :])
        attention_dist = tf.zeros([self.batch_size, self.vocab_size + max_oov], tf.float32) # adding zeros for oovs
        
        alpha_weights = tf.reshape(alpha_weights, [-1]) # flattening 
        attention_dist = tf.tensor_scatter_nd_add(attention_dist, indices=X_batch_indeces_ext, updates=alpha_weights)

        return attention_dist
        
        
    @tf.function
    def _compute_final_dist(self, p_gen, vocab_dist, attention_dist):
        
        return p_gen * vocab_dist + (1 - p_gen) * attention_dist
        

    @tf.function
    def _masked_average(self, values, mask):
        seq_lens = tf.reduce_sum(tf.cast(mask, tf.float32), axis=1) # getting length of each sequence in batch
        
        values = tf.where(mask, values, tf.zeros_like(values)) # applying mask

        values = tf.reduce_sum(values, axis=1) / seq_lens # normalization

        return tf.reduce_mean(values) # average over batch

        
    #@tf.function
    def train_step(self, X, y, y_teacher_force, X_batch_indeces_ext, max_oov, use_coverage=False):

        accuracy = 0
        top_k_accuracy = 0
        
        losses = tf.zeros([self.batch_size, 1]) # creating tensor to cancat to first column omitted later
        cov_losses = tf.zeros([self.batch_size, 1])
        
        coverage = tf.zeros([self.batch_size, self.Tx]) # initial coverage vector
        context = tf.zeros([self.batch_size, 1, 2 * self.a_units]) # inital context vector

        #h = tf.zeros([self.batch_size, self.h_units]) # hiden state # init by zeros
        #c = tf.zeros([self.batch_size, self.h_units]) # cell state     
        
        X_mask = tf.expand_dims(tf.not_equal(X, 0), axis=-1)
        y_mask = tf.not_equal(y, 0)

        with tf.GradientTape() as tape:
             
            a, h, c = self.encoder(X, tf.squeeze(X_mask)) # get activations + concatenated hidden states
            
            decoder_input = tf.expand_dims([1] * self.batch_size, 1) # creating <sos> token

            #context, _ = self.decoder.attention_module(a, tf.concat([h, c], -1), coverage, X_mask, use_coverage=False) # initial getting initial value of context vec
            
            for t in range(self.Ty): #decoder loop, iter for each output word
                
                vocab_dist, alpha_weights, context, h, c, p_gen = self.decoder(decoder_input, a, h, c, coverage, context, max_oov, X_mask, use_coverage) #attention_dist = alpha_weights
                
                coverage += alpha_weights # which words were attended already
                
                vocab_dist = tf.concat([vocab_dist, tf.zeros([self.batch_size, max_oov])], -1) #appending zeros accounting for oovs

                attention_dist = self._compute_attention_dist(alpha_weights, X_batch_indeces_ext, max_oov) # adds alpha weights to zeros vec to create dist [batch_size, vocab_size + max_oov]
                
                y_pred = self._compute_final_dist(p_gen, vocab_dist, attention_dist) #combines both
                
                # current time step ground truth
                y_true = y[:, t] 
                y_true = tf.reshape(y_true, [-1, 1])

                curr_step_losses = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
                losses = tf.concat([losses, tf.expand_dims(curr_step_losses, axis=-1)], axis=1)
                
                self.accuracy.update_state(y_true, y_pred)
                accuracy += self.accuracy.result()
                self.accuracy.reset_states()

                self.top_k_accuracy.update_state(y_true, y_pred)
                top_k_accuracy += self.top_k_accuracy.result()
                self.accuracy.reset_states()
                
                if use_coverage: # adding coverage loss = lambda * sum(min(alpha, coverage))
                    curr_step_cov_loss = self.lmbda * tf.reduce_sum(tf.math.minimum(alpha_weights, coverage), [1])
                    cov_losses = tf.concat([cov_losses, tf.expand_dims(curr_step_cov_loss, axis=-1)], axis=1)
                
                #teacher forcing - y without oov indeces
                decoder_input = tf.reshape(y_teacher_force[:, t], [-1, 1]) 
            
            loss = self._masked_average(losses[:, 1:], y_mask)
           
            if use_coverage:
                cov_loss = self._masked_average(cov_losses[:, 1:], y_mask)
                loss += cov_loss


        variables = self.encoder.trainable_weights + self.decoder.trainable_weights
        
        gradients = tape.gradient(loss, variables)
        
        # gradient clipping
        #gradients = [tf.clip_by_norm(grad, self.gradient_norm) if grad is not None else grad for grad in gradients]
        gradients, _ = tf.clip_by_global_norm(gradients, self.gradient_norm)
        
        self.optimizer.apply_gradients(zip(gradients, variables)) #gradient descent

        accuracy_avg = accuracy / self.batch_size
        top_k_accuracy_avg = top_k_accuracy / self.batch_size

        return loss, accuracy_avg, top_k_accuracy_avg
    

    def fit(self, epochs, train_data, val_data=None, lr=0.001, use_coverage=False, early_stopping=None, reduce_lr=None, save_freq=None, restore=False):
        print("...")
        
        if restore == False: # backup of saved weights if starting from scratch
            timestamp = datetime.now()
            timestamp = timestamp.strftime("%d_%m_%Y-%H:%M:%S")
            copy_tree(os.path.join(self.save_dir_path, "saved_models"), os.path.join(self.save_dir_path, "saved_models_backup_" + timestamp))

        self.history = {}
        self.history["train_loss"] = []; self.history["train_acc"] = []
        if val_data is not None: self.history["val_loss"] = []; self.history["val_acc"] = []

        #computing number of val and train batches
        total_train_batches = len(train_data) // self.batch_size
        if val_data is not None: total_val_batches = len(val_data) // self.batch_size

        train_indeces = list(range(len(train_data)))
        val_indeces = list(range(len(val_data)))
        
        self.optimizer.lr.assign(lr)
        print("inital learning rate: ", self.optimizer.lr.numpy())
        
        start_epoch = 0; start_batch_i = 0

        if restore:
            # one train step to "warm up the model" - otherwise raises error during opt.set_weights()
            X_batch, X_batch_indeces_ext, y_batch_teacher_force, y_batch, max_oov, _ = self._get_batch(0, train_data)
            self.train_step(X_batch, y_batch, y_batch_teacher_force, X_batch_indeces_ext,  max_oov, use_coverage)
            
            start_epoch, start_batch_i, train_indeces = self.load_model(return_train_metadata=True, load_optimizer=True)
        
        for epoch in range(start_epoch, epochs):
           
            if restore:
                train_data = train_data[train_indeces]     
                restore = False
            else:
                # shuffling val and train data after each before each epoch
                random.shuffle(train_indeces)
                train_data = train_data[train_indeces]

            if val_data is not None:
                    random.shuffle(val_indeces)
                    val_data = val_data[val_indeces]    
                
            t0 = time.time()
            
            # ---- TRAINING ---- 
            train_epoch_loss = 0
            for batch_i in range(start_batch_i, total_train_batches):
                
                X_batch, X_batch_indeces_ext, y_batch_teacher_force, y_batch, max_oov, _ = self._get_batch(batch_i, train_data)
                
                #running one step of mini batch gradient descent
                loss, accuracy, top_k_accuracy = self.train_step(X_batch, y_batch, y_batch_teacher_force, X_batch_indeces_ext,  max_oov, use_coverage)
    

                if batch_i % 10 == 0: # saving train metrics
                    self.history["train_loss"].append(loss.numpy())
                    self.history["train_acc"].append(accuracy.numpy())

                if save_freq is not None: # saving the model everty save_freq batches
                    if (batch_i - start_batch_i) % save_freq == 0:
                        self.save_model(save_train_metadata=True, epoch=epoch, batch_i=batch_i, train_indeces=train_indeces, save_optimizer=True)

                train_epoch_loss += loss # do i need average loss???
                
                print_status_bar(epoch, epochs, "train", batch_i, total_train_batches, loss, accuracy, top_k_accuracy, t0)
            
            start_batch_i = 0 # on next iter the start batch will be 0 always
            
            # ---- VALIDATION ----  
            if val_data is not None: #validation
                val_epoch_loss = 0
                for batch_i in range(0, total_val_batches):

                    X_batch, X_batch_indeces_ext, _, y_batch, max_oov, _ = self._get_batch(batch_i, val_data)
                                        
                    loss, accuracy, top_k_accuracy, rouge, _ = self.evaluate(X_batch, y_batch, X_batch_indeces_ext, max_oov, use_coverage, compute_rouge=True if batch_i == total_val_batches - 1 else False)
                    val_epoch_loss += loss
                    
                    if batch_i % 10 == 0:
                        self.history["val_loss"].append(loss.numpy())
                        self.history["val_acc"].append(accuracy.numpy())

                    print_status_bar(epoch, epochs, "val", batch_i, total_val_batches, loss, accuracy, top_k_accuracy, t0)
            
            # average loss over epoch
            train_epoch_loss /= total_train_batches
            val_epoch_loss /= total_val_batches

            # summary for entire epoch
            print(f"\r\r\nepoch: {epoch+1} DONE\
                \tavg_train_loss: {train_epoch_loss}\
                \texec_time: {(time.time() - t0)/60:.2f}min", end="")
            if val_data is not None: print(f"\tavg_val_loss: {val_epoch_loss} val_rouge_l:{rouge.numpy()}")
            
            # ---- SAVING MODEL ----
            if early_stopping is None: self.save_model()

            # ---- CALLBACKS ----
            # stopping training if not improvemnet occured for X epochs
            if early_stopping is not None:
                if val_epoch_loss - early_stopping.min_delta > early_stopping.prev_loss:
                    early_stopping.stag_cnt += 1
                else: 
                    early_stopping.stag_cnt = 0 
                    self.save_model() # if loss is smaller save model

                if early_stopping.stag_cnt >= early_stopping.patience:
                    print(f"{early_stopping.patience} epochs without improvement - halting")
                    return self.history
                
                early_stopping.prev_loss = val_epoch_loss

            # reducing learning rate by factor if no improvement exists for X epochs
            if reduce_lr is not None:
                if val_epoch_loss - reduce_lr.min_delta > reduce_lr.prev_loss:
                    reduce_lr.stag_cnt += 1
                else: reduce_lr.stag_cnt = 0

                if reduce_lr.stag_cnt >= early_stopping.patience:
                    reduce_lr.lr *= reduce_lr.factor
                    self.optimizer.lr.assign(reduce_lr.lr)

                    print(f"{reduce_lr.patience} epochs without improvement - reducing lr to {self.optimizer.lr.numpy()}")
                    

        print("training DONE")
        return self.history


    def _get_batch(self, batch_i, dataset):
        X_batch = []; y_batch_teacher_force = []; X_batch_ext = []; y_batch = []; oov_cnts = []; oov_vocabs = []
        for i in range(batch_i * self.batch_size, (batch_i + 1) * self.batch_size):
                X_ext, y_ext, oov_cnt, oov_vocab = dataset[i]

                X_batch.append(list(map(self._ext_to_unk, X_ext)))
                y_batch_teacher_force.append(list(map(self._ext_to_unk, y_ext)))
                X_batch_ext.append(X_ext)
                y_batch.append(y_ext)
                oov_cnts.append(oov_cnt) 
                oov_vocabs.append(oov_vocab)

        #converting lists to array  - NEEEDED TO CONVert???     
        X_batch = np.array(X_batch, np.int32)
        X_batch_indeces_ext = np.array(X_batch_ext, np.int32).flatten()
        y_batch_teacher_force = np.array(y_batch_teacher_force, np.int32)
        y_batch = np.array(y_batch, np.int32)
        
        X_batch_indeces_ext = np.array([[i, j] for i, j in zip(self.batch_indeces, X_batch_indeces_ext)], dtype=np.int32)
        
        #print(X_batch.shape, X_batch_indeces_ext.shape, y_batch_teacher_force.shape, y_batch.shape)

        #max_oov = max(oov_cnts) unbale to use because tf.function - different sized inputs trigger retracing - solution?
        max_oov = self.max_global_oov + 1

        return X_batch, X_batch_indeces_ext, y_batch_teacher_force, y_batch, max_oov, oov_vocabs       


    def evaluate(self, X, y, X_batch_indeces_ext, max_oov, use_coverage=False, compute_rouge=False, return_preds=False):
        
        accuracy = 0
        top_k_accuracy = 0

        losses = tf.zeros([self.batch_size, 1]) # creating tensor to cancat to first column omitted later
        cov_losses = tf.zeros([self.batch_size, 1])
        
        preds = tf.zeros([self.batch_size, 1])
        rouge = None

        coverage = tf.zeros([self.batch_size, self.Tx]) # coverage vector
        context = tf.zeros([self.batch_size, 1, 2 * self.a_units])
        
        X_mask = tf.expand_dims(tf.not_equal(tf.cast(X, tf.int32), 0), axis=-1)
        y_mask = tf.not_equal(tf.cast(y, tf.int32), 0)
               
        a, h, c = self.encoder(X, tf.squeeze(X_mask)) # get activations
            
        decoder_input = tf.expand_dims([1] * self.batch_size, 1) # creating <sos> token
      
        for t in range(self.Ty): #decoder loop, iter for each output word
            
            vocab_dist, alpha_weights, context, h, c, p_gen = self.decoder(decoder_input, a, h, c, coverage, context, max_oov, X_mask, use_coverage) #attention_dist = alpha_weights
            vocab_dist = tf.concat([vocab_dist, tf.zeros([self.batch_size, max_oov])], -1) #appending zeros accounting for oovs
            
            coverage += alpha_weights # which words were attended already

            attention_dist = self._compute_attention_dist(alpha_weights, X_batch_indeces_ext, max_oov) #computes dis across attention
                
            y_pred = self._compute_final_dist(p_gen, vocab_dist, attention_dist) #combines both
                
            # current time step ground truth
            y_true = y[:, t] 
            y_true = tf.reshape(y_true, [-1, 1])
            
            # loss
            curr_step_losses = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
            losses = tf.concat([losses, tf.expand_dims(curr_step_losses, axis=-1)], axis=1)
            
            # adding coverage loss
            if use_coverage: # adding coverage loss = lambda * sum(min(alpha, coverage))
                curr_step_cov_loss = self.lmbda * tf.reduce_sum(tf.math.minimum(alpha_weights, coverage), [1])
                cov_losses = tf.concat([cov_losses, tf.expand_dims(curr_step_cov_loss, axis=-1)], axis=1)

            # acccuracy
            self.accuracy.update_state(y_true, y_pred)
            accuracy += self.accuracy.result()
            self.accuracy.reset_states()

            # top k accuracy - if correct label was in top k argmax
            self.top_k_accuracy.update_state(y_true, y_pred)
            top_k_accuracy += self.top_k_accuracy.result()
            self.accuracy.reset_states()
            
            decoder_input = tf.argmax(y_pred, axis=1) # token with biggest predicted probability
            
            preds = tf.concat([preds, tf.cast(tf.expand_dims(decoder_input, axis=-1), tf.float32)], axis=1)
            
            # if predicted word is oov replacing with <unk> token
            decoder_input = [3 if index > self.vocab_size else index for index in decoder_input] 
            decoder_input = tf.expand_dims(decoder_input, axis=-1)
            
        loss = self._masked_average(losses[:, 1:], y_mask)

        if compute_rouge: rouge = tf_rouge_l(preds[:, 1:], y, 2)

        if use_coverage:
            cov_loss = self._masked_average(cov_losses[:, 1:], y_mask)
            loss += cov_loss

        accuracy_avg = accuracy / self.batch_size
        top_k_accuracy_avg = top_k_accuracy / self.batch_size
           
        return loss, accuracy_avg, top_k_accuracy_avg, rouge, preds[:, 1:]


    def indeces_to_words(self, indeces, oov_vocab, remove_paddings=False):
        transcript = ""
        for index in indeces:
            if index > self.vocab_size - 1:
                transcript += oov_vocab[index]
            elif remove_paddings and index == 0:
                continue
            else:
                transcript += self.index_dict[index]

            transcript += " "  
        return transcript


    def plot_attention(self, alpha_weights):
        pass
    
    @tf.function
    def beam_decode_batch(self, X, X_batch_indeces_ext, bw, batch_size, alpha=0.7):
        
        self.batch_size = batch_size

        # initializing parameters
        coverage = tf.zeros([self.batch_size, self.Tx]) # initial coverage vector
        context = tf.zeros([self.batch_size, 1, 2 * self.a_units]) # inital context vector
        
        X_mask = tf.expand_dims(tf.not_equal(tf.cast(X, tf.int32), 0), axis=-1)
        #prev_log_probs = tf.zeros([batch_size, bw])

        batch_indeces = tf.constant([i for i in range(batch_size) for _ in range(bw)]) # for choosing data in bw * batch_size x -1 format

        # encoder pass       
        a, h, c = self.encoder(X, tf.squeeze(X_mask)) # get activations
            
        decoder_input = tf.expand_dims([1] * batch_size, 1)

        # first decoder step to choose first bw candidates
        y_pred, h, c, coverage, context = self._one_decode_step(decoder_input, a, h, c, coverage, context, self.max_global_oov, X_mask, True, X_batch_indeces_ext) # changing max oov?
        
        self.batch_size *= bw # computing all beams in pralell inside a batch

        top_probs, top_words = tf.nn.top_k(y_pred, bw) # it does not matter whether first probs are logs
        
        prev_log_probs = tf.math.log(top_probs)

        # INITIALIZATION OF PARAMS FOR SEARCH
        # tensors to which next predicted words and their log probs will be stacked along last dim
        top_word_seq = tf.reshape(top_words, [batch_size * bw, 1])
        top_log_probs_seq = tf.reshape(prev_log_probs, [batch_size * bw, 1])

        paddings = tf.constant([[0, 0], [0, 0], [0, self.Ty - (0 + 1)]]) # padding by zeros to be able to concat all variable length sentences
        top_sentences = tf.pad(tf.expand_dims(top_words, axis=-1), paddings) # batch x bw x Ty later batch x all sentences x Ty
        
        top_sentences_log_probs = prev_log_probs # batch_size x number of saved sentences

       

        # at the begining same, will change with different input words
        h = tf.repeat(h, bw, 0)
        c = tf.repeat(c, bw, 0)
        context = tf.repeat(context, bw, 0)
        coverage = tf.repeat(coverage, bw, 0)
        
        # these are same each decode step but needs to be copied for beam
        a = tf.repeat(a, bw, 0)
        X_batch_indeces_ext = tf.repeat(X_batch_indeces_ext, bw, 0)
        X_mask = tf.repeat(X_mask, bw, 0)

        decoder_input = tf.reshape(top_words, [batch_size * bw, 1])

        for t in range(1, self.Ty): #? -1?

            y_pred, h, c, coverage, context = self._one_decode_step(decoder_input, a, h, c, coverage, context, self.max_global_oov, X_mask, True, X_batch_indeces_ext)

            log_probs = tf.math.log(y_pred) # applying logartihm for numerical stability
            
            log_probs = tf.reshape(log_probs, [batch_size, bw, -1]) # batch x bw x vocab_size
            
            #end_token_mask = tf.cast(tf.not_equal(log_probs, 2), tf.int32) # we do not count
            #log_probs *= end_token_mask
            #print(prev_log_probs.shape)
            #print(log_probs.shape)

            prev_log_probs = tf.repeat(tf.expand_dims(prev_log_probs, axis=-1), self.vocab_size + self.max_global_oov, axis=-1) # batch x B x vocab_size to be able to add
            log_probs += prev_log_probs # adding logprobs of best words from previous timestep

            log_probs = tf.reshape(log_probs, [batch_size, bw * (self.vocab_size + self.max_global_oov)]) # batch x bw * vocabsize

            top_log_probs, top_words = tf.nn.top_k(log_probs, bw) # output batch x bw
            prev_log_probs = top_log_probs

            top_words = top_words % self.vocab_size
            
            top_word_seq = tf.concat([top_word_seq, tf.reshape(top_words, [bw * batch_size, -1])], axis=-1) # transforming top_words into batch * bw x 1 to match top_word_seq shape
            
            # store top bw sentences and their probs at each timestep then boolean mask on sentences containing eos token # counting eos itself???
            paddings = tf.constant([[0, 0], [0, 0], [0, self.Ty - (t + 1)]]) # padding by zeros to be able to concat all variable length sentences
            top_word_seq_padded = tf.pad(tf.reshape(top_word_seq, [batch_size, bw, -1]), paddings)
            top_sentences = tf.concat([top_sentences, top_word_seq_padded], axis=1)

            #normalized_top_log_probs = 1 / (t + 1)**alpha * top_sentences #/????
            normalized_top_log_probs = 1 / (t + 1)**alpha * top_log_probs #/????

            top_sentences_log_probs = tf.concat([top_sentences_log_probs, top_log_probs], axis=-1)
        
            # maximizing CONDITIONAL PROBS - multiply by previous ones ???
            top_hidstate_indices = top_words // self.vocab_size
            
            top_hidstate_indices = tf.reshape(top_hidstate_indices, [-1]) # flattening
            #top_hidstate_indices = tf.stack([batch_indeces, top_hidstate_indices], axis=-1) # creating indicies with components (batch, top_hidd_state)
            top_hidstate_indices += bw * batch_indeces

    
            #h = tf.reshape(h, [batch_size, bw, -1]) # getting them into shape to be able two choose along two dimensions batch x b
            h = tf.gather(h, top_hidstate_indices) # out shape should be again batch * bw x hid
            c = tf.gather(c, top_hidstate_indices)

            top_word_seq = tf.gather(top_word_seq, top_hidstate_indices)

            #choosing top logprobs for next round!!!!
            top_log_probs = tf.gather_nd(top_log_probs, tf.stack([batch_indeces, top_hidstate_indices], axis=-1)) # indicies with components (batch, top_hidd_state)

            decoder_input = tf.reshape(top_words, [batch_size * bw, 1])
            
        
        # i am cutting the sequences after eos token after Ty steps, computtationally more expensive but i need full bw * batch_size example not to retrace and have other complications
        # letting <EOS> stay???
        eos_token_mask = tf.cast(tf.math.logical_not(tf.math.reduce_any((tf.equal(top_sentences, 2)), axis=-1)), tf.float32)

        top_sentences_log_probs *= eos_token_mask

        best_sentence_index = tf.math.argmax(top_sentences_log_probs, axis=-1)
    
        top_sentence = tf.gather_nd(top_sentences, tf.stack([[i for i in range(batch_size)], best_sentence_index], axis=-1))

        return top_sentence



    @tf.function        
    def _one_decode_step(self, decoder_input, a, h, c, coverage, context, max_oov, X_mask, use_coverage, X_batch_indeces_ext):

        vocab_dist, alpha_weights, context, h, c, p_gen = self.decoder(decoder_input, a, h, c, coverage, context, max_oov, X_mask, use_coverage) #attention_dist = alpha_weights
        
        vocab_dist = tf.concat([vocab_dist, tf.zeros([self.batch_size, max_oov])], -1) #appending zeros accounting for oovs
        coverage += alpha_weights # which words were attended already
        
        attention_dist = self._compute_attention_dist(alpha_weights, X_batch_indeces_ext, max_oov) #computes dis across attention 
        y_pred = self._compute_final_dist(p_gen, vocab_dist, attention_dist) #combines both

        return y_pred, h, c, coverage, context
    

    def _ext_to_unk(self, index):
        if index >= self.vocab_size: #############..greater tha equals??
            return 3
        else: return index


    def save_model(self, save_history=True, save_train_metadata=False, epoch=None, batch_i=None, train_indeces=None, save_optimizer=False):
        self.encoder.save_weights(os.path.join(self.save_dir_path, 'saved_models/encoder'), save_format='tf')
        self.decoder.save_weights(os.path.join(self.save_dir_path, 'saved_models/decoder'), save_format='tf')
      
        if save_history:
            with open(os.path.join(self.save_dir_path, 'train_history.pickle'), 'wb') as f:
                pickle.dump(self.history, f, protocol=pickle.HIGHEST_PROTOCOL)

        if save_optimizer:
            np.save(os.path.join(self.save_dir_path, 'saved_checkpoints/opt_weights.npy'), np.array(self.optimizer.get_weights(), dtype=np.object))

        if save_train_metadata:
            timestamp = datetime.now()
            timestamp = timestamp.strftime("%d/%m/%Y %H:%M:%S")

            train_metadata = {'timestamp' : timestamp, 'epoch' : epoch, 'batch_i' : batch_i, 'train_indeces' : train_indeces}

            with open(os.path.join(self.save_dir_path, 'saved_checkpoints/train_metadata.pickle'), 'wb') as f:
                pickle.dump(train_metadata, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            return None

        print("model saved")


    def load_model(self, return_train_metadata=False, load_optimizer=False):
        
        if load_optimizer:
            
            opt_weights = np.load(os.path.join(self.save_dir_path, 'saved_checkpoints/opt_weights.npy'), allow_pickle=True)
            """
            dummy_vars = self.encoder.trainable_weights + self.decoder.trainable_weights
            zero_grads = [tf.zeros_like(w) for w in dummy_vars]
            self.optimizer.apply_gradients(zip(zero_grads, dummy_vars))
            """
            self.optimizer.set_weights(opt_weights)
            print("optimizer loaded")

        self.encoder.load_weights(os.path.join(self.save_dir_path, 'saved_models/encoder'))
        self.decoder.load_weights(os.path.join(self.save_dir_path, 'saved_models/decoder'))

        if return_train_metadata:

            with open(os.path.join(self.save_dir_path, 'saved_checkpoints/train_metadata.pickle'), 'rb') as f:
                
                train_metadata = pickle.load(f)
                print("restoring checkpoint from:", train_metadata['timestamp'], "epoch:", train_metadata['epoch'], "batch_i: ", train_metadata['batch_i'])
                return train_metadata['epoch'], train_metadata['batch_i'], train_metadata['train_indeces'] 

        print("model loaded")
    



    