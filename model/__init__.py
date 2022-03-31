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

from helper_funcs import *
from tfrouge import tf_rouge_l

import sentencepiece as spm

from ._models import Encoder, Decoder

class TextSummarizer:
    
    from ._beam_search import beam_decode_batch, _one_decode_step
    from ._train_loop import fit, get_batch
    from ._backup_utils import save_model, load_model

    def __init__(self, Tx, Ty, batch_size, vocab_size, embedding_dim, a_units, h_units, word_dict, index_dict, max_global_oov, save_dir_path="", use_pgen=True, sp_mode=False):
        # hyperparameters
        self.Tx = Tx # length of input seq
        self.Ty = Ty# length of output seq (inclucding <eos>)
        self.batch_size = batch_size
        self.vocab_size = vocab_size # number of words in vocabulary
        self.max_global_oov = max_global_oov 

        self.a_units = a_units # encoder hidden units
        self.h_units = h_units # decoder hidden and cell state units
        
        self.lmbda = 1.0 # regularization parameter for coverage
        self.gradient_norm = 2.0 # for gradient clipping
        
        self.use_pgen = use_pgen 
        self.sp_mode = sp_mode # whether using sentencepiece bpe or normal word level tokenization

        if not sp_mode:
            self.word_dict = word_dict
            self.index_dict = index_dict

        self.eos = 2
        self.sos = 1
        self.unk = 3
        self.pad = 0

        # models
        self.encoder = Encoder(a_units, vocab_size, embedding_dim)
        self.decoder = Decoder(h_units, vocab_size, embedding_dim, Tx, batch_size, self.encoder.embedding)

        self.batch_indeces = np.array([i for i in range(self.batch_size) for _ in range(self.Tx)]) # used final dist function 
        
        #self.loss_function = tf.keras.losses.SparseCategoricalCrossentropy() #CE = - tf.reduce_sum(y_true * log(y_pred))
        self.optimizer = tf.keras.optimizers.Adagrad()

        #self.accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
        #self.top_k_accuracy = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5)
        
        self.save_dir_path = save_dir_path


    @tf.function
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
                
                vocab_dist, alpha_weights, context, h, c, p_gen = self.decoder(decoder_input, a, h, c, coverage, context, max_oov, X_mask, use_coverage, self.use_pgen) #attention_dist = alpha_weights
                
                coverage += alpha_weights # which words were attended already
                
                vocab_dist = tf.concat([vocab_dist, tf.zeros([self.batch_size, max_oov])], -1) #appending zeros accounting for oovs

                if self.use_pgen: 
                    attention_dist = self._compute_attention_dist(alpha_weights, X_batch_indeces_ext, max_oov) # adds alpha weights to zeros vec to create dist [batch_size, vocab_size + max_oov]
                    y_pred = self._compute_final_dist(p_gen, vocab_dist, attention_dist) #combines both
                
                else:
                    y_pred = vocab_dist

                # current time step ground truth
                y_true = y[:, t] 
                y_true = tf.reshape(y_true, [-1, 1])

                curr_step_losses = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
                losses = tf.concat([losses, tf.expand_dims(curr_step_losses, axis=-1)], axis=1)
                
                #self.accuracy.update_state(y_true, y_pred)
                #accuracy += self.accuracy.result()
                #self.accuracy.reset_states()

                #self.top_k_accuracy.update_state(y_true, y_pred)
                #top_k_accuracy += self.top_k_accuracy.result()
                #self.accuracy.reset_states()
                
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

        #accuracy_avg = accuracy / self.batch_size
        #top_k_accuracy_avg = top_k_accuracy / self.batch_size

        return loss
    

    @tf.function
    def evaluate(self, X, y, X_batch_indeces_ext, max_oov, use_coverage=False, compute_rouge=False):
        
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
            
            vocab_dist, alpha_weights, context, h, c, p_gen = self.decoder(decoder_input, a, h, c, coverage, context, max_oov, X_mask, use_coverage, self.use_pgen) #attention_dist = alpha_weights
            vocab_dist = tf.concat([vocab_dist, tf.zeros([self.batch_size, max_oov])], -1) #appending zeros accounting for oovs
            
            coverage += alpha_weights # which words were attended already

            if self.use_pgen:
                attention_dist = self._compute_attention_dist(alpha_weights, X_batch_indeces_ext, max_oov) #computes dis across attention    
                y_pred = self._compute_final_dist(p_gen, vocab_dist, attention_dist) #combines both
            else:
                y_pred = vocab_dist

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
            #self.accuracy.update_state(y_true, y_pred)
            #accuracy += self.accuracy.result()
            #self.accuracy.reset_states()

            # top k accuracy - if correct label was in top k argmax
            #self.top_k_accuracy.update_state(y_true, y_pred)
            #top_k_accuracy += self.top_k_accuracy.result()
            #self.accuracy.reset_states()
            
            decoder_input = tf.argmax(y_pred, axis=1) # token with biggest predicted probability
            
            preds = tf.concat([preds, tf.cast(tf.expand_dims(decoder_input, axis=-1), tf.float32)], axis=1)
            
            # if predicted word is oov replacing with <unk> token
            """
            unk_mask = tf.greater(decoder_input, self.vocab_size - 1)
            decoder_input = tf.where(tf.logical_not(unk_mask), decoder_input, self.unk)
            #decoder_input = [3 if index > self.vocab_size else index for index in decoder_input] 
            decoder_input = tf.expand_dims(decoder_input, axis=-1)
            """
            decoder_input = self._oov_to_unk(decoder_input)
            decoder_input = tf.expand_dims(decoder_input, axis=-1)

        loss = self._masked_average(losses[:, 1:], y_mask)

        if compute_rouge: rouge = tf_rouge_l(tf.cast(preds[:, 1:], tf.int32), y, 2)

        if use_coverage:
            cov_loss = self._masked_average(cov_losses[:, 1:], y_mask)
            loss += cov_loss

        #accuracy_avg = accuracy / self.batch_size
        #top_k_accuracy_avg = top_k_accuracy / self.batch_size
           
        return loss, rouge, preds[:, 1:]


    def _oov_to_unk(self, decoder_input):
        unk_mask = tf.greater(decoder_input, self.vocab_size - 1)
        decoder_input = tf.where(tf.logical_not(unk_mask), decoder_input, self.unk)
        #decoder_input = [3 if index > self.vocab_size else index for index in decoder_input] 
        
        return decoder_input


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


    def indeces_to_words(self, indeces, oov_vocab, remove_paddings=False, remove_eos=False, model_file=""):
        transcript = ""
        if self.sp_mode: 
            sp = spm.SentencePieceProcessor(model_file=model_file)
            
            oov_indeces = np.where(indeces >= self.vocab_size)
            oovs = indeces[oov_indeces]

            indeces[oov_indeces] = self.unk
            
            transcript = sp.Decode(indeces.tolist())

            for oov in oovs:
                transcript.replace("<unk>", str(oov), 1)

        else:
            for index in indeces:
                if index > self.vocab_size: ##### -1?????
                    transcript += oov_vocab[index]
                elif (remove_paddings and index == self.pad) or (remove_eos and index == self.eos):
                    continue
                else:
                    transcript += self.index_dict[index]

                transcript += " "  
        
        return transcript




    
    



    