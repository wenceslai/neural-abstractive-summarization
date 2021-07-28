import tensorflow as tf

def beam_decode_batch(self, X, X_batch_indeces_ext, bw, batch_size, use_coverage, block_unk, alpha=0.7):
        # TRY TO FIND WHICH WORDS ARE PREDICTED IN top wordS! TRY smaller batch size to see
        self.batch_size = batch_size

        NEG_INF = -1e12

        # initializing parameters
        coverage = tf.zeros([batch_size, self.Tx]) # initial coverage vector
        context = tf.zeros([batch_size, 1, 2 * self.a_units]) # inital context vector
        X_mask = tf.expand_dims(tf.not_equal(X, 0), axis=-1)
        batch_indeces = tf.constant([i for i in range(batch_size) for _ in range(bw)]) # for choosing data in bw * batch_size x -1 format
        
        # encoder pass       
        a, h, c = self.encoder(X, tf.squeeze(X_mask)) # get activations
            
        decoder_input = tf.expand_dims([self.sos] * batch_size, axis=1)

        # first decoder step to choose first bw candidates
        y_pred, h, c, coverage, context = self._one_decode_step(decoder_input, a, h, c, coverage, context, self.max_global_oov, X_mask, use_coverage, X_batch_indeces_ext) # changing max oov?
        
        self.batch_size *= bw # computing all beams in pralell inside a batch

        top_probs, top_words = tf.nn.top_k(y_pred, k=bw, sorted=False)
        print(top_probs.numpy())
        prev_log_probs = tf.math.log(top_probs)

        # INITIALIZATION OF PARAMS FOR SEARCH
        # tensors to which next predicted words and their log probs will be stacked along last dim
        top_word_seq = tf.reshape(top_words, [batch_size * bw, 1]) # sentences of current beams

        paddings = tf.constant([[0, 0], [0, 0], [0, self.Ty - (0 + 1)]]) # padding by zeros to be able to concat all variable length sentences
        top_sentences = tf.pad(tf.expand_dims(top_words, axis=-1), paddings) # batch x bw x Ty later batch x all sentences x Ty
        top_sentences_log_probs = prev_log_probs # holding log probs for all predicted sentences, batch_size x number of saved sentences
        
        is_complete_seq_mask = tf.zeros(top_sentences_log_probs.shape, tf.float32) # holds whether the sequence is complete or not i.e. has eos token at the end
       
        # at the begining same, will change with different input words
        
        h = tf.repeat(h, bw, 0)
        c = tf.repeat(c, bw, 0)
        context = tf.repeat(context, bw, 0)
        coverage = tf.repeat(coverage, bw, 0)
        
        # these are same each decode step but needs to be copied for beam
        a = tf.repeat(a, bw, 0)
        X_batch_indeces_ext = tf.repeat(X_batch_indeces_ext, bw, 0)
        X_mask = tf.repeat(X_mask, bw, 0)
        

        #h = tf.tile(h, [bw, 1])

        decoder_input = tf.reshape(top_words, [batch_size * bw, 1])

        Sigma = tf.zeros([batch_size, 1], tf.float32)

        for t in range(1, self.Ty): 
            
            y_pred, h, c, coverage, context = self._one_decode_step(decoder_input, a, h, c, coverage, context, self.max_global_oov, X_mask, use_coverage, X_batch_indeces_ext)
           
            log_probs = tf.math.log(y_pred) # applying logartihm for numerical stability
            
            log_probs = tf.reshape(log_probs, [batch_size, bw * (self.vocab_size + self.max_global_oov)])

            prev_log_probs = tf.repeat(prev_log_probs, self.vocab_size + self.max_global_oov, axis=-1) # batch x bw x vocab_size to be able to add
            
            log_probs += prev_log_probs # adding logprobs of best words from previous timestep
            
            """
            #log_probs = tf.reshape(log_probs, [batch_size, bw, -1]) # batch x bw x vocab_size
            #prev_log_probs = tf.repeat(tf.expand_dims(prev_log_probs, axis=-1), self.vocab_size + self.max_global_oov, axis=-1) # batch x bw x vocab_size to be able to add
            #log_probs += prev_log_probs # adding logprobs of best words from previous timestep
            #log_probs = tf.reshape(log_probs, [batch_size, bw * (self.vocab_size + self.max_global_oov)]) # batch x bw * vocabsize
            
            if block_unk: # set unk token probs to minimum
                indices = [[j, self.unk + i * (self.vocab_size + self.max_global_oov)] for j in range(batch_size) for i in range(bw)]
                updates = [tf.float32.min for _ in range(batch_size * bw)]
                log_probs = tf.tensor_scatter_nd_update(log_probs, indices, updates)
            """
            pad_mask = tf.tile(tf.concat([tf.ones([batch_size, 1]), tf.zeros([batch_size, self.vocab_size - 1 + self.max_global_oov])], axis=-1), [1, bw])
            log_probs += NEG_INF * pad_mask

            top_log_probs, top_indices = tf.nn.top_k(log_probs, bw, sorted=False) # output batch x bw
            
            top_words = top_indices % (self.vocab_size + self.max_global_oov)
            print(top_words.numpy()[0, :])# top_log_probs.numpy()[0, :])
            top_word_seq = tf.concat([top_word_seq, tf.reshape(top_words, [bw * batch_size, -1])], axis=-1) # transforming top_words into batch * bw x 1 to match top_word_seq shape
            
            is_eos = tf.cast(tf.equal(top_words, self.eos), tf.float32) # check if token is <eos>
            #print(is_eos.shape)
            Sigma += tf.reshape(tf.reduce_sum(is_eos, axis=1), [batch_size, 1])

            is_complete_seq_mask = tf.concat([is_complete_seq_mask, is_eos], axis=-1)
             
            paddings = tf.constant([[0, 0], [0, 0], [0, self.Ty - (t + 1)]]) # padding by zeros to be able to concat all variable length sentences
            top_word_seq_padded = tf.pad(tf.reshape(top_word_seq, [batch_size, bw, -1]), paddings)
            top_sentences = tf.concat([top_sentences, top_word_seq_padded], axis=1)
            
            #normalized_top_log_probs = (5 + t)**alpha / (5 + 1)**alpha * top_sentences #/????
            normalized_top_log_probs = 1 / (t + 1)**alpha * top_log_probs
            top_sentences_log_probs = tf.concat([top_sentences_log_probs, normalized_top_log_probs], axis=-1)
        
            # choosing params for next timestep
            top_beams_indices = top_indices // (self.vocab_size + self.max_global_oov)
            top_beams_indices = tf.reshape(top_beams_indices, [-1]) # flattening
            top_beams_indices += bw * batch_indeces

            h = tf.gather(h, top_beams_indices) # out shape should be again batch * bw x hid
            c = tf.gather(c, top_beams_indices)
            coverage = tf.gather(coverage, top_beams_indices)
            context = tf.gather(context, top_beams_indices)
            top_word_seq = tf.gather(top_word_seq, top_beams_indices)

            top_log_probs += NEG_INF * is_eos# sabotaging the probability of beams contatinig that predicted eos token to not pre expanded further

            prev_log_probs = top_log_probs

            decoder_input = tf.reshape(top_words, [batch_size * bw, 1])

        self.batch_size = batch_size # correcting batch size

        top_sentences_log_probs += 1e30 * is_complete_seq_mask # choosing only from completed sentences
       
        # PLUS TRY SUBTRACT FROM OTHERS
        best_sentence_index = tf.math.argmax(top_sentences_log_probs, axis=-1)
    
        top_sentence = tf.gather_nd(top_sentences, tf.stack([[i for i in range(batch_size)], best_sentence_index], axis=-1))

        return top_sentence, Sigma, top_sentences[0]

@tf.function
def _one_decode_step(self, decoder_input, a, h, c, coverage, context, max_oov, X_mask, use_coverage, X_batch_indeces_ext):

    vocab_dist, alpha_weights, context, h, c, p_gen = self.decoder(decoder_input, a, h, c, coverage, context, max_oov, X_mask, use_coverage) #attention_dist = alpha_weights
    
    vocab_dist = tf.concat([vocab_dist, tf.zeros([self.batch_size, max_oov])], -1) #appending zeros accounting for oovs
    coverage += alpha_weights # which words were attended already
    
    attention_dist = self._compute_attention_dist(alpha_weights, X_batch_indeces_ext, max_oov) #computes dis across attention 
    y_pred = self._compute_final_dist(p_gen, vocab_dist, attention_dist) #combines both

    return y_pred, h, c, coverage, context