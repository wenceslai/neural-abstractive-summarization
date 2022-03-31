import tensorflow as tf

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
            alpha_weights = tf.squeeze(alpha_weights, axis=-1)
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
        
        concat = tf.concat([tf.squeeze(context, axis=1), hidden_states, tf.squeeze(decoder_input, axis=1)], -1) # [batch_size, 3 * hidden_dim + embedding_dim] # SQUEEZE MODIF
        
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
    
  
    def call(self, X, a, h, c, coverage, context, max_oov, X_mask, use_coverage, use_pgen):
        # context [batch_size, 1, 2 * hidden_dim]
        
        X = self.embedding(X) # [batch_size, 1, embedding_dim]
     
        X = tf.concat([context, X], -1) # [batch_size, 1, 2 * hidden_dim + embedding_dim]
        X = self.W_merge(X) # [batch_size, 1, embedding_dim]

        #h, _, c = self.lstm(inputs=X, initial_state=[h, c]) # lstm returns seq, h, c - when return_sequences=False seq and h are equivalent
        _, h, c = self.lstm(inputs=X, initial_state=[h, c]) # ????

        hidden_states = tf.concat([h, c], axis=-1) # [batch_size, 2 * hidden_dim]
   
        context, alpha_weights = self.attention_module(a, hidden_states, coverage, X_mask, use_coverage)

        # github implementation uses c in addtion to h as input to p gen layer why?
        if use_pgen: p_gen = self.pointer_generator(context, hidden_states, tf.cast(X, tf.float32)) # shape [batches, 1]
        else: p_gen = None

        output = tf.concat([h, tf.squeeze(context, axis=1)], -1)
        output = self.output_linear_1(output)
        vocab_dist = self.output_linear_softmax_2(output)
        
        return vocab_dist, alpha_weights, context, h, c, p_gen

