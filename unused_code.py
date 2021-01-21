"""
code that for whatever reason i did not used at the end but could be useful later
"""

"""
def one_hot_lookup(seq): #converts one text
    X = np.zeros((len(seq), VOCAB_SIZE))

    for i, x in enumerate(seq):
        X[i, x] = 1
    return X

def indexes_to_ohe(texts):
    A = np.zeros((texts.shape[0], texts.shape[1], VOCAB_SIZE))

    for i, text_indexes in enumerate(texts):
        for j, index in enumerate(text_indexes):
            A[i, j, index] = 1


def text_to_indexes(s, max_len): #converts one text ot series of indexes from range 0 - vocab_size
    s = s.lower()
    words = re.findall(r"[\w']+|[.,!?;]", s)
    encoded_words = []
    i = 0

    while i < max_len - 1: #one less because of pad token
        if i < len(words):
            word = words[i]
            try: encoded_words.append(word_dict[word])
            except: encoded_words.append(word_dict["<unk>"])
        
        else: encoded_words.append(word_dict["<pad>"])
        i += 1

    return encoded_words + [word_dict["<eos>"]]


def dataset_to_indexes(texts, max_len): #calls text_to_indexes on eache datapoint and creates np array of shape examples x max_len
    A = np.empty((len(texts), max_len))
    A = A.astype('int32')
    
    for i, text in enumerate(texts):
        A[i] = text_to_indexes(text, max_len)
    return A


def create_batches(x, y):
    assert x.shape[0] == x.shape[0]
    
    batches = []
    n_batches = x.shape[0] // BATCH_SIZE

    for i in range(n_batches):
        batch_x = x[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
        batch_y = y[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
        batches.append((batch_x, batch_y))
    
    return batches





    def _word_to_index(self, word):
        if word not in self.word_dict:
            return word["<unk>"]
        else: return self.word_dict[word]

    def preprocess(self, line):
        """
        max_oovs = maximum amount of oov from whole batch
        assign indexes extended indexes to oov words
        in target modify the unk indexes to oov indexes
        """
        
        def handle_X_oovs(word):
            index = self._word_to_index(word)
            if index == 3: # if i == <unk>
                if word not in oov_vocab:
                    oov_cnt += 1
                    oov_vocab[word] = self.vocab_size - 1 + oov_cnt
                return oov_vocab[word]
            else:
                return index

        def handle_y_oovs(word):
            index = self._word_to_index(word)

            if index == 3 and word in oov_vocab:
                return oov_vocab[y_words[i]]
            else:
                return index

        defaults = [""] * 2
        fields = tf.io.decode_csv(line, record_defaults=defaults)
        #x and y strings are already preprocessed
        X_words = tf.strings.split(fields[0]).numpy() #splits by " " #needs decoding??
        y_words = tf.strings.split(fields[1]).numpy()
        print(type(X_words))
        #X_indeces = tf.zeros((self.Tx, ))
        #y_indeces_ext = tf.zeros((self.Ty, )) #Pad token musi byt nula!!!!!!!!!!!
        #X_indeces_ext = tf.zeros((self.Tx, ))
        
        oov_cnt = 0
        oov_vocab = {}

        X_indeces = tf.map_fn(lambda word: self.word_dict[word], X_words)
        X_indeces_ext = tf.map_fn(handle_X_oovs, X_words)
        y_indeces_ext = tf.map_fn(handle_y_oovs, y_words)
        """
        for i in tf.range(self.Tx):
            index = self._word_to_index(X_words[i])
            if index == 3: # if i == <unk>
                if X_words[i] not in oov_vocab:
                    oov_cnt += 1
                    oov_vocab[X_words[i]] = self.vocab_size - 1 + oov_cnt
                X_indeces_ext[i] == oov_vocab[X_words[i]]
            else:
                X_indeces_ext[i] == index

            #X_indeces[i] = index
             
        for i in tf.range(self.Ty):
            index = self.word_to_index(y_words[i])
            if index == 3 and y_words[i] in oov_vocab:
                y_indeces_ext = oov_vocab[y_words[i]]
            else:
                y_indeces_ext = index
        """          
            
        return X_indeces, y_indeces_ext, X_indeces_ext, oov_cnt, oov_vocab


    def csv_reader_dataset(self, file_path, repeat=1, n_reader=5, n_read_threads=None, shuffle_buffer_size=10000, n_parse_threads=5):
        
        dataset = tf.data.TextLineDataset(file_path).skip(1) #skipping the header row
        dataset = dataset.shuffle(shuffle_buffer_size).repeat(repeat) 
        dataset = dataset.map(tf.py_function(self.preprocess), num_parallel_calls=None)

        return dataset.batch(self.batch_size).prefetch(1)


"""