import time
import json_lines
import csv
def print_status_bar(epoch, stage, batch_i, total_batches, loss, t):
    
    bar_len = 24
    width = total_batches // bar_len
    batch_i += 1
    percentage = batch_i / total_batches * 100

    progress_done = ">" * (batch_i // width)
    progress_to_go = "." * (bar_len - (batch_i // width))
    
    print(f"\repoch: {epoch+1} \
        stage: {stage}\
        batch: {batch_i:03d}/{total_batches} \
        [{progress_done}>{progress_to_go}] ({percentage:.1f}%)\tloss: {loss:.5f}\tt:{(time.time() - t) // 60}min {(time.time() - t) % 60:.0f}s", end="")


def json_lines_to_csv(columns, source_file, dest_file):
    with json_lines.open(source_file, 'r') as json_file, open(dest_file, 'w') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()

        for line in json_file:
            writer.writerow(line)
            
#unused functions

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
    words = re.findall(r"[\w]+", s)
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
