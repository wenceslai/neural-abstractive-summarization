import time
import json_lines
import csv
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def print_status_bar(epoch, epochs, stage, batch_i, total_batches, loss, t):
    
    bar_len = 24
    width = total_batches // bar_len # how many batches for one progress tick
    #batch_i += 1
    percentage = batch_i / total_batches * 100
    progress_done = ">" * (batch_i // width) 
    progress_to_go = "." * (bar_len - (batch_i // width))
    
    print(f"\repoch: {epoch}/{epochs} \
        stage: {stage}\
        batch: {batch_i:03d}/{total_batches} \
        [{progress_done}>{progress_to_go}] ({percentage:.1f}%)\tloss: {loss:.5f}\tt+:{(time.time() - t) // 60:.0f}:{(time.time() - t) % 60:.0f}s", end="")

def create_merge_pieces(text, max_len, sp):

    #r = r"[\w'.,!?;\"]+"
    r = r"[\w]+|[.,!?;\"]"

    total_pieces = []

    words = re.findall(r, text)

    for word in words:
        pieces = sp.EncodeAsPieces(word)
        n_pieces = len(list(filter(lambda c: c != '▁', pieces)))
        
        if n_pieces >= max_len:
            s = ""
            for piece in pieces:
                print(piece)
                s += piece
            
            pieces = [s]

        total_pieces += pieces

    print(total_pieces)
    ids = sp.PieceToId(total_pieces)
    return total_pieces, ids


def json_lines_to_csv_dataset_sentencepiece(columns, source_file, dest_file, vocab_size, Tx, Ty, max_global_oov, max_pieces=None, max_Tx=1e5, max_Ty=1e5):
    #max_subwords = 
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(model_file='sp_czechsum_50K_model.model') # either 30K or 50K vocab

    with json_lines.open(source_file, 'r') as json_file, open(dest_file, 'w') as csv_file:

        writer = csv.writer(csv_file)

        for line in json_file:
            X_sent = line[columns[0]].lower()
            y_sent = line[columns[1]].lower()
            
            #r = "[#;|\'@#$\^*():\[\]\{\}]"
            r = r"[^\w.,!?;\" ]"

            X_sent = re.sub(r, "", X_sent)
            y_sent = re.sub(r, "", y_sent)

            X_pieces = sp.EncodeAsPieces(X_sent)
            y_pieces = sp.EncodeAsPieces(y_sent)

            if len(X_pieces > max_Tx) or len(y_pieces) > max_Ty: continue

            X_ids = sp.EncodeAsIds(X_sent)[:Tx]
            y_ids = sp.EncodeAsIds(y_sent)[:Ty]
            #print(X_pieces, X_ids, sp.Decode(X_ids), "\n\n\n")
            #print(len(X_sent), len(X_pieces), len(X_ids))
            oov_cnt = 0
            oov_vocab = {}
            
            eos_added = False
            X_len = len(X_ids)
            for i in range(Tx): # -1 for eos token
                if i < X_len:
                    if X_ids[i] == sp.unk_id():
                        if X_pieces[i] not in oov_vocab and oov_cnt < max_global_oov:
                            oov_cnt += 1
                            oov_vocab[X_pieces[i]] = vocab_size + oov_cnt

                        if X_pieces[i] in oov_vocab:
                            X_ids[i] = oov_vocab[X_pieces[i]]
                        
                        else:
                            X_ids[i] = sp.unk_id()
                
                elif not eos_added:
                    X_ids.append(sp.eos_id())
                    eos_added = True

                else:
                    X_ids.append(sp.pad_id())
            
            if not eos_added:
                X_ids[-1] = sp.eos_id()

            eos_added = False
            y_len = len(y_ids)

            for i in range(Ty):
                if i < y_len:
                    if y_ids[i] == sp.unk_id():
                        if y_pieces[i] in oov_vocab:
                            y_ids[i] = oov_vocab[y_pieces[i]]
                        
                        else: y_ids[i] = sp.unk_id()
                
                elif not eos_added:
                    y_ids.append(sp.eos_id())
                    eos_added = True

                else:
                    y_ids.append(sp.pad_id())
            
            if not eos_added:
                y_ids[-1] = sp.eos_id()


            line = X_ids + y_ids + [oov_cnt]

            for key, value in oov_vocab.items():
                line.append(key)
                line.append(value)
            
            writer.writerow(line)


def json_lines_to_csv_dataset(columns, source_file, dest_file, word_dict, vocab_size, Tx, Ty, max_global_oov):
    "writes dataset in form X, y, oov_cnt, oov_dict"
    def word_to_index(word):
        if word not in word_dict:
            return word_dict["<unk>"]
        else: return word_dict[word]

    #Ty -= 1 # one token reserved for <eos>

    with json_lines.open(source_file, 'r') as json_file, open(dest_file, 'w') as csv_file:
        #writer = csv.DictWriter(csv_file, fieldnames=columns, extrasaction='ignore')
        writer = csv.writer(csv_file)
        
        for line in json_file: #type(line) == dict
            oov_cnt = 0
            oov_vocab = {} #token index to word
            
            X_words = re.findall(r"[\w']+|[.,!?;\"]", line[columns[0]].lower())[:Tx]
            y_words = re.findall(r"[\w']+|[.,!?;\"]", line[columns[1]].lower())[:Ty]
                
            for i in range(Tx):
                if i < len(X_words):
                    index = word_to_index(X_words[i])
                    if index == 3:
                        if X_words[i] not in oov_vocab and oov_cnt < max_global_oov:
                            oov_cnt += 1
                            oov_vocab[X_words[i]] = vocab_size + oov_cnt #discarded -1
                        
                        if X_words[i] in oov_vocab:
                            X_words[i] = oov_vocab[X_words[i]]

                        else: X_words[i] = index
                    else:
                        X_words[i] = index
                else:
                    X_words.append(word_dict["<pad>"])

            eos_added = False
            y_len = len(y_words)

            for i in range(Ty):
                if i < y_len:
                    index = word_to_index(y_words[i])
                    if index == 3 and y_words[i] in oov_vocab:
                        y_words[i] = oov_vocab[y_words[i]]
                    else:
                        y_words[i] = index 
                elif not eos_added:
                    eos_added = True
                    y_words.append(word_dict["<eos>"])
                else:
                    y_words.append(word_dict["<pad>"])
            
            if not eos_added:
                y_words[-1] = word_dict["<eos>"]

            #y_words.append(word_dict["<eos>"])

            line = X_words + y_words + [oov_cnt]

            for key, value in oov_vocab.items():
                line.append(key)
                line.append(value)
            
            writer.writerow(line)


def read_csv_dataset(source_file, Tx, Ty):
    with open(source_file) as csv_file:
        reader = csv.reader(csv_file)
        dataset = []

        for row in reader:
            X = np.array(row[0:Tx]).astype(np.uint16)
            y = np.array(row[Tx:Tx + Ty]).astype(np.uint16)
            #print(len(row), len(X), len(y), Tx + Ty + 1, len(row) - 1)
            oov_cnt = int(row[Tx + Ty])

            oov_dict = {}
        
            for i in range(Tx + Ty + 1, len(row) - 1, 2):
                oov_dict[int(row[i + 1])] = row[i]

            dataset.append((X, y, oov_cnt, oov_dict))

        return np.array(dataset, dtype=object)

    
def json_lines_to_csv_old(columns, source_file, dest_file):
    with json_lines.open(source_file, 'r') as json_file, open(dest_file, 'w') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()

        for line in json_file: #type(line) == dict
            for column_name in columns:
                line[column_name] = ' '.join(re.findall(r"[\w']+|[.,!?;\"]", line[column_name].lower()))
            writer.writerow(line)


def json_lines_to_txt(source_file, dest_file, dataset_len=None, proportion=1.0):
  
  max_i = None if dataset_len is not None else proportion * dataset_len

  with json_lines.open(source_file, 'r') as json_file, open(dest_file, 'w') as output:
    for i, line in enumerate(json_file):

        text = line["text"].replace("\n", " ")

        sentences = split_text_to_sentences(text)
        for sentence in sentences:
            output.write(sentence.lower() + "\n")
        if max_i is not None and i >= max_i:
            print(f"used {i+1} sentences")
            break


def split_text_to_sentences(text):
    sentences = []
    prev_end_i = 0
    for i in range(1, len(text) - 1):
        if re.match(r"[.?!] [A-Z]", text[i-1:i+2]):
            sentences.append(text[prev_end_i:i])
            prev_end_i = i + 1
    
    sentences.append(text[prev_end_i:])
    return sentences


def text_len_stats(text_lens, T=None, oov_cnts=None, max_oov=None):
    to_test = [(text_lens, T)]
    if oov_cnts is not None: to_test.append((oov_cnts, max_oov))

    for dist, bound in to_test:
        
        ig, ax = plt.subplots(1, 3, figsize=(48,16))
        
        minval = np.min(dist)
        Q1, Q2, Q3 = np.percentile(dist, [25, 50, 75])
        maxval = np.max(dist)

        sns.histplot(dist, bins=100, ax=ax[0])
        ax[0].set_xlim(0, Q3 + 2.5 * (Q3 - Q1))
        sns.boxplot(dist, ax=ax[1])
        
        print("5num summary: ", minval, Q1, Q2, Q3, maxval)
        
        outliers = 0
        for l in dist:
            if l < Q1 - 1.5 * (Q3 - Q1) or l > Q3 + 1.5 * (Q3 - Q1): outliers += 1

        print("num of outlier: ", outliers)

        if bound is not None:
            deltas = []
            for l in dist: 
                if l > bound: 
                    deltas.append(l - bound)

            l = len(dist)
            print("proportion of sentences that are longer than the cutoff")
            print("delta > 1 ", len(deltas) / l)
            print("delta < 5: ", len([delta for delta in deltas if delta < 5]) / l)
            print("5 < delta < 10: ", len([delta for delta in deltas if 5 < delta < 10]) / l)
            print("10 < delta < 20: ", len([delta for delta in deltas if 10 < delta < 20]) / l)
            print("20 < delta < 40: ", len([delta for delta in deltas if 20 < delta < 40]) / l)
            print("40 < delta < 80: ", len([delta for delta in deltas if 40 < delta < 80]) / l)
            print("delta > 80: ", len([delta for delta in deltas if delta > 80]) / l)

        sns.histplot(deltas, ax=ax[2])


def fix_text(text):
    if type(text) == str:
        text = text.split()

    fixed_text = ""

    sentence_end_flag = False
    open_quotes_flag = False
    prev_quote_flag = False

    sentence_enders = ['.', '!', '?']
    other_punc = [',', ';']
    quotes = ['"', '\'']

    for i, token in enumerate(text):
        if i == 0:
            fixed_text += token.capitalize()

        elif token not in sentence_enders + other_punc + quotes:
            if not prev_quote_flag:
                if sentence_end_flag == False:
                    fixed_text += " " + token
                else:
                    fixed_text += " " + token.capitalize()
                    sentence_end_flag = False
            else:
                if sentence_end_flag == False:
                    fixed_text += token
                else:
                    fixed_text += token.capitalize()
                    sentence_end_flag = False
                
                prev_quote_flag = False

        elif token in sentence_enders:
            fixed_text += token
            sentence_end_flag = True
        
        elif token in quotes:
            if not open_quotes_flag:
                fixed_text += " " + token
                open_quotes_flag = True
                prev_quote_flag = True
            else:
                fixed_text += token
                open_quotes_flag = False

        elif token in other_punc:
            fixed_text += token
    
    return fixed_text


class CallbackReduceLearningRate:
    def __init__(self, patience=3, factor=0.1, min_delta=0, cooldown=0):
        self.patience = patience
        self.factor = factor
        self.min_delta = min_delta
        self.cooldown = cooldown
        self.prev_loss = 999999
        self.lr = 0.001


class CallbackEarlyStopping:
    def __init__(self, patience=3, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.stag_cnt = 0
        self.prev_loss = 999999
