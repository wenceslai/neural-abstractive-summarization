import time
import json_lines
import csv
import re
import numpy as np

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


def json_lines_to_csv_dataset(columns, source_file, dest_file, word_dict, vocab_size, Tx, Ty):
    "writes dataset in form X, y, oov_cnt, oov_dict"
    def word_to_index(word):
        if word not in word_dict:
            return word_dict["<unk>"]
        else: return word_dict[word]

    with json_lines.open(source_file, 'r') as json_file, open(dest_file, 'w') as csv_file:
        #writer = csv.DictWriter(csv_file, fieldnames=columns, extrasaction='ignore')
        writer = csv.writer(csv_file)
        
        for line in json_file: #type(line) == dict
            oov_cnt = 0
            oov_vocab = {} #token index to word
            
            X_words = re.findall(r"[\w']+|[.,!?;]", line[columns[0]].lower())[:Tx]
            y_words = re.findall(r"[\w']+|[.,!?;]", line[columns[1]].lower())[:Ty]

            for i in range(Tx):
                if i < len(X_words):
                    index = word_to_index(X_words[i])
                    if index == 3:
                        if X_words[i] not in oov_vocab:
                            oov_cnt += 1
                            oov_vocab[X_words[i]] = vocab_size - 1 + oov_cnt
                        X_words[i] = oov_vocab[X_words[i]]
                    else:
                        X_words[i] = index
                else:
                    X_words.append(word_dict["<pad>"])

            for i in range(Ty):
                if i < len(y_words):
                    index = word_to_index(y_words[i])
                    if index == 3 and y_words[i] in oov_vocab:
                        y_words[i] = oov_vocab[y_words[i]]
                    else:
                        y_words[i] = index 
                else:
                    y_words.append(word_dict["<pad>"])

            line = X_words + y_words + [oov_cnt]

            for key, value in oov_vocab.items():
                line.append(key)
                line.append(value)
            
            writer.writerow(line)


def read_csv_dataset(source_file, word_dict, vocab_size, Tx, Ty):
    with open(source_file) as csv_file:
        reader = csv.reader(csv_file)
        dataset = []

        for row in reader:
            X = np.array(row[0:Tx]).astype(np.uint16)
            y = np.array(row[Tx:Tx + Ty]).astype(np.uint16)
            oov_cnt = int(row[Tx + Ty])

            oov_dict = {}

            for i in range(Tx + Ty + 1, len(row) - 1, 2):
                
                oov_dict[row[i + 1]] = row[i]

            dataset.append((X, y, oov_cnt, oov_dict))

        return dataset

    
def json_lines_to_csv_old(columns, source_file, dest_file):
    with json_lines.open(source_file, 'r') as json_file, open(dest_file, 'w') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()

        for line in json_file: #type(line) == dict
            for column_name in columns:
                line[column_name] = ' '.join(re.findall(r"[\w']+|[.,!?;]", line[column_name].lower()))
            writer.writerow(line)


            
#unused functions

