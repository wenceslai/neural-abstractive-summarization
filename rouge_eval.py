import tensorflow as tf
import time
from datetime import datetime
import json
import os
from rouge import Rouge
from model import TextSummarizer
from helper_funcs import *
import random

def rouge_eval(dataset_path, load_prev, save_dir, bw, save_n_samples, use_coverage, batch_size, total_batches, word_dict, index_dict, vocab_size, Tx, Ty, max_global_oov, embedding_dim, a_units, h_units, sp_mode, use_pgen):
    
    # load the test set for eval
    print(f"loading test dataset from {dataset_path}")
    t0 = time.time()

    test_data = read_csv_dataset(dataset_path, Tx, Ty)

    assert len(test_data) > total_batches * batch_size and save_n_samples < total_batches * batch_size
    
    print(f"done in {time.time() - t0}s")

    # load the model

    model = TextSummarizer(Tx, Ty, batch_size, vocab_size, 
                        embedding_dim=embedding_dim, a_units=a_units, h_units=h_units, word_dict=word_dict, index_dict=index_dict, max_global_oov=max_global_oov + 1, save_dir_path=save_dir, sp_mode=sp_mode, use_pgen=use_pgen)

    #model.load_model()
    model.load_model(load_prev=load_prev)

    print(f"evluating on {total_batches * batch_size} examples...")

    random_sample_indices = sorted(random.sample(range(0, total_batches * batch_size - 1), save_n_samples)) # indicies of texts, references and preds saved in output

    t0 = time.time()

    preds = []; refs = []; texts = [];

    for batch_i in range(0, total_batches):

        X_batch, X_batch_indeces_ext, _, y_batch, _, oov_vocab = model.get_batch(batch_i, test_data)    

        #if not use_beam_search:
        #    _, _, _, _, y_preds = model.evaluate(X_batch, y_batch, X_batch_indeces_ext, max_global_oov, use_coverage, compute_rouge=False)
        #    y_preds = y_preds.numpy()
        #elif use_beam_search:
        

        y_preds, _, _, _, = model.beam_decode_batch(X_batch, X_batch_indeces_ext, bw, batch_size, False, block_unk=False)

        #_, _, y_preds = model.evaluate(X_batch, y_batch, X_batch_indeces_ext, max_global_oov, use_coverage, compute_rouge=False)
        
        #y_preds = tf.cast(y_preds, tf.int32)
        #y_preds = y_preds.numpy()
        
        #y_preds = model.beam_decode_batch(X_batch, X_batch_indeces_ext, bw, batch_size, use_coverage=use_coverage, block_unk=True).numpy()
           
        # saving them into text format
        for i in range(batch_size):
         
            pred = model.indeces_to_words(y_preds[i], oov_vocab[i], remove_paddings=True, remove_eos=True, model_file=os.path.join(save_dir, "sp_czechsum_50K_model.model")) # [:-1] to discard the <eos> token
            preds.append(pred)
            
            ref = model.indeces_to_words(y_batch[i], oov_vocab[i], remove_paddings=True, remove_eos=True, model_file=os.path.join(save_dir, "sp_czechsum_50K_model.model"))
            refs.append(ref)

            if i + batch_i * batch_size in random_sample_indices: # linearizing i to position among all of example
                text = model.indeces_to_words(X_batch_indeces_ext[i * Tx: (i + 1) * Tx, 1], oov_vocab[i], remove_paddings=True, model_file=os.path.join(save_dir, "sp_czechsum_50K_model.model"))
                texts.append(text)

        print(f"\r{batch_i + 1}/{total_batches}", end="")

    print(f"done in: {time.time() - t0}s")
    
    # evaluation
    print("evulating on rouge")

    rouge = Rouge()

    #for i in range(len(preds)):
    #    if preds[i] == "":
            
        
    scores = rouge.get_scores(preds, refs, avg=True) # hyps, refs

    # generating output json file
    output = []
    
    output.append(scores)

    output.append({"datetime" : str(datetime.now()), "bw" : bw, "Tx" : Tx, "Ty" : Ty, 
                    "hidden_units_encoder" : a_units, 
                    "hidden_units_decoder" : h_units, 
                    "vocab_size" : vocab_size,
                    "sample_size" : batch_size * total_batches,
                    "saved_summaries" : save_n_samples
                    })

    for i, index in enumerate(random_sample_indices):
        output.append({"text" : texts[i], "ref_summary" : refs[index], "pred_summary" : preds[index]})

    json_object = json.dumps(output, indent=4)

    with open(os.path.join(save_dir, "eval_summ_ " + load_prev["model_name"] + "_" + str(load_prev["epoch"]) + f"_{datetime.now()}.json"), "w") as output_file:
        output_file.write(json_object)

    print("eval done, output file saved")

    return scores



    




    

        



