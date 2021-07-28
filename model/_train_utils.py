import random
from datetime import datetime
from distutils.dir_util import copy_tree
import os
import time
from helper_funcs import *

def fit(self, epochs, train_data, val_data=None, lr=0.001, use_coverage=False, early_stopping=None, reduce_lr=None, save_freq=None, restore=False, model_name=""):
    print("...")
    
    if restore == False: # backup of saved weights if starting from scratch
        timestamp = datetime.now().strftime("%d_%m_%Y-%H:%M:%S")
        copy_tree(os.path.join(self.save_dir_path, "saved_models"), os.path.join(self.save_dir_path, "saved_models_backup_" + timestamp))

    self.history = {}
    self.history["train_loss"] = []#; self.history["train_acc"] = []
    if val_data is not None: self.history["val_loss"] = []#; self.history["val_acc"] = []

    #computing number of val and train batches
    total_train_batches = len(train_data) // self.batch_size
    if val_data is not None: total_val_batches = len(val_data) // self.batch_size
    train_indeces = list(range(len(train_data)))
    if val_data is not None: val_indeces = list(range(len(val_data)))
    
    self.optimizer.lr.assign(lr)
    print("inital learning rate: ", self.optimizer.lr.numpy())
    
    start_epoch = 0; start_batch_i = 0
    train_time = 0

    if restore:
        # one train step to "warm up the model" - otherwise raises error during opt.set_weights() this does not affect weights because reload the weights
        X_batch, X_batch_indeces_ext, y_batch_teacher_force, y_batch, max_oov, _ = self.get_batch(0, train_data)
        self.train_step(X_batch, y_batch, y_batch_teacher_force, X_batch_indeces_ext,  max_oov, use_coverage)
        
        start_epoch, start_batch_i, train_indeces, self.history["train_loss"], self.history["val_loss"], train_time, early_stopping.prev_loss = self.load_model(return_train_metadata=True, load_optimizer=True)
        #return train_metadata['epoch'], train_metadata['batch_i'], train_metadata['train_indeces'], train_metadata["train_losses"], train_metadata["val_losses"], train_metadata['train_time'], train_metadata['prev_epoch_val_loss']
           
        print("SDKJFLKDSJF", early_stopping.prev_loss)
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
            
            X_batch, X_batch_indeces_ext, y_batch_teacher_force, y_batch, max_oov, _ = self.get_batch(batch_i, train_data)
            
            #running one step of mini batch gradient descent
            loss = self.train_step(X_batch, y_batch, y_batch_teacher_force, X_batch_indeces_ext,  max_oov, use_coverage)

            if batch_i % 10 == 0: # saving train metrics
                self.history["train_loss"].append(loss.numpy())

            if save_freq is not None: # saving the model everty save_freq batches
                if (batch_i - start_batch_i) % save_freq == 0:
                    train_time += time.time() - t0
                    t0 = time.time()
                    self.save_model(save_train_metadata=True, epoch=epoch, batch_i=batch_i, train_indeces=train_indeces, train_time=train_time, prev_epoch_val_loss=early_stopping.prev_loss, save_optimizer=True)

            train_epoch_loss += loss # do i need average loss???
            
            print_status_bar(epoch, epochs, "train", batch_i, total_train_batches, loss, t0)
        
        start_batch_i = 0 # on next iter the start batch will be 0 always
        
        # ---- VALIDATION ----  
        if val_data is not None: #validation
            val_epoch_loss = 0
            val_losses = []
            for batch_i in range(0, total_val_batches):

                X_batch, X_batch_indeces_ext, _, y_batch, max_oov, _ = self.get_batch(batch_i, val_data)
                                    
                loss, _, _ = self.evaluate(X_batch, y_batch, X_batch_indeces_ext, max_oov, use_coverage, compute_rouge=False)#=True if batch_i == total_val_batches - 1 else False)
                val_epoch_loss += loss
                
                if batch_i % 10 == 0:
                    val_losses.append(loss.numpy())
                    #self.history["val_loss"].append(loss.numpy())

                print_status_bar(epoch, epochs, "val", batch_i, total_val_batches, loss, t0)
        
            self.history["val_loss"].append(val_losses) # saving the validation losses
            val_epoch_loss /= total_val_batches

        # average loss over epoch
        train_epoch_loss /= total_train_batches
        
        # SAVING EACH MODEL AND ITS WEIGHTS
        self.save_model()
        copy_tree(os.path.join(self.save_dir_path, "saved_models"), os.path.join(self.save_dir_path, "chp_" + model_name + "_ep" + str(epoch)))
        copy_tree(os.path.join(self.save_dir_path, "saved_checkpoints"), os.path.join(self.save_dir_path, "sm_" + model_name + "_ep" + str(epoch)))
        
        # summary for entire epoch
        print(f"\r\repoch: {epoch} DONE\
            \tavg_train_loss: {train_epoch_loss}\
            \texec_time: {(time.time() - t0)/60:.2f}min", end="")
        if val_data is not None: print(f"\tavg_val_loss: {val_epoch_loss}")
        #if rouge is not None: print(f"\val_rouge_l:{rouge.numpy()}")
        
        # ---- SAVING MODEL ----
        #if early_stopping is None: 

        # ---- CALLBACKS ----
        # stopping training if not improvemnet occured for X epochs
        if early_stopping is not None:
            if val_epoch_loss - early_stopping.min_delta > early_stopping.prev_loss:
                early_stopping.stag_cnt += 1
            else: 
                early_stopping.stag_cnt = 0 
                #self.save_model() # if loss is smaller save model

            if early_stopping.stag_cnt >= early_stopping.patience:
                print(f"{early_stopping.patience} epochs without improvement - halting")
                return self.history
            
            early_stopping.prev_loss = val_epoch_loss

        # reducing learning rate by factor if no improvement exists for X epochs
        """
        if reduce_lr is not None:
            if val_epoch_loss - reduce_lr.min_delta > reduce_lr.prev_loss:
                reduce_lr.stag_cnt += 1
            else: reduce_lr.stag_cnt = 0

            if reduce_lr.stag_cnt >= early_stopping.patience:
                reduce_lr.lr *= reduce_lr.factor
                self.optimizer.lr.assign(reduce_lr.lr)

                print(f"{reduce_lr.patience} epochs without improvement - reducing lr to {self.optimizer.lr.numpy()}")
        """     

    print("...the end")


def get_batch(self, batch_i, dataset):

    def _ext_to_unk(index):
            if index >= self.vocab_size: #############..greater tha equals??
                return 3
            else: return index
    
    X_batch = []; y_batch_teacher_force = []; X_batch_ext = []; y_batch = []; oov_cnts = []; oov_vocabs = []
    for i in range(batch_i * self.batch_size, (batch_i + 1) * self.batch_size):
            X_ext, y_ext, oov_cnt, oov_vocab = dataset[i]

            X_batch.append(list(map(_ext_to_unk, X_ext)))
            y_batch_teacher_force.append(list(map(_ext_to_unk, y_ext)))
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
    max_oov = self.max_global_oov# + 1

    return X_batch, X_batch_indeces_ext, y_batch_teacher_force, y_batch, max_oov, oov_vocabs 