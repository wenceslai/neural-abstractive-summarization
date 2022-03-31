from datetime import datetime
import pickle
import os
import numpy as np

def save_model(self, save_history=True, save_train_metadata=False, epoch=None, batch_i=None, train_indeces=None, train_time=None, early_stopping=None, save_optimizer=False):
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

            train_metadata = {'timestamp' : timestamp, 'epoch' : epoch, 'batch_i' : batch_i, 
            'train_indeces' : train_indeces, 'train_losses' : self.history["train_loss"], 'val_losses' : self.history["val_loss"], 
            'train_time' : train_time, 'early_stopping' : early_stopping}

            with open(os.path.join(self.save_dir_path, 'saved_checkpoints/train_metadata.pickle'), 'wb') as f:
                pickle.dump(train_metadata, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            return None

        print("\nmodel saved")


def load_model(self, return_train_metadata=False, load_optimizer=False, load_prev=None):
    
    checkpoint_filename = 'saved_checkpoints' if load_prev is None else "chp_" + load_prev["model_name"] + "_ep" + str(load_prev["epoch"])

    if load_optimizer:
        
        opt_weights = np.load(os.path.join(self.save_dir_path, checkpoint_filename + '/opt_weights.npy'), allow_pickle=True)
        """
        dummy_vars = self.encoder.trainable_weights + self.decoder.trainable_weights
        zero_grads = [tf.zeros_like(w) for w in dummy_vars]
        self.optimizer.apply_gradients(zip(zero_grads, dummy_vars))
        """
        self.optimizer.set_weights(opt_weights)
        print("optimizer loaded")

    model_filename = 'saved_models' if load_prev is None else "sm_" + load_prev["model_name"] + "_ep" + str(load_prev["epoch"]) 
    
    self.encoder.load_weights(os.path.join(self.save_dir_path, model_filename + '/encoder'))
    self.decoder.load_weights(os.path.join(self.save_dir_path, model_filename + '/decoder'))

    if return_train_metadata: 

        with open(os.path.join(self.save_dir_path, checkpoint_filename + '/train_metadata.pickle'), 'rb') as f:
            
            train_metadata = pickle.load(f)
            print("restoring checkpoint from:", train_metadata['timestamp'], "epoch:", train_metadata['epoch'], "batch_i: ", train_metadata['batch_i'], "t+: ", train_metadata['train_time'])
            
            return train_metadata['epoch'], train_metadata['batch_i'], train_metadata['train_indeces'], train_metadata["train_losses"], train_metadata["val_losses"], train_metadata['train_time'], train_metadata['early_stopping']
        
    print("\nmodel loaded")