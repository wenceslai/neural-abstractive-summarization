import tensorflow as tf

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
