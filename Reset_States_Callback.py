class Reset_States_Callback(Callback):
    def __init__(self):
        self.counter = 0


    def on_batch_begin(self, batch, logs={}):
        if self.counter % self.max_seq == 0:
            self.model.reset_states()
        self.counter += 1
