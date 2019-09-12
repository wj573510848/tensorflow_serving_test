import os

class Config:
    def __init__(self):

        self.max_seq_length=500
        self.batch_size=None
        
        self.release_path='./release'
        self.model_name='test_model_1'
        self.version='1'

        self.cur_dir=os.path.dirname(os.path.abspath(__file__))
