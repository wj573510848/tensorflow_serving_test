from flask import Flask,request
import tensorflow as tf
import time
import json

import basic_config
import cnn

class nn_model:
    def __init__(self):
        self._build()
    
    def _build(self):
        config=basic_config.Config()
        session_config = tf.ConfigProto(allow_soft_placement=True,
                                        log_device_placement=False)
        session_config.gpu_options.allow_growth = True
        g = tf.Graph()
        with g.as_default():
            with tf.device(None):
                self.sess=tf.Session(config=session_config)
                self.input_ids=tf.placeholder(dtype=tf.int64,shape=[config.batch_size,config.max_seq_length],name='input_ids')
                self.e1_mas=tf.placeholder(dtype=tf.int64,shape=[config.batch_size,config.max_seq_length],name='e1_mas')
                self.e2_mas=tf.placeholder(dtype=tf.int64,shape=[config.batch_size,config.max_seq_length],name='e2_mas')
                self.labels,self.probs=cnn.create_model(
                    input_ids=self.input_ids,
                e1_mas=self.e1_mas,
                e2_mas=self.e2_mas
                                      )
                self.sess.run(tf.global_variables_initializer())
    
    def predict(self,input_ids,e1_mas,e2_mas):
        t1=time.time()
        feed_dict={
            self.input_ids:input_ids,
            self.e1_mas:e1_mas,
            self.e2_mas:e2_mas
        }
        labels,probs=self.sess.run([self.labels,self.probs],feed_dict=feed_dict)
        cost_time=time.time()-t1

        res={
            'labels':labels,
            'probs':probs
        }
        return res,cost_time

model=nn_model()

app=Flask(__name__)

@app.route('/',methods=['POST'])
def index():
    data=request.form.get('data')
    if isinstance(data,str):
        data=json.loads(data)
    data=data['inputs']
    input_ids=data['input_ids']
    e1_mas=data['e1_mas']
    e2_mas=data['e2_mas']
    nn_res=model.predict(input_ids,e1_mas,e2_mas)
    res={
        'labels':nn_res[0]['labels'].tolist(),
        'probs':nn_res[0]['probs'].tolist(),
        'nn_cost_time':nn_res[1]

    }
    return json.dumps(res)
