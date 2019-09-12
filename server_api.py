import time
import requests
import numpy as np
import json
import tensorflow as tf
from concurrent.futures import ThreadPoolExecutor,wait

import basic_config

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_boolean('batch_test',False,'是否进行batch测试')
flags.DEFINE_string('batch_type','small','batch大小，small 或者 large')
flags.DEFINE_boolean('concurrent_test',False,'是否进行并发测试')
flags.DEFINE_string('host','0.0.0.0','host')

config=basic_config.Config()

def get_input_data(batch_size):
    # 生成输入数据
    input_ids=np.random.randint(1,200,size=[batch_size,config.max_seq_length],dtype=np.int64)
    e1_mas=np.random.randint(0,2,size=[batch_size,config.max_seq_length],dtype=np.int64)
    e2_mas=np.random.randint(0,2,size=[batch_size,config.max_seq_length],dtype=np.int64)    
    
    input_ids=input_ids.tolist()
    e1_mas=e1_mas.tolist()
    e2_mas=e2_mas.tolist()

    return input_ids,e1_mas,e2_mas

def get_tf_response(batch_size):
    
    input_ids,e1_mas,e2_mas=get_input_data(batch_size)
    host=FLAGS.host
    port='8502'
    version=1
    model_name='test_model'

    server_url = "http://{}:{}/v1/models/{}/versions/{}:predict".format(host, port, model_name,version)

    data = {
            'inputs': {
            'input_ids': input_ids,
            'e1_mas': e1_mas,
            'e2_mas':e2_mas
            }
        }
    headers = {'content-type': 'application/json'}

    t1=time.time()
    r = requests.post(server_url,
                          headers=headers,
                          data=json.dumps(data))
    cost_time=time.time()-t1
    
    return json.loads(r.text),cost_time


def get_flask_response(batch_size):
    
    
    host=FLAGS.host
    port='7123'

    server_url="http://{}:{}".format(host,port)

    input_ids,e1_mas,e2_mas=get_input_data(batch_size)

    data = {
            'inputs': {
            'input_ids': input_ids,
            'e1_mas': e1_mas,
            'e2_mas':e2_mas
            }
        }
    data={
        'data':json.dumps(data)
    }

    t1=time.time()
    r=requests.post(server_url,data=data)
    cost_time=time.time()-t1
    return json.loads(r.text),cost_time

def compare_01(batch_type='small',cycle=10):
    # 第一种比较
    # 循环十次求平均值

    res_tf=[]
    res_flask=[]
    res_flask_nn=[]
    if batch_type=='large':
        batch_range=[1,10,100,1000,10000]
    else:
        batch_range=list(range(1,11,1))
    def batch_test_for_tf():
        print("Test for tf serving...")
        for i in batch_range:
            print("Batch: {}".format(i))
            tmp=0
            for _ in range(cycle):
                _,c_time=get_tf_response(i)
                tmp+=c_time
            tmp=tmp/10
            res_tf.append(tmp)
    
    def batch_test_for_falsk():
        print("Test for flask serving...")
        for i in batch_range:
            print("Batch: {}".format(i))
            tmp01=0
            tmp02=0
            for _ in range(cycle):
                #time.sleep(1)
                f_res,c_time=get_flask_response(i)
                tmp02+=f_res['nn_cost_time']
                tmp01+=c_time
            tmp02=tmp02/10
            tmp01=tmp01/10
            res_flask.append(tmp01)
            res_flask_nn.append(tmp02)
    
    batch_test_for_falsk()
    batch_test_for_tf()
    lines=[]
    title=['序号','batch size','tf-serving','flask','rate(tf-serving/flask)']
    index=0
    for i,j,batch_size in zip(res_tf,res_flask,batch_range):
        index+=1
        tmp=[str(index)]
        tmp.append(str(batch_size))
        tmp.append(str(i*1000))
        tmp.append(str(j*1000))
        tmp.append(str(i/j))
        lines.append(tmp)
    print("|".join(title))
    sep=[" :-: "]*len(title)
    print("|".join(sep))
    for line in lines:
        print("|".join(line))

    
    lines=[]
    title=['序号','batch size','tf-serving','flask','rate(tf-serving/flask)']
    index=0
    for i,j,batch_size in zip(res_tf,res_flask_nn,batch_range):
        index+=1
        tmp=[str(index)]
        tmp.append(str(batch_size))
        tmp.append(str(i*1000))
        tmp.append(str(j*1000))
        tmp.append(str(i/j))
        lines.append(tmp)
    print("|".join(title))
    sep=[" :-: "]*len(title)
    print("|".join(sep))
    for line in lines:
        print("|".join(line))
    
def concurent_test_for_tf_serving(num):
    # num:并发数
    p=ThreadPoolExecutor()
    t1=time.time()
    res=[]
    for i in range(num):
        res.append(p.submit(get_tf_response(1)))
    res=wait(res)
    cost_time=time.time()-t1
    return cost_time
def concurent_test_for_flask_serving(num):
    # num:并发数
    p=ThreadPoolExecutor()
    t1=time.time()
    res=[]
    for i in range(num):
        res.append(p.submit(get_flask_response(1)))
    res=wait(res)
    cost_time=time.time()-t1
    return cost_time

def compare_02():
    # 并发测试
    res_tf=[]
    res_flask=[]

    test_list=[1,10,100,1000,10000]

    def test_tf():
        for i in test_list:
            res_tf.append(concurent_test_for_tf_serving(i))
    def test_flask():
        for i in test_list:
            res_flask.append(concurent_test_for_flask_serving(i))
    test_flask()
    test_tf()
    lines=[]
    title=['序号','并发数','tf-serving','flask','rate(tf-serving/flask)']
    index=0
    for i,j,num in zip(res_tf,res_flask,test_list):
        index+=1
        tmp=[str(index)]
        tmp.append(str(num))
        tmp.append(str(i*1000))
        tmp.append(str(j*1000))
        tmp.append(str(i/j))
        lines.append(tmp)
    print("|".join(title))
    sep=[" :-: "]*len(title)
    print("|".join(sep))
    for line in lines:
        print("|".join(line))


if __name__=="__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    # 进行batch 测试
    if FLAGS.batch_test:
        batch_type=FLAGS.batch_type
        if batch_type!='large':
            batch_type='small'
        compare_01(batch_type)
    # 进行并发测试
    if FLAGS.concurrent_test:
        compare_02()