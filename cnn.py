import tensorflow as tf


def text_cnn(input_ids,input_y,extras,vocab_size,keep_prob,filter_sizes,num_filters,pos_embedding_size,sequence_length,num_classes,is_training,hidden_size=768,l2_reg_lambda=0.0):
    
    pooled_outputs = []
    num_filters_total=len(filter_sizes)*num_filters
    with tf.variable_scope("char-embedding"):
        W_char= tf.get_variable(name='char-embedding',shape=[vocab_size, hidden_size],initializer=tf.truncated_normal_initializer(stddev=0.02))
    with tf.variable_scope("position-embedding"):
        position_size=2*extras.max_distance+1
        W = tf.get_variable(name='positon_embedding',shape=[position_size, pos_embedding_size],initializer=tf.truncated_normal_initializer(stddev=0.02))
    input_ids=tf.nn.embedding_lookup(W_char,input_ids)
    input_x_p1 = tf.nn.embedding_lookup(W, extras.e1_mas)
    input_x_p2 = tf.nn.embedding_lookup(W, extras.e2_mas)
    x = tf.concat([input_ids, input_x_p1, input_x_p2],2) #batch_size,max_length,hidden_size+pos_embedding_size
    embedded_chars_expanded = tf.expand_dims(x, -1)
    embed_size=input_ids.shape[2].value+pos_embedding_size*2
    initializer=tf.random_normal_initializer(stddev=0.1)
    for i, filter_size in enumerate(filter_sizes):
        with tf.variable_scope("convolution-pooling-%s" % filter_size):
            filter = tf.get_variable("filter-%s" % filter_size, [filter_size, embed_size, 1, num_filters],initializer=initializer)
            conv = tf.nn.conv2d(embedded_chars_expanded, filter, strides=[1, 1, 1, 1], padding="VALID",name="conv")
            conv = tf.contrib.layers.batch_norm(conv, is_training=is_training, scope='cnn_bn_{}'.format(filter_size))
            b = tf.get_variable("b-%s" % filter_size, [num_filters])
            h = tf.nn.relu(tf.nn.bias_add(conv, b),"relu") 
            pooled = tf.nn.max_pool(h, ksize=[1, sequence_length - filter_size + 1, 1, 1],strides=[1, 1, 1, 1], padding='VALID',name="pool")  
            pooled_outputs.append(pooled)
    
    h_pool = tf.concat(pooled_outputs,3)  # shape:[batch_size, 1, 1, num_filters_total]. tf.concat=>concatenates tensors along one dimension.where num_filters_total=num_filters_1+num_filters_2+num_filters_3
    h_pool_flat = tf.reshape(h_pool, [-1,num_filters_total])  # shape should be:[None,num_filters_total]. here this operation has some result as tf.sequeeze().e.g. x's shape:[3,3];tf.reshape(-1,x) & (3, 3)---->(1,9)
    
    if is_training:
        with tf.name_scope("dropout"):
            h_drop = tf.nn.dropout(h_pool_flat, keep_prob=keep_prob)
    else:
        h_drop=h_pool_flat
    h = tf.layers.dense(h_drop, num_filters_total, activation=tf.nn.tanh, use_bias=True)
    
    logits=tf.layers.dense(h,num_classes)
    if is_training:
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=input_y)
        loss = tf.reduce_mean(losses) 
        return loss
    else:
        return logits

class Extra:
    def __init__(self,e1_mas,e2_mas,max_distance=2):
        self.e1_mas=e1_mas
        self.e2_mas=e2_mas
        self.max_distance=max_distance
def create_model(input_ids,
                e1_mas,
                e2_mas,
                vocab_size=500,
                sequence_length=500,
                num_classes=3,
                filter_sizes=[2,3,4],
                num_filters=128,
                pos_embedding_size=32,
                hidden_size=128):
    extras=Extra(e1_mas=e1_mas,e2_mas=e2_mas)
    logits=text_cnn(input_ids=input_ids,
                    input_y=None,
                    vocab_size=vocab_size,
                    extras=extras,
                    keep_prob=1,
                    filter_sizes=filter_sizes,
                    num_filters=num_filters,
                    pos_embedding_size=pos_embedding_size,
                    sequence_length=sequence_length,
                    num_classes=num_classes,
                    is_training=False,
                    hidden_size=hidden_size,l2_reg_lambda=0.0)
    probs=tf.nn.softmax(logits)
    labels=tf.argmax(probs,-1)
    return labels,probs

def get_inputs(batch_size,max_seq_length):
    input_ids=tf.placeholder(dtype=tf.int64,shape=[batch_size,max_seq_length],name='input_ids')
    e1_mas=tf.placeholder(dtype=tf.int64,shape=[batch_size,max_seq_length],name='e1_mas')
    e2_mas=tf.placeholder(dtype=tf.int64,shape=[batch_size,max_seq_length],name='e2_mas')
    return input_ids,e1_mas,e2_mas

