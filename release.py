import tensorflow as tf
import os
import shutil

import cnn
import basic_config


def release_model():
    config=basic_config.Config()
    tf.logging.info("Release model...")
    tf.logging.info("Model name:{}".format(config.model_name))
    tf.logging.info('Version:{}'.format(config.version))
    save_folder = os.path.join(config.release_path, config.model_name, config.version)
    tf.logging.info("model saved in :%s" % save_folder)
    if os.path.isdir(save_folder):
        tf.logging.info("Folder exsited! Remove!")
        shutil.rmtree(save_folder)
    #tf.gfile.MakeDirs(save_folder)
    tf.logging.info("Make dir:%s" % save_folder)
    g = tf.Graph()
    with g.as_default():
        with tf.device(None):
            sess=tf.Session()
            input_ids,e1_mas,e2_mas=cnn.get_inputs(config.batch_size,config.max_seq_length)
            labels,probs=cnn.create_model(input_ids,e1_mas,e2_mas)
            sess.run(tf.global_variables_initializer())
            tf.saved_model.simple_save(session=sess,
                                       export_dir=save_folder,
                                       inputs={
                                           'input_ids': input_ids,
                                           'e1_mas': e1_mas,
                                           'e2_mas': e2_mas
                                       },
                                       outputs={
                                           'probs': probs,
                                           'labels': labels,
                                          # 'version': v
                                       })
if __name__=="__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    release_model()