import tensorflow as tf
from tensorflow.python.platform import gfile
import sys
import time

if len(sys.argv) < 2:
    print("Usage: python visual_tf_model.py <model.pb>")
    sys.exit(0)

model_file_name = sys.argv[1]
with tf.compat.v1.Session() as sess:
    with gfile.FastGFile(model_file_name, 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
        g_in = tf.compat.v1.import_graph_def(graph_def)
LOGDIR='log'

with tf.compat.v1.Graph().as_default():
    train_writer = tf.compat.v1.summary.FileWriter(LOGDIR)
    train_writer.add_graph(sess.graph)

while True:
    print("AQUI")
    time.sleep(1000)
