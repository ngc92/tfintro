import tensorflow as tf
import numpy as np

with tf.Graph().as_default() as graph:
    x = tf.placeholder(dtype=tf.float32, name="x")
    A = tf.Variable(np.random.rand(784, 10), trainable=True, dtype=tf.float32, name="W")
    b = tf.Variable(np.random.rand(10), trainable=True, dtype=tf.float32, name="b")
    mul = tf.matmul(x, A) + b
    y =  tf.identity(mul, name="y")

    writer = tf.summary.FileWriter("demo_graph", tf.get_default_graph())
    writer.close()

with tf.Graph().as_default() as graph:
    a = tf.placeholder(dtype=tf.float32, name="a", shape=())
    b = tf.placeholder(dtype=tf.float32, name="b", shape=())
    c = tf.placeholder(dtype=tf.float32, name="c", shape=())
    d = tf.placeholder(dtype=tf.float32, name="d", shape=())

    g = tf.add(a, b, name="g")
    h = tf.add(c, d, name="h")
    
    f = tf.add(g, h, name="f")
    writer = tf.summary.FileWriter("parallel_graph", tf.get_default_graph())
    writer.close()


