import tensorflow as tf
import numpy as np

class Config:
    """Holds model hyperparams and data information.
    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation. Use self.config.? instead of Config.?
    """

if __name__ == "__main__":
    sess = tf.InteractiveSession()

    # config = Config()
    # config.a = None
    # print config.a

    # enc_input = tf.placeholder(tf.float32,(5,2,4))
    # cell = tf.nn.rnn_cell.GRUCell(5)
    # a = tf.nn.dynamic_rnn(cell, enc_input, None, tf.zeros((tf.shape(enc_input)[0], cell.state_size),tf.float32))
    # tf.get_variable_scope().reuse_variables() #Without this, I get an error.
    # b = tf.nn.dynamic_rnn(cell, enc_input, None, tf.zeros((tf.shape(enc_input)[0], cell.state_size),tf.float32))

    # dec_input = enc_input[:,:-1,:]

    # a = tf.constant([[1,2,3],[4,5,6]])
    # print tf.Session().run(a)
    # b = a[:,:-1]
    # print tf.Session().run(b)

    # a = np.array([[1,2,3],[4,5,6]])
    # print a
    # b = np.array([[10,20,30],[40,50,60]])
    # print b
    # a[:,:-1] = b[:,1:]
    # print a
    # print b  

    # embedding = tf.constant([[10,20],[30,40],[50,60]])
    # idx = tf.constant([[[0],[1]],[[1],[0]]])
    # with tf.Session() as sess:
    #     print tf.nn.embedding_lookup(embedding, idx).eval()
    # a = np.random.randn(3, 2)
    # b = tf.Variable(a)
    # with tf.Session() as sess:
    #     sess.run(tf.global_variables_initializer())
    #     print b.eval()
    #     print sess.run(b)

    # a = tf.constant(np.arange(1,13), shape=[2, 2, 3], dtype=tf.int32)
    # b = tf.constant(np.arange(13,25), shape=[2, 3, 2], dtype=tf.int32)
    # c = tf.matmul(a, b)
    # with tf.Session() as sess:
    #     print c.eval()

    # a = tf.placeholder(tf.float32, (None, 3, 2))
    # b = tf.placeholder(tf.float32, (None, 2,4))
    # c = tf.matmul(a,b)

    # print c

    # a = tf.constant([[1,0],[0,1]])
    # b = tf.tile(a, [None,1])
    # print a
    # print b

    # a = tf.constant([[[1],[2]],[[3],[4]]]) # (2,2,1)
    # b = tf.reshape(a, (-1,1))
    # print b.eval()

    # a = tf.range(12)
    # a = tf.reshape(a , (-1,4))
    # print a.eval()

    # b = tf.range(24)
    # b = tf.reshape(b, (12,2))
    # print b.eval()
    # b = tf.reshape(b, (-1,4,2))
    # print b.eval()

    params = tf.constant([[10,0.1],[20,0.2],[30,0.3],[40,0.4]])
    ids = tf.constant([[1,1],[3,3]])
    print tf.nn.embedding_lookup(params,ids).eval()

    sess.close()

    # data = [(1,2,3),(4,5,6)]
    # batches = [np.array(col) for col in zip(*data)]
    # print zip(data)
    # print zip((1,2,3),(4,5,6))
    # print zip(*data)
    # print batches

    # a = np.array([[1,2],[3,4],[5,6]])
    # print a[[0,2]]
