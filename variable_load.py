import tensorflow as tf
sess = tf.InteractiveSession()

_spikes = tf.Variable([False] * 8, name='spikes')
saver = tf.train.Saver()

saver.restore(sess, "./spikes.ckpt")
print(_spikes.eval())

sess.close()