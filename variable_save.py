import tensorflow as tf
sess = tf.InteractiveSession()

raw_data = [1., 2., 8., -1., 0., 5.5, 6., 13]
_spikes = tf.Variable([False] * len(raw_data), name='spikes')
_spikes.initializer.run()        

saver = tf.train.Saver()

for i in range(1, len(raw_data)):
  if raw_data[i] - raw_data[i-1] > 5:
    spikes_val = _spikes.eval()  
    spikes_val[i] = True
    _updater = tf.assign(_spikes, spikes_val)
    _updater.eval()             

save_path = saver.save(sess, "./spikes.ckpt")
print("spikes data saved in file: %s" %save_path)

sess.close()