import tensorflow as tf
a = tf.constant([[3, 4, -1, 6],[0,2,1,3],[1,2,5,4]])
max_index = tf.nn.top_k(a, 4)[1]
value=tf.nn.top_k(a,4)[0]
# rev=tf.reverse(a, axis=1, name=None) 

sess = tf.Session()
value_max = sess.run( value)
print(value_max)