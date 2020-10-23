
#1   20 from ML. Tensor Flow Basics-Deep Learning with Neural Networks.
#tensor is an array-like object.
#/home/martha/anaconda3/lib/python3.7/site-packages/tensorflow/__init__.py

#import inspect line 5 and 6 prints tf path above
##print(inspect.getsourcefile(tensorflow))

import tensorflow as tf

x1 = tf.constant([5])
x2 = tf.constant([6])

#result = tf.mul(x1,x2)  #this is the official way to do itor
result = x1*x2
print(result)
#everything till this point will give an abstract tensor as result
#to see the result, we run a session as below

#sess = tf.Session() # this is deprecated, use code below
#sess = tf.compat.v1.Session()
#print(sess.run(result))
#sess.close()

# instead of lines 20-22, say

with tf.compat.v1.Session() as sess:
    print(sess.run(result))
#see project in tensor dir





