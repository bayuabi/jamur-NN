import tensorflow as tf
import numpy as np

LOGDIR = "/mnt/5C11F6571564DCB5/ML/graphs5"

x_train = np.array([[20,20],
                 [20,40],
                 [20,50],
                 [20,60],
                 [20,80],
                 [25,20],
                 [25,40],
                 [25,50],
                 [25,60],
                 [25,80],
                 [30,20],
                 [30,40],
                 [30,50],
                 [30,60],
                 [30,80],
                 [35,20],
                 [35,40],
                 [35,50],
                 [35,60],
                 [35,80],
                 [40,20],
                 [40,40],
                 [40,50],
                 [40,60],
                 [40,80]], dtype=np.float32)

y_train = np.array([[5,24],
             [5,24],
             [5,22],
             [5,22],
             [5,18],
             [9,24],
             [9,24],
             [9,22],
             [9,22],
             [9,18],
             [12,24],
             [12,24],
             [12,22],
             [12,22],
             [12,18],
             [12,24],
             [12,24],
             [12,22],
             [12,22],
             [12,18],
             [12,24],
             [12,24],
             [12,22],
             [12,22],
             [12,18]], dtype= np.float32)

x_test = np.array([[37,20]], dtype=np.float32)
y_test = np.array([[0,0,1],
                   [0,0,1],
                   [0,0,1],
                   [0,1,0],
                   [1,0,0],
                   [1,0,0]], dtype=np.float32)
#y_train = np.array([[5],[9],[12]],dtype=np.float32)
x_test = x_test/100
x_train = x_train/100
y_train = y_train

X = tf.placeholder(tf.float32, [None,2])
Y = tf.placeholder(tf.float32, [None,2])

weights = {
  "layer1" : tf.Variable(tf.random_normal([2, 10]), name="weight_hidden"),
  "layer2" : tf.Variable(tf.random_normal([10, 10]), name="weight_output"),
  "layer3" : tf.Variable(tf.random_normal([10, 10]), name="weight_output"),
  "layer4" : tf.Variable(tf.random_normal([10, 2]), name="weight_output")
}

biases = {
  "layer1" : tf.Variable(tf.random_normal([10]), name="bias_hidden"),
  "layer2" : tf.Variable(tf.random_normal([10]), name="bias_output"),
  "layer3" : tf.Variable(tf.random_normal([10]), name="bias_output"),
  "layer4" : tf.Variable(tf.random_normal([2]), name="bias_output")
}
'''
weights= {
    "layer1": tf.get_variable(name="w1", shape=[2,25], initializer=tf.random_normal_initializer()),
    "layer2": tf.get_variable(name="w2", shape=[25,25], initializer=tf.random_normal_initializer()),
    "layer3": tf.get_variable(name="w3", shape=[25,3], initializer=tf.random_normal_initializer())
}
biases= {
    "layer1": tf.get_variable(name="b1", shape=[1,25], initializer=tf.random_normal_initializer()),
    "layer2": tf.get_variable(name="b2", shape=[1,25], initializer=tf.random_normal_initializer()),
    "layer3": tf.get_variable(name="b3", shape=[1,3], initializer=tf.random_normal_initializer())
}
'''
def model(X, weights, biases):
    hidden1 = tf.add(tf.matmul(X, weights["layer1"]), biases["layer1"])
    hidden1 = tf.nn.sigmoid(hidden1)

    hidden2 = tf.add(tf.matmul(hidden1, weights["layer2"]), biases["layer2"])
    hidden2 = tf.nn.sigmoid(hidden2)

    hidden3 = tf.add(tf.matmul(hidden2, weights["layer3"]), biases["layer3"])
    hidden3 = tf.nn.sigmoid(hidden3)

    output = tf.add(tf.matmul(hidden3, weights["layer4"]), biases["layer4"])
    #output = tf.nn.sigmoid(output)
    return output

y_pred = model(X, weights, biases)
loss = tf.reduce_mean((tf.square(Y-y_pred))*0.5)
#loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=Y))
#loss = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(y_pred), reduction_indices=[1]))
optimizer = tf.train.GradientDescentOptimizer(0.9).minimize(loss)
b=tf.summary.scalar("loss",loss)
#summ = tf.summary.merge_all()
writer = tf.summary.FileWriter(LOGDIR)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer.add_graph(sess.graph)
    #writer.add_summary()
    for epoch in range(500000):
        _,l,a = sess.run([optimizer,loss,b],feed_dict={X:x_train, Y:y_train})
        if epoch % 1000 == 0:
            writer.add_summary(a,epoch)
            print("epoch ke-", epoch, "loss:", l)
    print(sess.run([weights,biases]))
    y_coba = model(X, weights, biases)
    print((sess.run(y_coba, feed_dict={X:x_train})))
    '''
    test_result = sess.run(y_pred, feed_dict={X:x_test})
    correct_pred = tf.equal(tf.argmax(test_result,1), tf.argmax(Y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, "float"))
    print(sess.run(accuracy,feed_dict={X:x_test, Y:y_test}))
    '''
