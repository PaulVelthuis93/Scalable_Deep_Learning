import tensorflow as tf
import numpy as np
import os
from PIL import Image
import random

trainingSet = []
testingSet = []
testingLabels = []
def transformImage(file):
    """Returns the transformed image as numpy array in the dimension (widht, length, color)"""

    im = Image.open(file)
    pix = im.load()
    width = im.size[0]
    height = im.size[1]

    #store the rgb information in a numpy array (width, height, color).
    picture = np.array([[pix[x,y] for y in range(height)] for x in range(width)], np.int32)

    return picture


def createDataSets(smilePath, nonSmilePath):
    """Returns a dataset of pictures and a dataset of according labels as numpy arrays
     with the dimensions (images, widht, height, color) and (image, smiling/not smiling). """

    pictures = []
    labels = []

    #transform all smiling pictures
    for root, dirs, files in os.walk(smilePath, True):
        for name in files:
            if name.endswith(".jpg"):
                pictures.append(transformImage(os.path.join(root, name)))
                labels.append(np.array([1], np.int32))

    # transform all non-smiling pictures
    for root, dirs, files in os.walk(nonSmilePath, True):
        for name in files:
            if name.endswith(".jpg"):
                pictures.append(transformImage(os.path.join(root, name)))
                labels.append(np.array([0], np.int32))

    return np.asarray(pictures), np.asarray(labels)

#TODO
#define training and testset
#define variables and place holders Y_=labels

#define the model for computing predictions
#define the loss function
#define the accuracy
#train with an optimizer

def splitIntoTrainAndTestData(pictures,labels,testingSplit):
    """Splits our dataset into train an test set based on the testing split size in percent.
    If a testing split of 20 is chosen 20% are going to be test data and 80% training data."""

    trainingLabels = []
    global testingSet
    global trainingSet
    global testingLabels

    for i in range(pictures.shape[0]):
        if random.randint(1, 100) > testingSplit:
            trainingSet.append(pictures[i])
            trainingLabels.append(labels[i])
        else:
            testingSet.append(pictures[i])
            testingLabels.append(labels[i])
    trainingSet1 = np.asarray(trainingSet)
    
    trainingSet = trainingSet1
    trainingLabels = np.asarray(trainingLabels)
    
    testingSet = np.asarray(testingSet)
    testingLabels = np.asarray(testingLabels)

    #print trainingSet.shape
    #print trainingLabels.shape
    #print testingSet.shape
    #print testingLabels.shape
    return trainingSet,trainingLabels,testingSet,testingLabels

def tensorPart(trainingSet,tainingLabels, testingSet, testingLabels):
    #input X 320x240 RGB images
    X = tf.placeholder(tf.float32, [None,320,240,3])
    Y_ = tf.placeholder(tf.float32, [None, 1])
    pkeep = tf.placeholder(tf.float32)
    #weight 1
    W1 = tf.Variable(tf.truncated_normal([5,5,3,8],stddev = 0.1))
    #bias
    B1=tf.Variable(tf.ones([8])/3)
    #conv layer 1
    Y1 = tf.nn.conv2d(X,W1, strides=[1,1,1,1], padding ='SAME')
    Y1Relu=tf.nn.relu(Y1+B1)
    #max layer pooling
    pool1 = tf.nn.max_pool(Y1Relu, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1') #halve structure to 160x120
    print pool1
    #reshape
    YY = tf.reshape(pool1, shape=[-1, 160 * 120 * 12]) #160*120 structure

    #weight 2
    W2 = tf.Variable(tf.truncated_normal([160*120*12,1],stddev = 0.1))
    #bias
    B2=tf.Variable(tf.ones([1])/3)

    #softmax
    Ylogits = tf.matmul(YY, W2) + B2
    Y = tf.nn.softmax(Ylogits)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(Ylogits, Y_)
    cross_entropy = tf.reduce_mean(cross_entropy)
    # accuracy of the trained model, between 0 (worst) and 1 (best)
    correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    learning_rate =0.0005
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

    init=tf.initialize_all_variables()
    sess=tf.Session()
    sess.run(init)

    # Train and test the model, store the accuracy and loss per iteration
    train_a = []
    train_c = []
    test_a = []
    test_c = []
    
    training_iter = trainingSet.shape[0]
    epoch_size = trainingSet.shape[0]/10
    for i in range(training_iter):
        test = False
        if i % epoch_size == 0:
            test = True
        a, c, ta, tc = training_step(i, test, test, sess, train_step, X, Y_, learning_rate,pkeep)
        train_a += a
        train_c += c
        test_a += ta
        test_c += tc
    
    # Calculate the maximal accuracy
    max = 0
    e = 0
    for i in range(epoch_size):
        if test_a[i] > max:
            max = test_a[i]
            e = i
    print("Maximal accuracy at Epoch %d - Testing dataset\n Accuracy=%.4f\n Loss=%.4f" % (e,test_a[e],test_c[e]))
    
    # Plot and visualise the accuracy and loss
    # accuracy training vs testing dataset
    plt.plot(train_a, label='training dataset')
    plt.plot(test_a, label='test dataset')
    plt.legend(bbox_to_anchor=(0, 1), loc='lower left', ncol=1)
    plt.xlabel('# epoch')
    plt.ylabel('accuracy')
    plt.grid(True)
    plt.show()

    # loss training vs testing dataset
    plt.plot(train_c, label='training dataset')
    plt.plot(test_c, label='test dataset')
    plt.xlabel('# epoch')
    plt.ylabel('loss')
    plt.grid(True)
    plt.show()

    # Zoom in on the tail of the plots
    zoom_point = 50
    x_range = range(zoom_point,training_iter/epoch_size)
    plt.plot(x_range, train_a[zoom_point:])
    plt.plot(x_range, test_a[zoom_point:])
    plt.xlabel('# epoch')
    plt.ylabel('accuracy')
    plt.grid(True)
    plt.show()

    plt.plot(x_range, train_c[zoom_point:])
    plt.plot(x_range, test_c[zoom_point:])
    plt.xlabel('# epoch')
    plt.ylabel('loss')
    plt.grid(True)
    plt.show()


def training_step(i, update_test_data, update_train_data, sess, train_step, X, Y_,learning_rate,pkeep):

    print "\r", i,
    ####### actual learning 
    # reading batches of 100 images with 100 labels
    batch_X = trainingSet 
    batch_Y = testingSet
    
    # adapt the learning rate
    decay_rate = 0.95
    decay_steps = 500
    if i > decay_steps:
        lr = learning_rate * decay_rate**(int(i/decay_steps))
    else:
        lr = learning_rate
    
    # the backpropagation training step
    sess.run(train_step, feed_dict={X: batch_X, Y_: batch_Y, learning_rate: lr})
    
    ####### evaluating model performance for printing purposes
    # evaluation used to later visualize how well you did at a particular time in the training
    train_a = []
    train_c = []
    test_a = []
    test_c = []
    if update_train_data:
        a, c = sess.run([accuracy, cross_entropy], feed_dict={X: batch_X, Y_: batch_Y})
        train_a.append(a)
        train_c.append(c)

    if update_test_data:
        a, c = sess.run([accuracy, cross_entropy], feed_dict={X: testingSet, Y_: testingLabels})
        test_a.append(a)
        test_c.append(c)

    return (train_a, train_c, test_a, test_c)    
    
def main(argv=None):
    pictures, labels = createDataSets("AMFED/AMFED/happiness/", "AMFED/AMFED/nonHapiness/")
    #print pictures.shape
    #print labels.shape
    testingSplit = 20
    trainingSet, trainingLabels, testingSet, testingLabels = splitIntoTrainAndTestData(pictures,labels,testingSplit)
    tensorPart(trainingSet,trainingLabels,testingSet,testingLabels)
    
if __name__ == '__main__':
    main()
