import tensorflow as tf
import numpy as np
import os
from PIL import Image
import random
import matplotlib.pyplot as plt

def transformImage(file):
    """Returns the transformed image as numpy array in the dimension (widht, length, color)"""

    im = Image.open(file)
    pix = im.load()
    width = im.size[0]
    height = im.size[1]

    #store the rgb information in a numpy array (width, height, color).
    picture = np.array([[pix[x,y] for y in range(height)] for x in range(width)], np.int32)

    return picture


def createDataSets(smilePath, nonSmilePath, dataSetSize, testingSplit):
    """Createts the training and test datasets from the images in smilePath and nonSmilePath.
     The split is set based on the testing split size in percent.
     If a testing split of 20 is chosen 20% are going to be test data and 80% training data."""

    trainingLabels = []
    trainingSet = []
    testingLabels = []
    testingSet = []

    #transform all smiling pictures
    for root, dirs, files in os.walk(smilePath, True):
        i=0
        #static for loop
        for name in files:
        #all images
        #for name in files:
            if name.endswith(".jpg") and (i<(dataSetSize/2) or dataSetSize == -1):
                if random.randint(1, 100) > testingSplit:
                    trainingSet.append(transformImage(os.path.join(root, name)))
                    trainingLabels.append(np.array([1,0], np.int32))
                else:
                    testingSet.append(transformImage(os.path.join(root, name)))
                    testingLabels.append(np.array([1,0], np.int32))
                i=i+1

    # transform all non-smiling pictures
    for root, dirs, files in os.walk(nonSmilePath, True):
        k=0
        #all images
        #for name in files:
        #static for loop
        for name in files:
            if name.endswith(".jpg") and (k<(dataSetSize/2) or dataSetSize == -1):
                if random.randint(1, 100) > testingSplit:
                    # insert to a random position to avoid overfitting
                    insertPosition = random.randint(0, len(trainingLabels))
                    trainingSet.insert(insertPosition, transformImage(os.path.join(root, name)))
                    trainingLabels.insert(insertPosition, np.array([0, 1], np.int32))
                else:
                    # insert to a random position to avoid overfitting
                    insertPosition = random.randint(0, len(trainingLabels))
                    testingSet.insert(insertPosition, transformImage(os.path.join(root, name)))
                    testingLabels.insert(insertPosition, np.array([0, 1], np.int32))
                k=k+1

    return trainingSet,trainingLabels,testingSet,testingLabels


def tensorPart(trainingSet,trainingLabels,testingSet,testingLabels,batchSize):
    X = tf.placeholder(tf.float32, [None, 320, 240, 3])
    Y_ = tf.placeholder(tf.float32, [None, 2])

    # weight 1
    W1 = tf.Variable(tf.truncated_normal([5, 5, 3, 12], stddev=0.1))
    # bias
    B1 = tf.Variable(tf.ones([12]) / 3)
    # conv layer 1
    Y1 = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='SAME')
    Y1Relu = tf.nn.relu(Y1 + B1)
    # max layer pooling
    pool1 = tf.nn.max_pool(Y1Relu, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')  # halve structure to 160x120

    # reshape
    YY = tf.reshape(pool1, shape=[-1, 120 * 160 * 12])  # 160*120 structure

    # weight 2
    W2 = tf.Variable(tf.truncated_normal([160 * 120 * 12, 2], stddev=0.1))
    # bias
    B2 = tf.Variable(tf.ones([2]) / 3)

    # softmax
    Ylogits = tf.matmul(YY, W2) + B2
    Y = tf.nn.softmax(Ylogits)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(Ylogits, Y_)
    cross_entropy = tf.reduce_mean(cross_entropy)
    # accuracy of the trained model, between 0 (worst) and 1 (best)
    correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    learning_rate = 0.0005
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    # sets to plot
    train_a = []
    train_c = []
    test_a = []
    test_c = []

    # train the model in batches
    for step in range(0,len(trainingSet),batchSize):
        # use the next batch
        batchBegin = step
        batchEnd = step+batchSize
        if batchEnd > len(trainingSet):
            batchEnd = len(trainingSet)

        batch_X = np.asarray(trainingSet[batchBegin:batchEnd])
        batch_Y = np.asarray(trainingLabels[batchBegin:batchEnd])

        # train
        sess.run(train_step, feed_dict={X: batch_X, Y_: batch_Y})
        a, c = sess.run([accuracy, cross_entropy], feed_dict={X: batch_X, Y_: batch_Y})
        train_a.append(a)
        train_c.append(c)
        a, c = sess.run([accuracy, cross_entropy], feed_dict={X: testingSet, Y_: testingLabels})
        test_a.append(a)
        test_c.append(c)

    return train_a, train_c, test_a, test_c


def plotResults(train_a, test_a, train_c, test_c):
    # Plot and visualise the accuracy and loss
    # accuracy training vs testing dataset
    plt.plot(train_a, label='training dataset')
    plt.plot(test_a, label='test dataset')
    plt.legend(bbox_to_anchor=(0, 0.95), loc='lower left', ncol=1)
    plt.xlabel('# batch')
    plt.ylabel('accuracy')
    plt.grid(True)
    #plt.show()
    plt.savefig('accuracy.png')
    plt.clf()

    # loss training vs testing dataset
    plt.plot(train_c, label='training dataset')
    plt.plot(test_c, label='test dataset')
    plt.legend(bbox_to_anchor=(0, 0.95), loc='lower left', ncol=1)
    plt.xlabel('# batch')
    plt.ylabel('loss')
    plt.grid(True)
    #plt.show()
    plt.savefig('loss.png')


def main(argv=None):
    dataSetSize = 750 # use -1 for all images
    testingSplit = 20 # in % of total data-set size
    batchSize = 25
    trainingSet, trainingLabels, testingSet, testingLabels = createDataSets("AMFED/AMFED/happiness/","AMFED/AMFED/nonHapiness/",dataSetSize,testingSplit)
    #batchSize = len(testingSet) #to train in batches of the testing set size
    print "size of training set:", len(trainingSet), len(trainingLabels)
    print "size of testing set:", len(testingSet), len(testingLabels)
    train_a, train_c, test_a, test_c = tensorPart(trainingSet,trainingLabels,testingSet,testingLabels,batchSize)
    #print "Training and Testing - Accurracy, Cross Entropy:"
    #print train_a, train_c
    #print test_a, test_c
    plotResults(train_a, test_a, train_c, test_c)

if __name__ == '__main__':
    main()
