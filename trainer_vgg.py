import numpy as np
import os
import sys

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Read in training data and labels

# Some useful parsing functions

# male/female -> 0/1
def parseSexLabel(string):
    if (string.startswith('male')):
        return 0
    if (string.startswith('female')):
        return 1
    print("ERROR parsing sex from " + string)


# child/teen/adult/senior -> 0/1/2/3
def parseAgeLabel(string):
    if (string.startswith('child')):
        return 0
    if (string.startswith('teen')):
        return 1
    if (string.startswith('adult')):
        return 2
    if (string.startswith('senior')):
        return 3
    print("ERROR parsing age from " + string)


# serious/smiling -> 0/1
def parseExpLabel(string):
    if (string.startswith('serious')):
        return 0
    if (string.startswith('smiling') or string.startswith('funny')):
        return 1
    print("ERROR parsing expression from " + string)


# Count number of training instances

numTraining = 0

for line in open("MITFaces/faceDR"):
    if line.find('_missing descriptor') < 0:
        numTraining += 1

dimensions = 128 * 128

trainingFaces = np.zeros([numTraining, dimensions])
trainingSexLabels = np.zeros(numTraining)  # Sex - 0 = male; 1 = female
trainingAgeLabels = np.zeros(numTraining)  # Age - 0 = child; 1 = teen; 2 = male
trainingExpLabels = np.zeros(numTraining)  # Expression - 0 = serious; 1 = smiling

index = 0
for line in open("MITFaces/faceDR"):
    if line.find('_missing descriptor') >= 0:
        continue
    # Parse the label data
    parts = line.split()
    trainingSexLabels[index] = parseSexLabel(parts[2])
    trainingAgeLabels[index] = parseAgeLabel(parts[4])
    trainingExpLabels[index] = parseExpLabel(parts[8])
    # Read in the face
    fileName = "MITFaces/rawdata/" + parts[0]
    fileIn = open(fileName, 'rb')
    trainingFaces[index, :] = np.fromfile(fileIn, dtype=np.uint8, count=dimensions) / 255.0
    fileIn.close()
    # And move along
    index += 1

# Count number of validation/testing instances

numValidation = 0
numTesting = 0

# Assume they're all Validation
for line in open("MITFaces/faceDS"):
    if line.find('_missing descriptor') < 0:
        numTraining += 1
    numValidation += 1

# And make half of them testing
numTesting = int(numValidation / 2)
numValidation -= numTesting

validationFaces = np.zeros([numValidation, dimensions])
validationSexLabels = np.zeros(numValidation)  # Sex - 0 = male; 1 = female
validationAgeLabels = np.zeros(numValidation)  # Age - 0 = child; 1 = teen; 2 = male
validationExpLabels = np.zeros(numValidation)  # Expression - 0 = serious; 1 = smiling

testingFaces = np.zeros([numTesting, dimensions])
testingSexLabels = np.zeros(numTesting)  # Sex - 0 = male; 1 = female
testingAgeLabels = np.zeros(numTesting)  # Age - 0 = child; 1 = teen; 2 = male
testingExpLabels = np.zeros(numTesting)  # Expression - 0 = serious; 1 = smiling

index = 0
for line in open("MITFaces/faceDS"):
    if line.find('_missing descriptor') >= 0:
        continue

    # Parse the label data
    parts = line.split()
    if (index < numTesting):
        testingSexLabels[index] = parseSexLabel(parts[2])
        testingAgeLabels[index] = parseAgeLabel(parts[4])
        testingExpLabels[index] = parseExpLabel(parts[8])
        # Read in the face
        fileName = "MITFaces/rawdata/" + parts[0]
        fileIn = open(fileName, 'rb')
        testingFaces[index, :] = np.fromfile(fileIn, dtype=np.uint8, count=dimensions) / 255.0
        fileIn.close()
    else:
        vIndex = index - numTesting
        validationSexLabels[vIndex] = parseSexLabel(parts[2])
        validationAgeLabels[vIndex] = parseAgeLabel(parts[4])
        validationExpLabels[vIndex] = parseExpLabel(parts[8])
        # Read in the face
        fileName = "MITFaces/rawdata/" + parts[0]
        fileIn = open(fileName, 'rb')
        validationFaces[vIndex, :] = np.fromfile(fileIn, dtype=np.uint8, count=dimensions) / 255.0
        fileIn.close()

    # And move along
    index += 1
print("Data loaded")
sys.stdout.flush()

import random
from scipy import ndarray
import skimage as sk
from skimage import transform
from skimage import util
import copy

def random_rotation(image_array):
    # pick a random degree of rotation between 25% on the left and 25% on the right
    random_degree = random.uniform(-25, 25)
    return sk.transform.rotate(image_array, random_degree)

def random_noise(image_array):
    # add random noise to the image
    return sk.util.random_noise(image_array)

def horizontal_flip(image_array):
    # horizontal flip doesn't need skimage, it's easy as flipping the image array of pixels !
    return image_array[:, ::-1]

new_faces = np.zeros([trainingFaces.shape[0]*10, dimensions])
new_trainingSexLabels = np.zeros(trainingFaces.shape[0]*10)  # Sex - 0 = male; 1 = female
new_trainingAgeLabels = np.zeros(trainingFaces.shape[0]*10)  # Age - 0 = child; 1 = teen; 2 = male
new_trainingExpLabels = np.zeros(trainingFaces.shape[0]*10)  # Expression - 0 = serious; 1 = smiling

for i in range(trainingFaces.shape[0]):
    new_index = i*10
    new_faces[new_index, :] = copy.deepcopy(trainingFaces[i])
    reshaped = np.reshape(trainingFaces[i], [128,128])
    #print(reshaped.shape)
    rotated = random_rotation(reshaped)
    new_faces[(new_index + 1),:] = np.reshape(rotated,[128*128])
    rotated = random_rotation(reshaped)
    new_faces[(new_index + 2), :] = np.reshape(rotated, [128 * 128])
    rotated = random_rotation(reshaped)
    new_faces[(new_index + 3), :] = np.reshape(rotated, [128 * 128])
    rotated = random_rotation(reshaped)
    new_faces[(new_index + 4), :] = np.reshape(rotated, [128 * 128])
    new_faces[(new_index + 5),:] = random_noise(trainingFaces[i,:])
    new_faces[(new_index + 6), :] = random_noise(trainingFaces[i, :])
    new_faces[(new_index + 7), :] = random_noise(trainingFaces[i, :])
    new_faces[(new_index + 8), :] = random_noise(trainingFaces[i, :])
    flipped = horizontal_flip(reshaped)
    new_faces[(new_index + 9),:] = np.reshape(flipped,[128*128])
    for j in range(new_index, new_index+10):
        new_trainingSexLabels[j] = trainingSexLabels[i]
        new_trainingAgeLabels[j] = trainingAgeLabels[i]
        new_trainingExpLabels[j] = trainingExpLabels[i]

trainingFaces = copy.deepcopy(new_faces)
trainingSexLabels = copy.deepcopy(new_trainingSexLabels)
trainingAgeLabels = copy.deepcopy(new_trainingAgeLabels)
trainingExpLabels = copy.deepcopy(new_trainingExpLabels)

import tensorflow as tf

####### MODIFIABLE PARAMETERS ######

task = "Age"  # Options are "Sex", "Age", "Expression"
network = "vggD" # Options are "vggC", "vggD", "vggE", "vggE1"
device = '/gpu:0' # '/cpu:0' or '/gpu:0'

batch_size = 16
n_epochs = 500
learning_rate = 0.0001

n_filters_conv1 = 64
filter_size_conv1 = 3
stride1 = 1

n_filters_conv2 = 64
filter_size_conv2 = 3
stride2 = 1

n_filters_conv3 = 128
filter_size_conv3 = 3
stride3 = 1

n_filters_conv4 = 128
filter_size_conv4 = 3
stride4 = 1

n_filters_conv5 = 256
filter_size_conv5 = 3
stride5 = 1

n_filters_conv6 = 256
filter_size_conv6 = 3
stride6 = 1

n_filters_conv7 = 256
filter_size_conv7 = 3
stride7 = 1
if network == "vggC":
    filter_size_conv7 = 1

n_filters_conv75 = 256 ## used for vggE and vggE1 only
filter_size_conv75 = 3
stride75 = 1
if network == "vggE1":
    filter_size_conv75 = 1

n_filters_conv8 = 512
filter_size_conv8 = 3
stride8 = 1

n_filters_conv9 = 512
filter_size_conv9 = 3
stride9 = 1

n_filters_conv10 = 512
filter_size_conv10 = 3
stride10 = 1
if network == "vggC":
    filter_size_conv10 = 1

n_filters_conv105 = 512 ## used for vggE and vggE1 only
filter_size_conv105 = 3
stride105 = 1
if network == "vggE1":
    filter_size_conv105 = 1

n_filters_conv11 = 512
filter_size_conv11 = 3
stride11 = 1

n_filters_conv12 = 512
filter_size_conv12 = 3
stride12 = 1

n_filters_conv13 = 512
filter_size_conv13 = 3
stride13 = 1
if network == "vggC":
    filter_size_conv13 = 1

n_filters_conv135 = 512 ## used for vggE and vggE1 only
filter_size_conv135 = 3
stride135 = 1
if network == "vggE1":
    filter_size_conv135 = 1

fc1_layer_size = 4096
fc2_layer_size = 4096

display_step = 1
saver_step = 10

####################################

def make_one_hot(labels):
    global n_classes
    one_label = np.zeros(n_classes)
    new_labels = [one_label] * len(labels)
    for i in range(len(labels)):
        # print(i)
        one_label = np.zeros(n_classes)
        one_label[int(labels[i])] = 1
        new_labels[i] = one_label
        # print(labels[i])
        # print(new_labels[i])
    return np.array(new_labels)


class Dataset:
    def __init__(self, data, labels):
        # print(data.shape)
        self.data = data.reshape(
            [-1, 128, 128, 1])  # tf.convert_to_tensor(data.reshape([-1,128,128,1]), dtype=tf.float32)
        self.labels = labels  # .reshape(-1,n_classes) # n_classes
        self.batch_index = 0

    def randomize(self, sess):
        shuffled_data = np.empty(self.data.shape, dtype=self.data.dtype)
        shuffled_labels = np.empty(self.labels.shape, dtype=self.labels.dtype)
        permutation = np.random.permutation(len(self.data))
        for old_index, new_index in enumerate(permutation):
            shuffled_data[new_index] = self.data[old_index]
            shuffled_labels[new_index] = self.labels[old_index]
        self.data = shuffled_data
        self.labels = shuffled_labels

    def next_batch(self, b_size):
        start = self.batch_index
        end = self.batch_index + b_size
        self.batch_index = end
        return self.data[start:end], self.labels[start:end]


def conv_relu_layer(input, n_input, n_filters, filter_size, stride):
    weights = tf.Variable(tf.truncated_normal(shape=[filter_size, filter_size, n_input, n_filters], stddev=0.05))
    biases = tf.Variable(tf.constant(0.05, shape=[n_filters]))
    conv_layer = tf.nn.conv2d(input=input, filter=weights, strides=[1, stride, stride, 1], padding='SAME')
    conv_layer += biases
    c_r_layer = tf.nn.relu(conv_layer)
    return c_r_layer

def maxpool_relu_layer(input):
    m_layer = tf.nn.max_pool(value=input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    m_r_layer = tf.nn.relu(m_layer)
    return m_r_layer

def flat_layer(input_layer):
    shape = input_layer.get_shape()
    num_features = shape[1:4].num_elements()
    flat_layer = tf.reshape(input_layer, [-1, num_features])
    return flat_layer

def fc_layer(input, n_inputs, n_outputs, use_relu=True):
    weights = tf.Variable(tf.truncated_normal(shape=[n_inputs, n_outputs], stddev=0.05))
    biases = tf.Variable(tf.constant(0.05, shape=[n_outputs]))
    fc_layer = tf.matmul(input, weights) + biases
    if use_relu:
        fc_layer = tf.nn.relu(fc_layer)
    return fc_layer


# print("after layer defns, before model defined")

if task == "Sex":
    n_classes = 2
    train_labels = make_one_hot(trainingSexLabels)
    valid_labels = make_one_hot(validationSexLabels)
    test_labels = make_one_hot(testingSexLabels)
    train_data = Dataset(trainingFaces, train_labels)
    valid_data = Dataset(validationFaces, valid_labels)
    test_data = Dataset(testingFaces, test_labels)
    model = "models/aug10-sex-model"
elif task == "Age":
    n_classes = 4
    train_labels = make_one_hot(trainingAgeLabels)
    valid_labels = make_one_hot(validationAgeLabels)
    test_labels = make_one_hot(testingAgeLabels)
    train_data = Dataset(trainingFaces, train_labels)
    valid_data = Dataset(validationFaces, valid_labels)
    test_data = Dataset(testingFaces, test_labels)
    model = "models/aug10-age-model"
elif task == "Expression":
    n_classes = 2
    train_labels = make_one_hot(trainingExpLabels)
    valid_labels = make_one_hot(validationExpLabels)
    test_labels = make_one_hot(testingExpLabels)
    train_data = Dataset(trainingFaces, train_labels)
    valid_data = Dataset(validationFaces, valid_labels)
    test_data = Dataset(testingFaces, test_labels)
    model = "models/aug10-exp-model"
else:
    print("Please set task to one of the three options")
    sys.stdout.flush()

img_size = 128
num_channels = 1  # greyscale
n_batches = trainingFaces.shape[0] // batch_size
val_batches = validationFaces.shape[0] // batch_size

with tf.device(device):
    # set up VGG
    g = tf.Graph()
    with g.as_default():
        keep_prob = tf.placeholder(tf.float32)
        X = tf.placeholder(tf.float32, shape=[None, img_size, img_size, num_channels], name='X')
        y_true = tf.placeholder(tf.float32, shape=[None, n_classes], name='y_true')
        y_true_class = tf.argmax(y_true, dimension=1)

        conv1 = conv_relu_layer(input=X, n_input=num_channels, n_filters=n_filters_conv1,
                                filter_size=filter_size_conv1, stride = stride1)
        conv2 = conv_relu_layer(input=conv1, n_input=n_filters_conv1, n_filters=n_filters_conv2,
                                filter_size=filter_size_conv2, stride = stride2)
        max1 = maxpool_relu_layer(conv2)
        drop1 = tf.nn.dropout(max1, keep_prob)
        conv3 = conv_relu_layer(input=drop1, n_input=n_filters_conv2, n_filters=n_filters_conv3,
                                filter_size=filter_size_conv3, stride=stride3)
        conv4 = conv_relu_layer(input=conv3, n_input=n_filters_conv3, n_filters=n_filters_conv4,
                                filter_size=filter_size_conv4, stride=stride4)
        max2 = maxpool_relu_layer(conv4)
        drop2 = tf.nn.dropout(max2, keep_prob)
        conv5 = conv_relu_layer(input=drop2, n_input=n_filters_conv4, n_filters=n_filters_conv5,
                                filter_size=filter_size_conv5, stride = stride5)
        conv6 = conv_relu_layer(input=conv5, n_input=n_filters_conv5, n_filters=n_filters_conv6,
                                filter_size=filter_size_conv6, stride = stride6)
        conv7 = conv_relu_layer(input=conv6, n_input=n_filters_conv6, n_filters=n_filters_conv7,
                                filter_size=filter_size_conv7, stride = stride7)
        if network == "vggE" or network == "vggE1":
            conv75 = conv_relu_layer(input=conv7, n_input=n_filters_conv7, n_filters=n_filters_conv75,
                                     filter_size=filter_size_conv75, stride=stride75)
            max3 = maxpool_relu_layer(conv75)
        else:
            max3 = maxpool_relu_layer(conv7)
        drop3 = tf.nn.dropout(max3, keep_prob)
        conv8 = conv_relu_layer(input=drop3, n_input=n_filters_conv7, n_filters=n_filters_conv8,
                                filter_size=filter_size_conv8, stride=stride8)
        conv9 = conv_relu_layer(input=conv8, n_input=n_filters_conv8, n_filters=n_filters_conv9,
                                filter_size=filter_size_conv9, stride=stride9)
        conv10 = conv_relu_layer(input=conv9, n_input=n_filters_conv9, n_filters=n_filters_conv10,
                                filter_size=filter_size_conv10, stride=stride10)
        if network == "vggE" or network == "vggE1":
            conv105 = conv_relu_layer(input=conv10, n_input=n_filters_conv10, n_filters=n_filters_conv105,
                                      filter_size=filter_size_conv105, stride=stride105)
            max4 = maxpool_relu_layer(conv105)
        else:
            max4 = maxpool_relu_layer(conv10)
        drop4 = tf.nn.dropout(max4, keep_prob)
        conv11 = conv_relu_layer(input=drop4, n_input=n_filters_conv10, n_filters=n_filters_conv11,
                                 filter_size=filter_size_conv11, stride=stride11)
        conv12 = conv_relu_layer(input=conv11, n_input=n_filters_conv11, n_filters=n_filters_conv12,
                                 filter_size=filter_size_conv12, stride=stride12)
        conv13 = conv_relu_layer(input=conv12, n_input=n_filters_conv12, n_filters=n_filters_conv13,
                                 filter_size=filter_size_conv13, stride=stride13)
        if network == "vggE" or network == "vggE1":
            conv135 = conv_relu_layer(input=conv13, n_input=n_filters_conv13, n_filters=n_filters_conv135,
                                      filter_size=filter_size_conv135, stride=stride135)
            max5 = maxpool_relu_layer(conv135)
        else:
            max5 = maxpool_relu_layer(conv13)
        drop5 = tf.nn.dropout(max5, keep_prob)
        flat = flat_layer(drop5)
        fc1 = fc_layer(input=flat, n_inputs=flat.get_shape()[1:4].num_elements(), n_outputs=fc1_layer_size)
        #fc1 = fc_layer(input=max5, n_inputs=filter_size_conv13, n_outputs=fc1_layer_size)
        fc2 = fc_layer(input=fc1, n_inputs=fc1_layer_size, n_outputs=fc2_layer_size)
        fc3 = fc_layer(input=fc2, n_inputs=fc2_layer_size, n_outputs=n_classes, use_relu=False)  # n_outputs=n_classes
        y_pred = tf.nn.softmax(fc3, name="y_pred")
        y_pred_class = tf.argmax(y_pred, dimension=1)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=fc3, labels=y_true) # changed fc3 to y_pred
        cost = tf.reduce_mean(cross_entropy)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
        correct_prediction = tf.equal(y_pred_class, y_true_class)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="accuracy")
        # print("Graph initialised")

        saver = tf.train.Saver()

        # session run with one kind of label
        with tf.Session() as sess:
            # print("inside session")
            # Run the initializer
            sess.run(tf.global_variables_initializer())
            # print("Initialised")
            val_acc = 0
            val_loss = 0
            for val in range(val_batches):
                x_valid_batch, y_valid_batch = valid_data.next_batch(batch_size)
                feed_dict_val = {X: x_valid_batch, y_true: y_valid_batch, keep_prob: 1.0}
                val_acc += sess.run(accuracy, feed_dict=feed_dict_val)
                val_loss += sess.run(cost, feed_dict=feed_dict_val)
            val_acc = val_acc / val_batches
            val_loss = val_loss / val_batches
            msg = "Pre-training (Epoch {0}) --- Training Accuracy: {1:>6.2%}, Validation Accuracy: {2:>6.2%},  Validation Loss: {3:.3f}"
            print(msg.format(0, 0, val_acc, val_loss))  # , val_loss))
            sys.stdout.flush()
            for i in range(1, n_epochs + 1):
                train_data.randomize(sess)
                train_data.batch_index = 0
                valid_data.randomize(sess)
                valid_data.batch_index = 0
                acc = 0
                val_acc = 0
                val_loss = 0
                for batch in range(n_batches):
                    # if batch % 10 == 0:
                    #    print('Batch', batch, 'of', n_batches, 'done')
                    x_batch, y_true_batch = train_data.next_batch(batch_size)
                    feed_dict_train = {X: x_batch, y_true: y_true_batch, keep_prob: 0.75}
                    sess.run(optimizer, feed_dict=feed_dict_train)
                    acc += sess.run(accuracy, feed_dict=feed_dict_train)
                if i % display_step == 0:
                    valid_data.batch_index = 0
                    for j in range(val_batches):
                        x_valid_batch, y_valid_batch = valid_data.next_batch(batch_size)
                        feed_dict_val = {X: x_valid_batch, y_true: y_valid_batch, keep_prob: 1.0}
                        val_acc += sess.run(accuracy, feed_dict=feed_dict_val)
                        val_loss += sess.run(cost, feed_dict=feed_dict_val)
                acc = acc / n_batches
                val_acc = val_acc / val_batches
                val_loss = val_loss / val_batches
                msg = "Training Epoch {0} --- Training Accuracy: {1:>6.2%}, Validation Accuracy: {2:>6.2%},  Validation Loss: {3:.3f}"
                print(msg.format(i, acc, val_acc, val_loss))  # , val_loss))
                sys.stdout.flush()
                if i % saver_step == 0 or val_acc > 0.85:
                    save_path = saver.save(sess, model+"_"+network+"_"+str(i))

print("Done!")
# print(numTraining)
# print(numValidation)
# print(numTesting)
# print(trainingFaces.shape)
# print(validationFaces.shape)
# print(testingFaces.shape)
