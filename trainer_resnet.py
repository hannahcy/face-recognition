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

import tensorflow as tf

####### MODIFIABLE PARAMETERS ######

task = "Sex"  # Options are "Sex", "Age", "Expression"
network = "resnet" # Options are "vggC", "vggD", "vggE", "vggE1"
device = '/cpu:0' # '/cpu:0' or '/gpu:0'

batch_size = 16
n_epochs = 1000
learning_rate = 0.0001

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

def conv_pool_layer(input, n_input, n_filters, filter_size, stride):
    weights = tf.Variable(tf.truncated_normal(shape=[filter_size, filter_size, n_input, n_filters], stddev=0.05))
    biases = tf.Variable(tf.constant(0.05, shape=[n_filters]))
    conv_layer = tf.nn.conv2d(input=input, filter=weights, strides=[1, stride, stride, 1], padding='SAME')
    conv_layer += biases
    return conv_layer

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

def residual_layer(input1, input2):
    return tf.add(input1,input2)

def residual_layer_pad(input_small, input_big, pad_small, pad_big):
    rank = tf.rank(input_small)
    paddings = tf.convert_to_tensor([[0, 0], [0, 0], [0, 0], [pad_small, pad_small]])
    padded_input_small = tf.pad(input_small, paddings, "CONSTANT")
    paddings = tf.convert_to_tensor([[0, 0], [pad_big, pad_big], [pad_big, pad_big], [0, 0]])
    padded_input_big = tf.pad(input_big, paddings, "CONSTANT")
    return tf.add(padded_input_small,padded_input_big)

# print("after layer defns, before model defined")

if task == "Sex":
    n_classes = 2
    train_labels = make_one_hot(trainingSexLabels)
    valid_labels = make_one_hot(validationSexLabels)
    test_labels = make_one_hot(testingSexLabels)
    train_data = Dataset(trainingFaces, train_labels)
    valid_data = Dataset(validationFaces, valid_labels)
    test_data = Dataset(testingFaces, test_labels)
    model = "models/sex-model"
elif task == "Age":
    n_classes = 4
    train_labels = make_one_hot(trainingAgeLabels)
    valid_labels = make_one_hot(validationAgeLabels)
    test_labels = make_one_hot(testingAgeLabels)
    train_data = Dataset(trainingFaces, train_labels)
    valid_data = Dataset(validationFaces, valid_labels)
    test_data = Dataset(testingFaces, test_labels)
    model = "models/age-model"
elif task == "Expression":
    n_classes = 2
    train_labels = make_one_hot(trainingExpLabels)
    valid_labels = make_one_hot(validationExpLabels)
    test_labels = make_one_hot(testingExpLabels)
    train_data = Dataset(trainingFaces, train_labels)
    valid_data = Dataset(validationFaces, valid_labels)
    test_data = Dataset(testingFaces, test_labels)
    model = "models/exp-model"
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
        X = tf.placeholder(tf.float32, shape=[None, img_size, img_size, num_channels], name='X')
        y_true = tf.placeholder(tf.float32, shape=[None, n_classes], name='y_true')
        y_true_class = tf.argmax(y_true, dimension=1)

        conv_init = conv_relu_layer(input=X, n_input=num_channels, n_filters=64,
                                filter_size=7, stride = 2)
        max1 = maxpool_relu_layer(conv_init)
        conv64_1 = conv_relu_layer(input=max1, n_input=64, n_filters=64,
                                filter_size=3, stride = 1)
        conv64_2 = conv_relu_layer(input=conv64_1, n_input=64, n_filters=64,
                                filter_size=3, stride = 1)
        res1 = residual_layer(max1,conv64_2)
        conv64_3 = conv_relu_layer(input=res1, n_input=64, n_filters=64,
                                   filter_size=3, stride=1)
        conv64_4 = conv_relu_layer(input=conv64_3, n_input=64, n_filters=64,
                                   filter_size=3, stride=1)
        res2 = residual_layer(res1, conv64_4)
        conv64_5 = conv_relu_layer(input=res2, n_input=64, n_filters=64,
                                   filter_size=3, stride=1)
        conv64_6 = conv_relu_layer(input=conv64_5, n_input=64, n_filters=64,
                                   filter_size=3, stride=1)
        res3 = residual_layer(res2, conv64_6)
        conv128_1 = conv_pool_layer(input=res3, n_input=64, n_filters=128,
                                    filter_size=3, stride=2)
        conv128_2 = conv_relu_layer(input=conv128_1, n_input=128, n_filters=128,
                                    filter_size=3, stride=1)
        res4 = residual_layer_pad(res3,conv128_2,32,8)
        conv128_3 = conv_relu_layer(input=res4, n_input=128, n_filters=128,
                                    filter_size=3, stride=1)
        conv128_4 = conv_relu_layer(input=conv128_3, n_input=128, n_filters=128,
                                    filter_size=3, stride=1)
        res5 = residual_layer(res4, conv128_4)
        conv128_5 = conv_relu_layer(input=res5, n_input=128, n_filters=128,
                                    filter_size=3, stride=1)
        conv128_6 = conv_relu_layer(input=conv128_5, n_input=128, n_filters=128,
                                    filter_size=3, stride=1)
        res6 = residual_layer(res5, conv128_6)
        conv128_7 = conv_relu_layer(input=res6, n_input=128, n_filters=128,
                                    filter_size=3, stride=1)
        conv128_8 = conv_relu_layer(input=conv128_7, n_input=128, n_filters=128,
                                    filter_size=3, stride=1)
        res7 = residual_layer(res6, conv128_8)
        conv256_1 = conv_pool_layer(input=res7, n_input=128, n_filters=256,
                                    filter_size=3, stride=2)
        conv256_2 = conv_relu_layer(input=conv256_1, n_input=256, n_filters=256,
                                    filter_size=3, stride=1)
        res8 = residual_layer_pad(res4, conv256_2, 64, 8)
        conv256_3 = conv_relu_layer(input=res8, n_input=256, n_filters=256,
                                    filter_size=3, stride=1)
        conv256_4 = conv_relu_layer(input=conv256_3, n_input=256, n_filters=256,
                                    filter_size=3, stride=1)
        res9 = residual_layer(res8, conv256_4)
        conv256_5 = conv_relu_layer(input=res9, n_input=256, n_filters=256,
                                    filter_size=3, stride=1)
        conv256_6 = conv_relu_layer(input=conv256_5, n_input=256, n_filters=256,
                                    filter_size=3, stride=1)
        res10 = residual_layer(res9, conv256_6)
        conv256_7 = conv_relu_layer(input=res10, n_input=256, n_filters=256,
                                    filter_size=3, stride=1)
        conv256_8 = conv_relu_layer(input=conv256_7, n_input=256, n_filters=256,
                                    filter_size=3, stride=1)
        res11 = residual_layer(res10, conv256_8)
        conv256_9 = conv_relu_layer(input=res11, n_input=256, n_filters=256,
                                    filter_size=3, stride=1)
        conv256_10 = conv_relu_layer(input=conv256_9, n_input=256, n_filters=256,
                                    filter_size=3, stride=1)
        res12 = residual_layer(res11, conv256_10)
        conv256_11 = conv_relu_layer(input=res12, n_input=256, n_filters=256,
                                    filter_size=3, stride=1)
        conv256_12 = conv_relu_layer(input=conv256_11, n_input=256, n_filters=256,
                                     filter_size=3, stride=1)
        res13 = residual_layer(res12, conv256_12)
        conv512_1 = conv_pool_layer(input=res12, n_input=256, n_filters=512,
                                     filter_size=3, stride=2)
        conv512_2 = conv_relu_layer(input=conv512_1, n_input=512, n_filters=512,
                                     filter_size=3, stride=1)
        res14 = residual_layer_pad(res13, conv512_2, 128, 8)
        conv512_3 = conv_relu_layer(input=res14, n_input=512, n_filters=512,
                                    filter_size=3, stride=1)
        conv512_4 = conv_relu_layer(input=conv512_3, n_input=512, n_filters=512,
                                    filter_size=3, stride=1)
        res15 = residual_layer(res14, conv512_4)
        conv512_5 = conv_relu_layer(input=res15, n_input=512, n_filters=512,
                                    filter_size=3, stride=1)
        conv512_6 = conv_relu_layer(input=conv512_5, n_input=512, n_filters=512,
                                    filter_size=3, stride=1)
        res16 = residual_layer(res15, conv512_6)
        avg_pool = tf.nn.pool(res16, window_shape=[2,2], pooling_type='AVG', padding='SAME')
        flat = flat_layer(avg_pool)
        fc1 = fc_layer(input=flat, n_inputs=flat.get_shape()[1:4].num_elements(), n_outputs=n_classes)
        #fc1 = fc_layer(input=max5, n_inputs=filter_size_conv13, n_outputs=fc1_layer_size)
        #fc2 = fc_layer(input=fc1, n_inputs=fc1_layer_size, n_outputs=fc2_layer_size)
        #fc3 = fc_layer(input=fc2, n_inputs=fc2_layer_size, n_outputs=n_classes, use_relu=False)  # n_outputs=n_classes
        y_pred = tf.nn.softmax(fc1, name="y_pred")
        y_pred_class = tf.argmax(y_pred, dimension=1)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=fc1, labels=y_true)
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
                feed_dict_val = {X: x_valid_batch, y_true: y_valid_batch}
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
                    feed_dict_train = {X: x_batch, y_true: y_true_batch}
                    sess.run(optimizer, feed_dict=feed_dict_train)
                    acc += sess.run(accuracy, feed_dict=feed_dict_train)
                if i % display_step == 0:
                    valid_data.batch_index = 0
                    for j in range(val_batches):
                        x_valid_batch, y_valid_batch = valid_data.next_batch(batch_size)
                        feed_dict_val = {X: x_valid_batch, y_true: y_valid_batch}
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
