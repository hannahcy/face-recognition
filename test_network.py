import numpy as np
import tensorflow as tf
import sys

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


def computeLabels(faceData):
    n, d = faceData.shape
    # Zero arrays for the labels, should be able to do better than this
    estSexLabels = np.zeros(n)
    estAgeLabels = np.zeros(n)
    estExpLabels = np.zeros(n)
    return estSexLabels, estAgeLabels, estExpLabels

estS, estA, estE = computeLabels(testingFaces)
# I'll do stuff with the above to evaluate the accuracy of your methods


tf.reset_default_graph()

# Create some variables.
accuracy = tf.get_variable("accuracy", shape=[3])
#v2 = tf.get_variable("v2", shape=[5])

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

# Later, launch the model, use the saver to restore variables from disk, and
# do some work with the model.
with tf.Session() as sess:
  # Restore variables from disk.
  saver.restore(sess, "models/sex-model_128_60")
  print("Model restored.")
  # Check the values of the variables
  print("v1 : %s" % accuracy.eval())
  #print("v2 : %s" % v2.eval())