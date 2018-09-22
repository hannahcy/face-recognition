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

num_child = (trainingAgeLabels == 0).sum()
num_teen = (trainingAgeLabels == 1).sum()
num_adult = (trainingAgeLabels == 2).sum()
num_senior = (trainingAgeLabels == 3).sum()
total = trainingAgeLabels.shape

print(num_child, num_child/total, num_teen, num_teen/total, num_adult, num_adult/total, num_senior, num_senior/total)