import tensorflow as tf
slim = tf.contrib.slim

import numpy as np

from optparse import OptionParser
import wget
import tarfile
import os
import cv2
import time

import default_inc_res_v2
import resnet_v1

from sklearn import linear_model
from sklearn import svm

TRAIN = 0
VAL = 1
TEST = 2

import sys

if sys.version_info[0] == 3:
	print ("Using Python 3")
	import pickle as cPickle
else:
	print ("Using Python 2")
	import cPickle

# Load the model
resnet_checkpoint_file = '/netscratch/siddiqui/CrossLayerPooling/tf-clp/resnet_v1_152.ckpt'
if not os.path.isfile(resnet_checkpoint_file):
	# Download file from the link
	url = 'http://download.tensorflow.org/models/resnet_v1_152_2016_08_28.tar.gz'
	filename = wget.download(url)

	# Extract the tar file
	tar = tarfile.open(filename)
	tar.extractall()
	tar.close()

inc_res_v2_checkpoint_file = '/netscratch/siddiqui/CrossLayerPooling/tf-clp/inception_resnet_v2_2016_08_30.ckpt'
if not os.path.isfile(inc_res_v2_checkpoint_file):
	# Download file from the link
	url = 'http://download.tensorflow.org/models/inception_resnet_v2_2016_08_30.tar.gz'
	filename = wget.download(url)

	# Extract the tar file
	tar = tarfile.open(filename)
	tar.extractall()
	tar.close()

# Command line options
parser = OptionParser()

parser.add_option("-m", "--model", action="store", type="string", dest="model", default="ResNet", help="Model to be used for Cross-Layer Pooling")
parser.add_option("--batchSize", action="store", type="int", dest="batchSize", default=1, help="Batch size to be used")
parser.add_option("--numEpochs", action="store", type="int", dest="numEpochs", default=1, help="Number of epochs")

parser.add_option("--imageWidth", action="store", type="int", dest="imageWidth", default=224, help="Image width for feeding into the network")
parser.add_option("--imageHeight", action="store", type="int", dest="imageHeight", default=224, help="Image height for feeding into the network")
parser.add_option("--imageChannels", action="store", type="int", dest="imageChannels", default=3, help="Number of channels in the image")

parser.add_option("--dataFile", action="store", type="string", dest="dataFile", default="/netscratch/siddiqui/CrossLayerPooling/data/data.txt", help="Training data file")

# Parse command line options
(options, args) = parser.parse_args()
print (options)

IMAGENET_MEAN = [123.68, 116.779, 103.939] # RGB
USE_IMAGENET_MEAN = False


# Reads an image from a file, decodes it into a dense tensor
def _parse_function(filename, label, split):
	image_string = tf.read_file(filename)
	# img = tf.image.decode_image(image_string)
	img = tf.image.decode_jpeg(image_string)

	# img = tf.reshape(img, [options.imageHeight, options.imageWidth, options.imageChannels])
	img = tf.image.resize_images(img, [options.imageHeight, options.imageWidth])
	img.set_shape([options.imageHeight, options.imageWidth, options.imageChannels])
	img = tf.cast(img, tf.float32) # Convert to float tensor
	print (img.shape)
	return img, filename, label, split

# A vector of filenames
print ("Loading data from file: %s" % (options.dataFile))
with open(options.dataFile) as f:
	imageFileNames = f.readlines()
	imNames = []
	imLabels = []
	imSplit = []
	for imName in imageFileNames:
		imName = imName.strip().split(' ')
		imNames.append(imName[0])
		imLabels.append(int(imName[1]))
		imSplit.append(int(imName[2]))

	# imageFileNames = [x.strip().split(' ') for x in imageFileNames] # FileName and Label is separated by a space
	imNames = tf.constant(imNames)
	imLabels = tf.constant(imLabels)
	imSplit = tf.constant(imSplit)

dataset = tf.contrib.data.Dataset.from_tensor_slices((imNames, imLabels, imSplit))
dataset = dataset.map(_parse_function)
dataset = dataset.batch(options.batchSize)

iterator = dataset.make_initializable_iterator()

with tf.name_scope('Model'):
	# Data placeholders
	inputBatchImages, inputBatchImageNames, inputBatchImageLabels, inputBatchImageSplit = iterator.get_next()
	print ("Data shape: %s" % str(inputBatchImages.get_shape()))

	if options.model == "IncResV2":
		scaledInputBatchImages = tf.scalar_mul((1.0/255), inputBatchImages)
		scaledInputBatchImages = tf.subtract(scaledInputBatchImages, 0.5)
		scaledInputBatchImages = tf.multiply(scaledInputBatchImages, 2.0)

		# Create model
		arg_scope = default_inc_res_v2.inception_resnet_v2_arg_scope()
		with slim.arg_scope(arg_scope):
			logits, aux_logits, end_points = default_inc_res_v2.inception_resnet_v2(scaledInputBatchImages, is_training=False)

			# Get the lower layer and upper layer activations
			lowerLayerActivations = end_points["s"]
			upperLayerActivations = end_points["s"]

		# Create list of vars to restore before train op
		variables_to_restore = slim.get_variables_to_restore(include=["InceptionResnetV2"])

	elif options.model == "ResNet":
		if USE_IMAGENET_MEAN:
			print (inputBatchImages.shape)
			channels = tf.split(axis=3, num_or_size_splits=options.imageChannels, value=inputBatchImages)
			for i in range(options.imageChannels):
				channels[i] -= IMAGENET_MEAN[i]
			processedInputBatchImages = tf.concat(axis=3, values=channels)
			print (processedInputBatchImages.shape)
		else:
			imageMean = tf.reduce_mean(inputBatchImages, axis=[1, 2], keep_dims=True)
			print ("Image mean shape: %s" % str(imageMean.shape))
			processedInputBatchImages = inputBatchImages - imageMean

		# Create model
		arg_scope = resnet_v1.resnet_arg_scope()
		with slim.arg_scope(arg_scope):
			logits, end_points = resnet_v1.resnet_v1_152(processedInputBatchImages, is_training=False)

			# Get the lower layer and upper layer activations
			lowerLayerActivations = end_points["Model/resnet_v1_152/block3/unit_15/bottleneck_v1"]
			upperLayerActivations = end_points["Model/resnet_v1_152/block3/unit_20/bottleneck_v1"]

		# Create list of vars to restore before train op
		variables_to_restore = slim.get_variables_to_restore(include=["resnet_v1_152"])

	else:
		print ("Error: Unknown model selected")
		exit(-1)

'''
##### Matlab Code #####
z(cnt+(k-1)*D+1:cnt+k*D) = sum(X(index(active_id),:).*repmat(Coding(index(active_id),k),[1,D]));

for i = 0:sum(prod(SPM_Config))*m-1
	z(i*D+1:(i+1)*D) =	z(i*D+1:(i+1)*D)/(1e-7 + norm(z(i*D+1:(i+1)*D))); 
end

z = FisherVectorSC_Pooling(SPM_Config, x_h, y_h, LF_L2_4, LF_L2_5, option);
z = (z - mean(z(:))) / std(z(:));
z = sqrt(abs(z)) .* sign(z);
z = z / (1e-7 + norm(z));
'''

numChannelsLowerLayer = lowerLayerActivations.get_shape()[-1]
numChannelsUpperLayer = upperLayerActivations.get_shape()[-1]

print ("Number of channels in lower layer: %d" % (numChannelsLowerLayer))
print ("Number of channels in upper layer: %d" % (numChannelsUpperLayer))

with tf.variable_scope("clp"):
	# CLP output variable
	clp = tf.get_variable("clp", initializer=tf.zeros([numChannelsLowerLayer * numChannelsUpperLayer]), dtype=tf.float32)

i = tf.constant(0)
while_condition = lambda i: tf.less(i, numChannelsUpperLayer)
def loop_body(i):
	# Load the corresponding channel from upper layer
	upperLayerFeatureMap = tf.expand_dims(upperLayerActivations[:, :, :, i], -1)
	
	# Perform the multiplication
	c = tf.to_float(lowerLayerActivations * upperLayerFeatureMap)

	# Reduce sum
	c = tf.reduce_sum(c, axis=list(range(len(upperLayerActivations.get_shape())-1)))

	# Normalize the feature vector c
	c = c / (1e-7 + tf.norm(c))

	# print(c.get_shape())

	# Assign it to clp
	with tf.variable_scope("clp", reuse=True):
		clp = tf.get_variable("clp", initializer=tf.zeros([numChannelsLowerLayer * numChannelsUpperLayer]), dtype=tf.float32)
	assignOp = clp[i * numChannelsLowerLayer : (i+1) * numChannelsLowerLayer].assign(c)
	with tf.control_dependencies([assignOp]):
		# Increment i
		return [tf.add(i, 1)]

loopNode = tf.while_loop(while_condition, loop_body, [i])

with tf.control_dependencies([loopNode]):
	# Standardize the feature vector from CLP
	mean, var = tf.nn.moments(clp, axes=[0])
	clpVector = (clp - mean) / var

	# Signed normalization
	clpVector = tf.sqrt(tf.abs(clpVector)) * tf.sign(clpVector)

	# Normalize using the norm
	clpVector = clpVector / (1e-7 + tf.norm(clpVector))

# GPU config
config = tf.ConfigProto()
config.gpu_options.allow_growth=True

with tf.Session(config=config) as sess:
	# Initialize all vars
	sess.run(tf.global_variables_initializer())
	sess.run(tf.local_variables_initializer())

	# Write the graph to file
	summaryWriter = tf.summary.FileWriter("./logs", graph=tf.get_default_graph())

	# Restore the model params
	checkpointFileName = resnet_checkpoint_file if options.model == "ResNet" else inc_res_v2_checkpoint_file
	print ("Restoring weights from file: %s" % (checkpointFileName))

	# 'Saver' op to save and restore all the variables
	saver = tf.train.Saver(variables_to_restore)
	saver.restore(sess, checkpointFileName)

	# Initialize the dataset iterator
	sess.run(iterator.initializer)

	imFeatures = []
	imNames = []
	imLabels = []
	imSplit = []
	try:
		step = 0
		while True:
			start_time = time.time()

			namesOut, labelsOut, splitOut, clpOut = sess.run([inputBatchImageNames, inputBatchImageLabels, inputBatchImageSplit, clpVector])
			imFeatures.append(clpOut)
			imNames.extend(namesOut)
			imLabels.extend(labelsOut)
			imSplit.extend(splitOut)

			# print ("Image Name: %s" % (namesOut[0]))
			# print ("Image Label: %s" % (labelsOut[0]))

			duration = time.time() - start_time

			# Print an overview fairly often.
			if step % 100 == 0:
				print('Processing file # %d (%.3f sec)' % (step, duration))
			step += 1
	except tf.errors.OutOfRangeError:
		print('Done training for %d epochs, %d steps.' % (options.numEpochs, step))

# Save the computed features to pickle file
clpFeatures = np.array(imFeatures)
names = np.array(imNames)
labels = np.array(imLabels)
split = np.array(imSplit)

# Remove the previous variables
imFeatures = None
imNames = None
imLabels = None
imSplit = None

print ("List content sample")
print (names[:10])
print (labels[:10])
print (split[:10])

SAVE_FEATURES = False
if SAVE_FEATURES:
	print ("Saving features to file")
	np.save("/netscratch/siddiqui/CrossLayerPooling/data/imFeatures.npy", clpFeatures)
	np.save("/netscratch/siddiqui/CrossLayerPooling/data/imNames.npy", names)
	np.save("/netscratch/siddiqui/CrossLayerPooling/data/imLabels.npy", labels)
	print ("Saving complete!")

print ("Training Linear Model with Hinge Loss")
# clf = linear_model.SGDClassifier(n_jobs=-1)
clf = svm.LinearSVR(C=10.0)

# clf.fit(clpFeatures[:trainEndIndex], labels[:trainEndIndex])
trainData = (clpFeatures[split == TRAIN], labels[split == TRAIN])
print ("Number of images in training set: %d" % (trainData[0].shape[0]))
clf.fit(trainData[0], trainData[1])

print ("Training complete!")
if SAVE_FEATURES:
	with open('/netscratch/siddiqui/CrossLayerPooling/data/svm.pkl', 'wb') as fid:
		cPickle.dump(clf, fid)

print ("Evaluating validation accuracy")
validationData = (clpFeatures[split == VAL], labels[split == VAL])
print ("Number of images in validation set: %d" % (validationData[0].shape[0]))
clf.fit(validationData[0], validationData[1])
validationAccuracy = clf.score(validationData[0], validationData[1])
print ("Validation accuracy: %f" % (validationAccuracy))

print ("Evaluating test accuracy")
testData = (clpFeatures[split == TEST], labels[split == TEST])
print ("Number of images in test set: %d" % (testData[0].shape[0]))
clf.fit(testData[0], testData[1])
testAccuracy = clf.score(testData[0], testData[1])
print ("Test accuracy: %f" % (testAccuracy))

print ("Evaluation complete!")