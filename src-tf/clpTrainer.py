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

TRAIN = 1
VALIDATION = 2
TEST = 3

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

parser.add_option("--imageChannels", action="store", type="int", dest="imageChannels", default=3, help="Number of channels in the image")

parser.add_option("--trainRecordsFile", action="store", type="string", dest="trainRecordsFile", default="/netscratch/siddiqui/CrossLayerPooling/data/train.record", help="Training data file")
parser.add_option("--validationRecordsFile", action="store", type="string", dest="validationRecordsFile", default="/netscratch/siddiqui/CrossLayerPooling/data/val.record", help="Validation data file")
parser.add_option("--testRecordsFile", action="store", type="string", dest="testRecordsFile", default="/netscratch/siddiqui/CrossLayerPooling/data/test.record", help="Test data file")

# Parse command line options
(options, args) = parser.parse_args()
print (options)

IMAGENET_MEAN = [123.68, 116.779, 103.939] # RGB
USE_IMAGENET_MEAN = False

# Function for reading in examples from the TF-Records file
def read_and_decode(filename_queue):
	reader = tf.TFRecordReader()
	_, serialized_example = reader.read(filename_queue)
	features = tf.parse_single_example(
		serialized_example,
		# Defaults are not specified since both keys are required.
		features={
			'image/height': tf.FixedLenFeature([], tf.int64),
			'image/width': tf.FixedLenFeature([], tf.int64),
			'image/channels': tf.FixedLenFeature([], tf.int64),
			'image/filename': tf.FixedLenFeature([], tf.string),
			'image/source_id': tf.FixedLenFeature([], tf.string),
			'image/key/sha256': tf.FixedLenFeature([], tf.string),
			'image/encoded': tf.FixedLenFeature([], tf.string),
			'image/format': tf.FixedLenFeature([], tf.string),
			'image/class/text': tf.FixedLenFeature([], tf.string),
			'image/class/label': tf.FixedLenFeature([], tf.int64),
		})

	# Convert from a scalar string tensor (whose single string has
	# length mnist.IMAGE_PIXELS) to a uint8 tensor with shape
	# [mnist.IMAGE_PIXELS].
	# image = tf.image.decode_jpeg(features['image/encoded'], tf.uint8)
	image = tf.image.decode_image(features['image/encoded'], 3)
	# image = tf.decode_raw(features['image/encoded'], tf.uint8)
	height = tf.cast(features['image/height'], tf.int32)
	width = tf.cast(features['image/width'], tf.int32)
	channels = tf.cast(features['image/channels'], tf.int32)

	image_shape = tf.stack([height, width, 3])
	image = tf.reshape(image, image_shape)

	image = tf.image.resize_images(image, [224, 224])
	print (image.get_shape())
	image = tf.cast(image, tf.float32)

	# Convert label from a scalar uint8 tensor to an int32 scalar.
	label = tf.cast(features['image/class/label'], tf.int32)

	return image, label

def inputs(filename, batch_size, num_epochs):
	"""Reads input data num_epochs times.
	Args:
		train: Selects between the training (True) and validation (False) data.
		batch_size: Number of examples per returned batch.
		num_epochs: Number of times to read the input data, or 0/None to
		train forever.
	Returns:
		A tuple (images, labels), where:
		* images is a float tensor with shape [batch_size, mnist.IMAGE_PIXELS]
		in the range [-0.5, 0.5].
		* labels is an int32 tensor with shape [batch_size] with the true label,
		a number in the range [0, mnist.NUM_CLASSES).
		Note that an tf.train.QueueRunner is added to the graph, which
		must be run using e.g. tf.train.start_queue_runners().
	"""
	if not num_epochs: num_epochs = None
	# if datasetType == TRAIN:
	# 	filename = options.trainRecordsFile
	# elif datasetType == VALIDATION:
	# 	filename = options.validationRecordsFile
	# elif datasetType == TEST:
	# 	filename = options.testRecordsFile
	# else:
	# 	print ("Error: Unknown dataset type")
	# 	exit (-1)

	with tf.name_scope('input'):
		filename_queue = tf.train.string_input_producer([options.trainRecordsFile, options.validationRecordsFile, options.testRecordsFile], num_epochs=num_epochs)

		# Even when reading in multiple threads, share the filename
		# queue.
		# image, label = read_and_decode(filename_queue)
		image, label = read_and_decode(filename_queue)

		# Shuffle the examples and collect them into batch_size batches.
		# (Internally uses a RandomShuffleQueue.)
		# We run this in two threads to avoid being a bottleneck.
		images, sparse_labels = tf.train.batch(
			[image, label], batch_size=batch_size, num_threads=2,
			capacity=1000 + 3 * batch_size)
		# images, sparse_labels = tf.train.shuffle_batch(
		# 	[image, label], batch_size=batch_size, num_threads=2,
		# 	capacity=1000 + 3 * batch_size,
		# 	# Ensures a minimum amount of shuffling of examples.
		# 	min_after_dequeue=1000)

		return images, sparse_labels


with tf.name_scope('Model'):
	# Data placeholders
	# inputBatchImages = tf.placeholder(dtype=tf.float32, shape=[None, None, None, options.imageChannels], name="inputBatchImages")

	# Input images and labels.
	inputBatchImages, inputBatchLabels = inputs(filename=options.trainRecordsFile, batch_size=options.batchSize, num_epochs=options.numEpochs)

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
		variables_to_restore = slim.get_variables_to_restore(include=["resnet_v1"])

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
	clpVector = (clp - mean) / tf.sqrt(var)

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

	# Start input enqueue threads.
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(sess=sess, coord=coord)

	trainFeatures = []
	trainLabels = []
	try:
		step = 0
		while not coord.should_stop():
			start_time = time.time()

			labelsOut, clpOut = sess.run([inputBatchLabels, clpVector])
			trainFeatures.append(clpOut)
			trainLabels.append(labelsOut[0])

			duration = time.time() - start_time

			# Print an overview fairly often.
			if step % 100 == 0:
				print('Processing file # %d (%.3f sec)' % (step, duration))
			step += 1
	except tf.errors.OutOfRangeError:
		print('Done training for %d epochs, %d steps.' % (options.numEpochs, step))
	finally:
		# When done, ask the threads to stop.
		coord.request_stop()

		# Wait for threads to finish.
		coord.join(threads)

# Save the computed features to pickle file
clpFeatures = np.array(trainFeatures)
labels = np.array(trainLabels)

SAVE_FEATURES = True
if SAVE_FEATURES:
	print ("Saving features to file")
	np.save("/netscratch/siddiqui/CrossLayerPooling/data/trainFeatures.npy", clpFeatures)
	np.save("/netscratch/siddiqui/CrossLayerPooling/data/trainLabels.npy", labels)
	print ("Saving complete!")

# Divide the dataset into train, test and validation set
trainEndIndex = 1309
validationEndIndex = trainEndIndex + 237
testEndIndex = validationEndIndex + 663
assert(testEndIndex == 2209)

print ("Training Linear Model with Hinge Loss")
clf = linear_model.SGDClassifier(n_jobs=-1)
# clf = svm.LinearSVR(C=10.0)

clf.fit(clpFeatures[:trainEndIndex], labels[:trainEndIndex])

# from liquidSVM import *
# model = mcSVM(banana.train, mcType="OvA_hinge", useCells=True)

print ("Training complete!")
with open('/netscratch/siddiqui/CrossLayerPooling/data/svm.pkl', 'wb') as fid:
	cPickle.dump(clf, fid)

print ("Evaluating validation accuracy")
validationAccuracy = clf.score(clpFeatures[trainEndIndex:validationEndIndex], labels[trainEndIndex:validationEndIndex])
print ("Validation accuracy: %f" % (validationAccuracy))

print ("Evaluating test accuracy")
testAccuracy = clf.score(clpFeatures[validationEndIndex:testEndIndex], labels[validationEndIndex:testEndIndex])
print ("Test accuracy: %f" % (testAccuracy))

print ("Evaluation complete!")