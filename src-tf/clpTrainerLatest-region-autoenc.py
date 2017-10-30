import tensorflow as tf
slim = tf.contrib.slim

import math
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
parser.add_option("--numEpochs", action="store", type="int", dest="numEpochs", default=1, help="Number of epochs to be trained for RBM")

parser.add_option("--imageWidth", action="store", type="int", dest="imageWidth", default=224, help="Image width for feeding into the network")
parser.add_option("--imageHeight", action="store", type="int", dest="imageHeight", default=224, help="Image height for feeding into the network")
parser.add_option("--imageChannels", action="store", type="int", dest="imageChannels", default=3, help="Number of channels in the image")

parser.add_option("--featureSpacing", action="store", type="int", dest="featureSpacing", default=3, help="Number of channels in the feautre vector to skip from all sides")
parser.add_option("--localRegionSize", action="store", type="int", dest="localRegionSize", default=1, help="Filter size for extraction of lower layer features")
parser.add_option("--compressedLowerLayerSize", action="store", type="int", dest="compressedLowerLayerSize", default=64, help="Size of the compressed lower layer feature vector")
parser.add_option("--compressedFeatureVectorSize", action="store", type="int", dest="compressedFeatureVectorSize", default=512, help="Size of the compressed feature vector")

parser.add_option("--learningRate", action="store", type="float", dest="learningRate", default=1e-4, help="Learning rate for the auto-encoder")
parser.add_option("--regularizationParam", action="store", type="float", dest="regularizationParam", default=1e-5, help="Regularization param for L2 regularization")

parser.add_option("--useImageNetMean", action="store_true", dest="useImageNetMean", default=False, help="Use Image Net mean for normalization")
parser.add_option("--saveFeatures", action="store_true", dest="saveFeatures", default=False, help="Whether to save computed features")

parser.add_option("--dataFile", action="store", type="string", dest="dataFile", default="/netscratch/siddiqui/CrossLayerPooling/data/data.txt", help="Training data file")

# Parse command line options
(options, args) = parser.parse_args()
print (options)

# Define params
IMAGENET_MEAN = [123.68, 116.779, 103.939] # RGB
USE_IMAGENET_MEAN = options.useImageNetMean
SAVE_FEATURES = options.saveFeatures
FEATURE_SPACING = options.featureSpacing # Leave these features from each side
LOCAL_REGION_SIZE = options.localRegionSize # Use this filter size to capture features
REGION_SIZE_PADDING = int((LOCAL_REGION_SIZE - 1) / 2)
LOCAL_REGION_DIM = LOCAL_REGION_SIZE * LOCAL_REGION_SIZE

LEARNING_RATE = options.learningRate
REGULARIZATION_PARAM = options.regularizationParam

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
	numItemsInDataset = len(imageFileNames)
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
dataset = dataset.shuffle(buffer_size=numItemsInDataset, seed=0)
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

def lrelu(x, leak=0.2, name="lrelu"):
	"""Leaky rectifier.
	Parameters
	----------
	x : Tensor
		The tensor to apply the nonlinearity to.
	leak : float, optional
		Leakage parameter.
	name : str, optional
		Variable scope to use.
	Returns
	-------
	x : Tensor
		Output of the nonlinearity.
	"""
	with tf.variable_scope(name):
		f1 = 0.5 * (1 + leak)
		f2 = 0.5 * (1 - leak)
		return f1 * x + f2 * abs(x)

def convolutionalAutoEncoder(inputVector,
				n_filters,
				filter_sizes=[3, 3]):
	# projection = tf.layers.conv2d(inputs=inputVector, filters=1024, kernel_size=(1, 1), use_bias=False, name='projectionLayer_1')
	# projection = tf.layers.conv2d(inputs=tf.nn.relu(projection), filters=64, kernel_size=(1, 1), use_bias=False, name='projectionLayer_2')
	# reconstruction = tf.layers.conv2d_transpose(inputs=tf.nn.relu(projection), filters=1024, kernel_size=(1, 1), use_bias=False, name='projectionLayer_transpose_1')
	# reconstruction = tf.layers.conv2d_transpose(inputs=tf.nn.relu(reconstruction), filters=int(inputVector.get_shape()[-1]), kernel_size=(1, 1), use_bias=False, name='projectionLayer_transpose_2')

	current_input = inputVector
	n_filters[0]  = int(inputVector.get_shape()[-1])

	# Build the encoder
	encoder = []
	shapes = []
	lastLayerIdx = len(n_filters) - 2
	for layer_i, n_output in enumerate(n_filters[1:]):
		n_input = current_input.get_shape().as_list()[3]
		shapes.append(current_input.get_shape().as_list())
		W = tf.Variable(
			tf.random_uniform([
				filter_sizes[layer_i],
				filter_sizes[layer_i],
				n_input, n_output],
				-1.0 / math.sqrt(n_input),
				1.0 / math.sqrt(n_input)))
		b = tf.Variable(tf.zeros([n_output]))
		encoder.append(W)
		# output = lrelu(
		# 	tf.add(tf.nn.conv2d(
		# 		current_input, W, strides=[1, 1, 1, 1], padding='SAME'), b))
		output = tf.add(tf.nn.conv2d(current_input, W, strides=[1, 1, 1, 1], padding='SAME'), b)
		if lastLayerIdx != layer_i:
			output = lrelu(output)
		current_input = output

	# %%
	# store the latent representation
	z = current_input
	encoder.reverse()
	shapes.reverse()

	# %%
	# Build the decoder using the same weights
	for layer_i, shape in enumerate(shapes):
		W = encoder[layer_i]
		b = tf.Variable(tf.zeros([W.get_shape().as_list()[2]]))
		# output = lrelu(tf.add(
		# 	tf.nn.conv2d_transpose(
		# 		current_input, W,
		# 		tf.stack([tf.shape(inputVector)[0], shape[1], shape[2], shape[3]]),
		# 		strides=[1, 1, 1, 1], padding='SAME'), b))
		current_input = lrelu(current_input)
		output = tf.add(
			tf.nn.conv2d_transpose(
				current_input, W,
				tf.stack([tf.shape(inputVector)[0], shape[1], shape[2], shape[3]]),
				strides=[1, 1, 1, 1], padding='SAME'), b)
		current_input = output

	# %%
	# now have the reconstruction through the network
	y = current_input

	return z, y

# Crop the features
if LOCAL_REGION_SIZE > 1:
	lowerLayerActivations = tf.extract_image_patches(lowerLayerActivations, [1,LOCAL_REGION_SIZE,LOCAL_REGION_SIZE,1], [1,1,1,1], [1,1,1,1], 'SAME')

# Perform feature compression to make the computations tractable
lowerLayerActivationsCompressed, lowerLayerActivationsReconstructed = convolutionalAutoEncoder(lowerLayerActivations, n_filters=[1, options.compressedLowerLayerSize])

# compRepVer = tf.verify_tensor_all_finite(lowerLayerActivationsCompressed, "Infinite values on compressed activations", name=None)
# reconsRepVer = tf.verify_tensor_all_finite(lowerLayerActivationsReconstructed, "Infinite values on reconstructed activations", name=None)

print ("Lower layer shape: %s" % str(lowerLayerActivations.get_shape()))
print ("Lower layer reconstruction shape: %s" % str(lowerLayerActivationsReconstructed.get_shape()))
print ("Compressed lower layer shape: %s" % str(lowerLayerActivationsCompressed.get_shape()))

# Loss for the lower layers
reconstructionLossLowerLayer = tf.reduce_sum(tf.square(lowerLayerActivationsReconstructed - lowerLayerActivations)) + REGULARIZATION_PARAM * tf.reduce_sum(lowerLayerActivationsCompressed)

lowerLayerActivations = lowerLayerActivationsCompressed
if FEATURE_SPACING > 0:
	lowerLayerActivations = lowerLayerActivations[:, FEATURE_SPACING : -FEATURE_SPACING, FEATURE_SPACING : -FEATURE_SPACING, :]
	upperLayerActivations = upperLayerActivations[:, FEATURE_SPACING : -FEATURE_SPACING, FEATURE_SPACING : -FEATURE_SPACING, :]

numChannelsLowerLayer = lowerLayerActivations.get_shape()[-1]
numChannelsUpperLayer = upperLayerActivations.get_shape()[-1]

# print ("Lower layer shape: %s" % str(lowerLayerActivations.get_shape()))
print ("Upper layer shape: %s" % str(upperLayerActivations.get_shape()))

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
	assignOp = clp[i * (numChannelsLowerLayer) : (i+1) * (numChannelsLowerLayer)].assign(c)
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

def denseAutoEncoder(inputVector, dimensions):
	current_input = tf.expand_dims(inputVector, 0)
	dimensions[0]  = int(inputVector.get_shape()[-1])

	# %% Build the encoder
	encoder = []
	lastLayerIdx = len(dimensions) - 2
	for layer_i, n_output in enumerate(dimensions[1:]):
		with tf.variable_scope("encoder_" + str(layer_i + 1)):
			n_input = int(current_input.get_shape()[1])
			limit = math.sqrt(6.0 / (n_input + n_output))
			W = tf.Variable(tf.random_uniform([n_input, n_output], -limit, limit))
			b = tf.Variable(tf.zeros([n_output]))
			encoder.append(W)
			# out = tf.matmul(current_input, W) + b
			# lastLayer = n_output == len(dimensions) - 2
			# if not lastLayer:
			# 	output = lrelu(out, name="encoder_lrelu_" + str(layer_i + 1))
			output = tf.matmul(current_input, W) + b
			if lastLayerIdx != layer_i:
				output = lrelu(output)
			current_input = output

	# %% latent representation
	z = current_input
	encoder.reverse()
	current_input = lrelu(current_input, name="encoder_lrelu_" + str(len(dimensions) - 2 + 1))

	# %% Build the decoder using the SAME weights
	lastIndex = len(encoder) - 1
	for layer_i, n_output in enumerate(dimensions[:-1][::-1]):
		# with tf.variable_scope("encoder_" + str(layer_i + 1)):
		W = tf.transpose(encoder[layer_i])
		b = tf.Variable(tf.zeros([n_output]))
		# Add non-linearity if not the last layer
		# if layer_i != lastIndex:
		# 	out = tf.matmul(current_input, W) + b
		# 	output = lrelu(out, name="decoder_lrelu_" + str(layer_i + 1))
		# 	# output = lrelu(tf.matmul(current_input, W) + b, name="decoder_lrelu_" + str(layer_i + 1))
		# else:
		# 	output = tf.matmul(current_input, W) + b
		current_input = lrelu(current_input)
		output = tf.matmul(current_input, W) + b
		current_input = output

	return tf.squeeze(z), tf.squeeze(current_input)

# Perform feature compression to make the computations tractable
clpVectorCompressed, clpVectorReconstruction = denseAutoEncoder(clpVector, dimensions=[1, options.compressedFeatureVectorSize])

# Loss for the clp vector
reconstructionLossCLPVector = tf.reduce_sum(tf.square(clpVectorReconstruction - clpVector)) + REGULARIZATION_PARAM * tf.reduce_sum(clpVectorCompressed)

# Optimizer
loss = reconstructionLossLowerLayer + reconstructionLossCLPVector
optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss)

# GPU config
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
# config.log_device_placement=True

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

	imFeatures = []
	imNames = []
	imLabels = []
	imSplit = []

	for epoch in range(options.numEpochs + 1):
		generateOutputVectors = epoch == options.numEpochs

		# Initialize the dataset iterator
		sess.run(iterator.initializer)
		try:
			step = 0
			while True:
				start_time = time.time()

				if generateOutputVectors:
					namesOut, labelsOut, splitOut, clpOut = sess.run([inputBatchImageNames, inputBatchImageLabels, inputBatchImageSplit, clpVectorCompressed])
					assert (not np.isnan(clpOut).any())
					assert (not np.isinf(clpOut).any())
					imFeatures.append(clpOut)
					imNames.extend(namesOut)
					imLabels.extend(labelsOut)
					imSplit.extend(splitOut)

					duration = time.time() - start_time

					if step % 100 == 0:
						print('Processing file # %d (%.3f sec)' % (step, duration))
				else:
					currentLoss, _ = sess.run([loss, optimizer])
					duration = time.time() - start_time
					print ("Step: %d | Duration: %.3f seconds | Loss: %f" % (step, duration, currentLoss))

				step += 1
		except tf.errors.OutOfRangeError:
			print('Done training for %d epochs, %d steps.' % (epoch, step))

# Save the computed features to pickle file
clpFeatures = np.array(imFeatures)
names = np.array(imNames)
labels = np.array(imLabels)
split = np.array(imSplit)

print ("CLP Features shape: %s" % str(clpFeatures.shape))

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
clf = svm.LinearSVC(C=10.0)

# clf.fit(clpFeatures[:trainEndIndex], labels[:trainEndIndex])
trainData = (clpFeatures[split == TRAIN], labels[split == TRAIN])
print ("Number of images in training set: %d" % (trainData[0].shape[0]))
clf.fit(trainData[0], trainData[1])
print ("Training complete!")
trainAccuracy = clf.score(trainData[0], trainData[1])
print ("Train accuracy: %f" % (trainAccuracy))

if SAVE_FEATURES:
	with open('/netscratch/siddiqui/CrossLayerPooling/data/svm.pkl', 'wb') as fid:
		cPickle.dump(clf, fid)

print ("Evaluating validation accuracy")
validationData = (clpFeatures[split == VAL], labels[split == VAL])
print ("Number of images in validation set: %d" % (validationData[0].shape[0]))
validationAccuracy = clf.score(validationData[0], validationData[1])
print ("Validation accuracy: %f" % (validationAccuracy))

print ("Evaluating test accuracy")
testData = (clpFeatures[split == TEST], labels[split == TEST])
print ("Number of images in test set: %d" % (testData[0].shape[0]))
testAccuracy = clf.score(testData[0], testData[1])
print ("Test accuracy: %f" % (testAccuracy))

print ("Evaluation complete!")