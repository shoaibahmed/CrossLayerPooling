import tensorflow as tf
slim = tf.contrib.slim

import pandas
import numpy as np

from optparse import OptionParser
import wget
import tarfile
import os
import cv2
import time
import shutil

from sklearn import linear_model
from sklearn import svm
from sklearn.decomposition import PCA

TRAIN = 0
VAL = 1
TEST = 2

import sys

if sys.version_info[0] == 3:
	print ("Using Python 3")
	import pickle
else:
	print ("Using Python 2")
	import cPickle as pickle

# Command line options
parser = OptionParser()

parser.add_option("-m", "--modelName", action="store", dest="modelName", default="ResNet-152", choices=["NASNet", "IncResV2", "ResNet-152"], help="Name of the model to be used for Cross-layer pooling")
parser.add_option("-t", "--trainModel", action="store_true", dest="trainModel", default=False, help="Train model")
parser.add_option("-c", "--testModel", action="store_true", dest="testModel", default=False, help="Test model")
parser.add_option("-d", "--debug", action="store_true", dest="debug", default=False, help="Debug flag to enable debugging (high verbosity)")

parser.add_option("--batchSize", action="store", type="int", dest="batchSize", default=1, help="Batch size to be used")
parser.add_option("--useSGD", action="store_true", dest="useSGD", default=False, help="Use SGD instead of a Convex Optimizer")
parser.add_option("--useLabelId", action="store_true", dest="useLabelId", default=False, help="Use label ID instead of the class name")
parser.add_option("--numEpochs", action="store", type="int", dest="numEpochs", default=1, help="Number of epochs")
parser.add_option("--imageWidth", action="store", type="int", dest="imageWidth", default=224, help="Image width for feeding into the network")
parser.add_option("--imageHeight", action="store", type="int", dest="imageHeight", default=224, help="Image height for feeding into the network")
parser.add_option("--imageChannels", action="store", type="int", dest="imageChannels", default=3, help="Number of channels in the image")

parser.add_option("--featureSpacing", action="store", type="int", dest="featureSpacing", default=3, help="Number of channels in the feautre vector to skip from all sides")
parser.add_option("--localRegionSize", action="store", type="int", dest="localRegionSize", default=1, help="Filter size for extraction of lower layer features")
parser.add_option("--performPCAOnFeatures", action="store_true", dest="performPCAOnFeatures", default=False, help="Perform PCA on the computed features")

parser.add_option("--useImageNetMean", action="store_true", dest="useImageNetMean", default=False, help="Use Image Net mean for normalization")
parser.add_option("--saveFeatures", action="store_true", dest="saveFeatures", default=False, help="Whether to save computed features")

parser.add_option("--rootDirectory", action="store", type="string", dest="rootDirectory", default="./data/Plant_Seedings/", help="Directory containing the data")
parser.add_option("--pretrainedModelsDir", action="store", type="string", dest="pretrainedModelsDir", default="./pretrained/", help="Directory containing the pretrained models")
parser.add_option("--outputDir", action="store", type="string", dest="outputDir", default="./output/", help="Ouput directory to store the output")
parser.add_option("--logsDir", action="store", type="string", dest="logsDir", default="./logs", help="Directory for saving logs")

parser.add_option("--numParallelLoaders", action="store", type="int", dest="numParallelLoaders", default=8, help="Number of parallel loaders to be used for data loading")

# Parse command line options
(options, args) = parser.parse_args()
print (options)

# Check if the pretrained directory exists
if not os.path.exists(options.pretrainedModelsDir):
	print ("Warning: Pretrained models directory not found!")
	os.makedirs(options.pretrainedModelsDir)
	assert os.path.exists(options.pretrainedModelsDir)

# Clone the repository if not already existent
if not os.path.exists(os.path.join(options.pretrainedModelsDir, "models/research/slim")):
	print ("Cloning TensorFlow models repository")
	import git # gitpython

	class Progress(git.remote.RemoteProgress):
		def update(self, op_code, cur_count, max_count=None, message=''):
			print (self._cur_line)

	git.Repo.clone_from("https://github.com/tensorflow/models.git", os.path.join(options.pretrainedModelsDir, "models"), progress=Progress())
	print ("Repository sucessfully cloned!")

# Add the path to the tensorflow models repository
sys.path.append(os.path.join(options.pretrainedModelsDir, "models/research/slim"))
sys.path.append(os.path.join(options.pretrainedModelsDir, "models/research/slim/nets"))

import inception_resnet_v2
import resnet_v1
import nasnet.nasnet as nasnet

# Import the model
if options.modelName == "NASNet":
	print ("Downloading pretrained NASNet model")
	nasCheckpointFile = checkpointFileName = os.path.join(options.pretrainedModelsDir, options.modelName, 'model.ckpt')
	if not os.path.isfile(nasCheckpointFile + '.index'):
		# Download file from the link
		url = 'https://storage.googleapis.com/download.tensorflow.org/models/nasnet-a_large_04_10_2017.tar.gz'
		fileName = wget.download(url, options.pretrainedModelsDir)
		print ("File downloaded: %s" % fileName)

		# Extract the tar file
		tar = tarfile.open(fileName)
		tar.extractall(path=os.path.join(options.pretrainedModelsDir, options.modelName))
		tar.close()

	# Update image sizes
	options.imageHeight = options.imageWidth = 331

elif options.modelName == "IncResV2":
	print ("Downloading pretrained Inception ResNet v2 model")
	incResV2CheckpointFile = checkpointFileName = os.path.join(options.pretrainedModelsDir, options.modelName, 'inception_resnet_v2_2016_08_30.ckpt')
	if not os.path.isfile(incResV2CheckpointFile):
		# Download file from the link
		url = 'http://download.tensorflow.org/models/inception_resnet_v2_2016_08_30.tar.gz'
		fileName = wget.download(url, options.pretrainedModelsDir)
		print ("File downloaded: %s" % fileName)

		# Extract the tar file
		tar = tarfile.open(fileName)
		tar.extractall(path=os.path.join(options.pretrainedModelsDir, options.modelName))
		tar.close()

	# Update image sizes
	options.imageHeight = options.imageWidth = 299

elif options.modelName == "ResNet-152":
	print ("Downloading pretrained ResNet-152 model")
	resNet152CheckpointFile = checkpointFileName = os.path.join(options.pretrainedModelsDir, options.modelName, 'resnet_v1_152.ckpt')
	if not os.path.isfile(resNet152CheckpointFile):
		# Download file from the link
		url = 'http://download.tensorflow.org/models/resnet_v1_152_2016_08_28.tar.gz'
		fileName = wget.download(url, options.pretrainedModelsDir)
		print ("File downloaded: %s" % fileName)

		# Extract the tar file
		tar = tarfile.open(fileName)
		tar.extractall(path=os.path.join(options.pretrainedModelsDir, options.modelName))
		tar.close()

	# Update image sizes
	options.imageHeight = options.imageWidth = 224

else:
	print ("Error: Model %s not found!" % options.modelName)
	exit (-1)

# Check if the pretrained directory exists
if os.path.exists(options.outputDir):
	print ("Warning: Output directory already exists. Removing previous directory.")
	shutil.rmtree(options.outputDir)

os.makedirs(options.outputDir)

# Define params
IMAGENET_MEAN = [123.68, 116.779, 103.939] # RGB
USE_IMAGENET_MEAN = options.useImageNetMean
REGION_SIZE_PADDING = int((options.localRegionSize - 1) / 2)
LOCAL_REGION_DIM = options.localRegionSize * options.localRegionSize

# Reads an image from a file, decodes it into a dense tensor
def parseFunction(filename, label, split):
	imageString = tf.read_file(filename)
	# img = tf.image.decode_image(imageString, channels=3)
	img = tf.image.decode_jpeg(imageString, channels=3) # decode_PNG and Decode_jpeg now decodes all shapes

	# img = tf.reshape(img, [options.imageHeight, options.imageWidth, options.imageChannels])
	img = tf.image.resize_images(img, [options.imageHeight, options.imageWidth])
	img.set_shape([options.imageHeight, options.imageWidth, options.imageChannels])
	img = tf.cast(img, tf.float32) # Convert to float tensor
	print (img.shape)
	return img, filename, label, split

# A vector of filenames
print ("Loading data from directory: %s" % (options.rootDirectory))
imNames = []
imLabels = []
imSplit = []
imClasses = {}
clsInstances = {}
clsCounter = 0
for root, dirs, files in os.walk(options.rootDirectory):
	label = -1 if options.useLabelId else 'N/A'
	clsName = 'N/A'
	if "train" in root:
		split = TRAIN

		# Use the last part in the directory tree as the class label
		clsName = root.split(os.sep)[-1]
		if clsName == "train":
			continue
		if clsName not in imClasses:
			imClasses[clsName] = clsCounter
			clsCounter += 1
		
		if options.useLabelId:
			label = imClasses[clsName]
		else:
			label = clsName
	elif "test" in root:
		split = TEST
	else:
		print ("Error: Unable to infer the split from directory (%s). Skipping directory." % (root))
		continue
		# print ("Error: Unable to infer the split (%s)" % (root))
		# exit (-1)

	# Class counters
	if clsName not in clsInstances:
		clsInstances[clsName] = 0

	if options.useLabelId:
		print ("Directory: %s | Split: %d | Class name: %s | Class label: %d" % (os.path.basename(root), split, clsName, label))
	else:
		print ("Directory: %s | Split: %d | Class name: %s | Class label: %s" % (os.path.basename(root), split, clsName, label))
	for file in files:
		isImage = any([True if file.endswith(ext) else False for ext in [".jpg", ".jpeg", ".png"]])
		if isImage:
			fileName = str(os.path.abspath(os.path.join(root, file)))
			imNames.append(fileName)
			imLabels.append(label)
			imSplit.append(split)
			clsInstances[clsName] += 1
			if options.debug:
				if options.useLabelId:
					print ("File: %s | Split: %d | Class name: %s | Class label: %d" % (fileName, split, clsName, label))
				else:
					print ("File: %s | Split: %d | Class name: %s | Class label: %s" % (fileName, split, clsName, label))

# Class name to index
imClassesToIdx = {v: k for k, v in imClasses.items()}

print ("************** Dataset statistics **************")
imSplit = np.array(imSplit)
numFiles = imSplit.shape[0]
numTrainExamples = np.sum(imSplit == TRAIN)
numValExamples = np.sum(imSplit == VAL)
numTestExamples = np.sum(imSplit == TEST)
print ("Total examples: %d | Training examples: %d | Validation examples: %d | Test examples: %d" % (numFiles, numTrainExamples, numValExamples, numTestExamples))
print ("Class instances:", clsInstances)

imNames = np.array(imNames)
testImageNames = imNames[imSplit == TEST]
print ("Test image names:", testImageNames[:3])
imNames = tf.constant(imNames)
imLabels = tf.constant(imLabels)
imSplit = tf.constant(imSplit)

dataset = tf.data.Dataset.from_tensor_slices((imNames, imLabels, imSplit))
dataset = dataset.map(parseFunction, num_parallel_calls=options.numParallelLoaders)
dataset = dataset.shuffle(buffer_size=numFiles, seed=0)
dataset = dataset.batch(options.batchSize)

iterator = dataset.make_initializable_iterator()

with tf.name_scope('Model'):
	# Data placeholders
	inputBatchImages, inputBatchImageNames, inputBatchImageLabels, inputBatchImageSplit = iterator.get_next()
	print ("Data shape: %s" % str(inputBatchImages.get_shape()))

	if (options.modelName == "IncResV2") or (options.modelName == "NASNet"):
		scaledInputBatchImages = tf.scalar_mul((1.0/255), inputBatchImages)
		scaledInputBatchImages = tf.subtract(scaledInputBatchImages, 0.5)
		scaledInputBatchImages = tf.multiply(scaledInputBatchImages, 2.0)

		# Create model
		if options.modelName == "IncResV2":
			arg_scope = inception_resnet_v2.inception_resnet_v2_arg_scope()
			with slim.arg_scope(arg_scope):
				logits, aux_logits, endPoints = inception_resnet_v2.inception_resnet_v2(scaledInputBatchImages, is_training=False)

				# Get the lower layer and upper layer activations
				lowerLayerActivations = endPoints["?"]
				upperLayerActivations = endPoints["?"]

			# Create list of vars to restore before train op
			variables_to_restore = slim.get_variables_to_restore(include=["InceptionResnetV2"])

		elif options.modelName == "NASNet":
			arg_scope = nasnet.nasnet_large_arg_scope()
			with slim.arg_scope(arg_scope):
				# logits, endPoints = nasnet.build_nasnet_large(scaledInputBatchImages, is_training=options.trainModel, num_classes=numClasses)
				logits, endPoints = nasnet.build_nasnet_large(scaledInputBatchImages, is_training=False)

				# TODO: Get the lower layer and upper layer activations
				lowerLayerActivations = endPoints["?"]
				upperLayerActivations = endPoints["?"]

		else:
			print ("Error: Model %s not found!" % options.modelName)
			exit (-1)

	elif options.modelName == "ResNet-152":
		if USE_IMAGENET_MEAN:
			print (inputBatchImages.shape)
			channels = tf.split(axis=3, num_or_size_splits=options.imageChannels, value=inputBatchImages)
			for i in range(options.imageChannels):
				channels[i] -= IMAGENET_MEAN[i]
			processedInputBatchImages = tf.concat(axis=3, values=channels)
			print (processedInputBatchImages.shape)
		else:
			imageMean = tf.reduce_mean(inputBatchImages, axis=[1, 2], keepdims=True)
			print ("Image mean shape: %s" % str(imageMean.shape))
			processedInputBatchImages = inputBatchImages - imageMean

		# Create model
		arg_scope = resnet_v1.resnet_arg_scope()
		with slim.arg_scope(arg_scope):
			logits, endPoints = resnet_v1.resnet_v1_152(processedInputBatchImages, is_training=False)

			# Get the lower layer and upper layer activations
			lowerLayerActivations = endPoints["resnet_v1_152/block3/unit_15/bottleneck_v1"]
			upperLayerActivations = endPoints["resnet_v1_152/block3/unit_20/bottleneck_v1"]

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

# Crop the features
if options.localRegionSize > 1:
	lowerLayerActivations = tf.extract_image_patches(lowerLayerActivations, [1, options.localRegionSize, options.localRegionSize, 1], [1, 1, 1, 1], [1, 1, 1, 1], 'SAME')

lowerLayerActivations = lowerLayerActivations[:, options.featureSpacing : -options.featureSpacing, options.featureSpacing : -options.featureSpacing, :]
upperLayerActivations = upperLayerActivations[:, options.featureSpacing : -options.featureSpacing, options.featureSpacing : -options.featureSpacing, :]

numChannelsLowerLayer = lowerLayerActivations.get_shape()[-1]
numChannelsUpperLayer = upperLayerActivations.get_shape()[-1]

print ("Lower layer shape: %s" % str(lowerLayerActivations.get_shape()))
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

# GPU config
config = tf.ConfigProto()
config.gpu_options.allow_growth=True

with tf.Session(config=config) as sess:
	# Initialize all vars
	sess.run(tf.global_variables_initializer())
	sess.run(tf.local_variables_initializer())

	# Write the graph to file
	summaryWriter = tf.summary.FileWriter(options.logsDir, graph=tf.get_default_graph())

	# Restore the model params
	checkpointFileName = resNet152CheckpointFile if options.modelName == "ResNet-152" else incResV2CheckpointFile if options.modelName == "IncResV2" else nasCheckpointFile if options.modelName == "NASNet" else None
	if checkpointFileName == None:
		print ("Error: Unable to find model checkpoint file (%s)" % options.modelName)
		exit (-1)

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
				print("Processing file # %d (%.3f sec)" % (step, duration))
			step += 1
	except tf.errors.OutOfRangeError:
		print("Done training for %d epochs, %d steps." % (options.numEpochs, step))

# Save the computed features to pickle file
clpFeatures = np.array(imFeatures)
names = np.array(imNames)
labels = np.array(imLabels)
split = np.array(imSplit)

if options.performPCAOnFeatures:
	# Compress the whole CLP features
	pca = PCA(n_components=1024)
	print ("Dimensions before reduction:", clpFeatures.shape)
	pca.fit(clpFeatures[split == TRAIN]) # Fit on train data
	clpFeatures = pca.transform(clpFeatures) # Perform dimensioanlity reduction on all the features
	print ("Dimensions after reduction:", clpFeatures.shape)

# Remove the previous variables
imFeatures = None
imNames = None
imLabels = None
imSplit = None

print ("List content sample")
print (names[:10])
print (labels[:10])
print (split[:10])

if options.saveFeatures:
	print ("Saving features to file")
	np.save(os.path.join(options.outputDir, "imFeatures.npy"), clpFeatures)
	np.save(os.path.join(options.outputDir, "imNames.npy"), names)
	np.save(os.path.join(options.outputDir, "imLabels.npy"), labels)
	np.save(os.path.join(options.outputDir, "imSplit.npy"), split)
	np.save(os.path.join(options.outputDir, "imClasses.npy"), imClasses)
	print ("Saving complete!")

if options.useSGD:
	print ("Training Linear SVM with Hinge Loss using SGD Optimizer")
	clf = linear_model.SGDClassifier(loss="hinge", penalty="l2", n_jobs=-1)
else:
	print ("Training Linear SVM with Hinge Loss using Convex Optimizer")
	clf = svm.LinearSVC(C=10.0)

# clf.fit(clpFeatures[:trainEndIndex], labels[:trainEndIndex])
trainData = (clpFeatures[split == TRAIN], labels[split == TRAIN])
print ("Number of images in training set: %d" % (trainData[0].shape[0]))
clf.fit(trainData[0], trainData[1])
print ("Training complete!")
trainAccuracy = clf.score(trainData[0], trainData[1])
print ("Train accuracy: %f" % (trainAccuracy))

if options.saveFeatures:
	with open(os.path.join(options.outputDir, "svm.pkl"), "wb") as fid:
		pickle.dump(clf, fid)

if numValExamples > 0:
	print ("Evaluating validation accuracy")
	validationData = (clpFeatures[split == VAL], labels[split == VAL])
	print ("Number of images in validation set: %d" % (validationData[0].shape[0]))
	validationAccuracy = clf.score(validationData[0], validationData[1])
	print ("Validation accuracy: %f" % (validationAccuracy))

if numTestExamples > 0:
	print ("Evaluating test accuracy")
	testData = (clpFeatures[split == TEST], labels[split == TEST])
	print ("Number of images in test set: %d" % (testData[0].shape[0]))
	testAccuracy = clf.score(testData[0], testData[1])
	print ("Test accuracy: %f" % (testAccuracy))

	# Save test predictions to file
	print ("Number of test images: %d" % len(testImageNames))
	testPredictions = clf.predict(testData[0])
	with open(os.path.join(options.outputDir, "output.pkl"), "wb") as fid:
		pickle.dump([testImageNames, testPredictions], fid)

	df = pandas.read_csv(os.path.join(options.rootDirectory, "sample_submission.csv"))
	# df = df.set_index("file", drop = False)

	for idx, imageName in enumerate(testImageNames):
		pred = testPredictions[idx]
		if not options.useLabelId:
			pred = pred.decode("utf-8")
		_, imageName = os.path.split(imageName) # Crop the complete path name
		df.ix[df["file"] == imageName, "species"] = imClassesToIdx[pred] if options.useLabelId else pred

	df.to_csv(os.path.join(options.outputDir, "submission.csv"), index=False, encoding='utf8')

	# with open(os.path.join(options.outputDir, "predictions.csv"), "w") as file:
	# 	file.write("%s,%s\n" % ("file", "species"))
	# 	for idx, imageName in enumerate(testImageNames):
	# 		pred = testPredictions[idx]
	# 		if not options.useLabelId:
	# 			pred = pred.decode("utf-8")
	# 		_, imageName = os.path.split(imageName) # Crop the complete path name
	# 		if options.useLabelId:
	# 			file.write("%s,%s,%d\n" % (imageName, imClassesToIdx[pred], pred))
	# 		else:
	# 			file.write("%s,%s\n" % (imageName, pred))

print ("Evaluation complete!")