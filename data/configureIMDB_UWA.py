import os
import random
import shutil
from os import listdir
from os.path import isfile, join
from optparse import OptionParser

def traverseDirectory(options):
	imagesFile = open(options.imagesOutputFile, 'w')
	imagesClassFile = open(options.imagesClassOutputFile, 'w')
	trainTestSplitFile = open(options.trainTestSplitOutputFile, 'w')
	classes = {}

	# Create images directory (remove if already exists)
	if os.path.isdir(options.imagesOutputDirectory):
		shutil.rmtree(options.imagesOutputDirectory)

	os.mkdir(options.imagesOutputDirectory)

	counter = 1
	keysCounter = 1
	for root, dirs, files in os.walk(options.rootDirectory):
		path = root.split('/')
		print "Directory:", os.path.basename(root)
		for file in files:
			if file.endswith(options.searchString):
				fileName = str(os.path.abspath(os.path.join(root, file))).encode('string-escape')
				
				# Extract class name
				if os.name == 'nt':
					separator = '\\' # Windows
				else:
					separator = '/' # Ubuntu

				fileNameList = fileName.split(separator) 

				# Class Name and Number
				className = fileNameList[9]
				if className in classes:
					pass
				else:
					classes[className] = keysCounter
					keysCounter += 1
				classNumber = classes[className]

				# Create the class folder is non-existent
				outputFileDirectory = str(os.path.abspath(options.imagesOutputDirectory + separator + className))

				if not(os.path.isdir(outputFileDirectory)):
					os.mkdir(outputFileDirectory)

				# Copy the file in the class folder
				'''
				if isWindows:
					outputFileDirectory = outputFileDirectory + '\\' + className + '_' + str(counter) + '.jpg'
				else:
					outputFileDirectory = outputFileDirectory + '/' + className + '_' + str(counter) + '.jpg'
				'''

				shutil.copy2(fileName, outputFileDirectory)

				imagesFile.write(str(counter) +  ' ' + className + separator + file + '\n')
				imagesClassFile.write(str(counter) + ' ' + str(classNumber) + '\n')

				# Specify train test split using random number
				if "train" in fileName:
					trainTestSplitFile.write(str(counter) + ' 1' + '\n')
				elif "validation" in fileName:
					trainTestSplitFile.write(str(counter) + ' 2' + '\n')
				else:
					trainTestSplitFile.write(str(counter) + ' 0' + '\n')

				counter += 1

	imagesFile.close()
	imagesClassFile.close()
	trainTestSplitFile.close()

	# Write classes to file
	classesFile = open(options.classesOutputFile, 'w')
	for className in classes:
		classesFile.write(str(classes[className]) + ' ' + className + '\n')

		counter += 1

	classesFile.close()

if __name__ == "__main__":

	# Command line options
	parser = OptionParser()
	parser.add_option("-d", "--dir", action="store", type="string", dest="rootDirectory", default=u".", help="Root directory to be searched")
	parser.add_option("--searchString", action="store", type="string", dest="searchString", default=".jpg", help="Criteria for finding relevant files")
	parser.add_option("--imagesOutputFile", action="store", type="string", dest="imagesOutputFile", default="images.txt", help="Name of images text file")
	parser.add_option("--classesOutputFile", action="store", type="string", dest="classesOutputFile", default="classes.txt", help="Name of classes text file")
	parser.add_option("--imagesOutputDirectory", action="store", type="string", dest="imagesOutputDirectory", default="../images", help="Images Output Directory")
	parser.add_option("--imagesClassOutputFile", action="store", type="string", dest="imagesClassOutputFile", default="image_class_labels.txt", help="Label of each image specified in images file")
	parser.add_option("--trainTestSplitOutputFile", action="store", type="string", dest="trainTestSplitOutputFile", default="train_test_split.txt", help="Output file specifying the train test split")

	# Parse command line options
	(options, args) = parser.parse_args()

	traverseDirectory(options)

	print "Done"
