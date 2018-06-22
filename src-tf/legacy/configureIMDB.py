import os
import random
import shutil
from os import listdir
from os.path import isfile, join
from optparse import OptionParser

TRAIN = 0
VAL = 1
TEST = 2

def traverseDirectory(options):
	dataFile = open(options.dataOutputFile, 'w')
	elaboratedDataFile = open(options.dataOutputFile + "-full.txt", 'w')
	classes = {}

	counter = 0
	keysCounter = 0
	for root, dirs, files in os.walk(options.rootDirectory):
		path = root.split('/')
		print "Directory:", os.path.basename(root)
		for file in files:
			if file.endswith(options.searchString):
				fileName = str(os.path.abspath(os.path.join(root, file))).encode('string-escape')
				fileNameList = fileName.split(os.sep) 

				# Class Name and Number
				className = fileNameList[-2]
				if className in classes:
					pass
				else:
					classes[className] = keysCounter
					keysCounter += 1
				classNumber = classes[className]

				# Create the class folder is non-existent
				outputFileDirectory = str(os.path.abspath(os.path.join(options.imagesOutputDirectory, className)))

				imageClass = classNumber
				imageSplit = TRAIN if "train" in fileName else VAL if "validation" in fileName else TEST
				splitName = "train" if "train" in fileName else "validation" if "validation" in fileName else "test"
				dataFile.write(fileName + ' ' + str(imageClass) + ' ' + str(imageSplit) + '\n')
				elaboratedDataFile.write(fileName + ' ' + className + ' ' + splitName + '\n')

				counter += 1

	dataFile.close()
	elaboratedDataFile.close()

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
	parser.add_option("--dataOutputFile", action="store", type="string", dest="dataOutputFile", default="data.txt", help="Name of data file containing the image names, label and data split")
	parser.add_option("--classesOutputFile", action="store", type="string", dest="classesOutputFile", default="classes.txt", help="Name of classes text file")
	parser.add_option("--imagesOutputDirectory", action="store", type="string", dest="imagesOutputDirectory", default="../images", help="Images Output Directory")

	# Parse command line options
	(options, args) = parser.parse_args()

	traverseDirectory(options)

	print "Done"
