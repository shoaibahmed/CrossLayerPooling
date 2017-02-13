#ifndef STDAFX_H_
#define STDAFX_H_

// General C++ includes
#include <iostream>
#include <string>
#include <chrono>
#include <fstream>

// Caffe includes
#include <caffe/caffe.hpp>

// OpenCV includes
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/ml.hpp>


// Pre-processor directives
#define DEBUG 0
#define TRAINED_FILE "./model/ResNet-152-model.caffemodel"
#define MODEL_FILE "./model/ResNet-152-deploy.prototxt"

#define TRAIN_EXAMPLE 1
#define VALIDATION_EXAMPLE 2
#define TEST_EXAMPLE 0
#define NUM_CLASSES 17

#define PERFORM_PCA false
#define NUM_COMP_PCA 512

#define PCA_FILE_NAME "pca.xml"
#define CLASSIFIER_NAME "mySVM.xml"

// Both blobs are with relu
#define USE_GPU true
#define PROFILE_MODE true
#define LOWER_BLOB_NAME "res4b15"
#define HIGHER_BLOB_NAME "res4b20"
#define REGION_SIZE 1 // 3x3 (require PCA)
#define FEATURE_SPACING 3
#define PERFORM_FEATURE_NORMALIZATION true

// http://caffe.berkeleyvision.org/tutorial/net_layer_blob.html
// Index(n, k, h, w) -> ((n * K + k) * H + h) * W + w
#define NUM_IMAGES 1 // N
#define HIGHER_BLOB_CHANNELS 1024 // (K)
#define LOWER_BLOB_CHANNELS 1024 // (K)
#define BLOB_SIZE 14 // (H, W)

#endif