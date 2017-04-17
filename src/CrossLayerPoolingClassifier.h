#ifndef CROSS_LAYER_POOLING_CLASSIFIER_H_
#define CROSS_LAYER_POOLING_CLASSIFIER_H_

#include "stdafx.h"
#include "FeatureExtractor.h"
#include "Classifier.h"
#include "Utils.h"

class CrossLayerPoolingClassifier
{
public:
	CrossLayerPoolingClassifier();
	void trainClassifier(std::string data_dir, std::string images_dir);
	int generatePrediction(cv::Mat& img);

private:
	Classifier* classifier;
	FeatureExtractor* feature_extractor;
	cv::PCA* pca;
	bool pca_data_loaded;
	bool classifier_loaded;

	// Methods
	void savePCAData();
	bool loadPCAData();
};

#endif