#ifndef FEATURE_EXTRACTOR_H_
#define FEATURE_EXTRACTOR_H_

#include "stdafx.h"
#include "Utils.h"

using namespace caffe;  // NOLINT(build/namespaces)
using namespace cv;

class FeatureExtractor 
{
public:
	FeatureExtractor(const string& model_file, const string& trained_file);
	void extractFeatures(const cv::Mat& img, cv::Mat& cross_pooled_features);
	int getCrossPooledFeaturesSize(int num_comp_pca);

private:
	void wrapInputLayer(std::vector<cv::Mat>* input_channels);
	void preprocess(const cv::Mat& img, std::vector<cv::Mat>* input_channels);

	shared_ptr<Net<float> > net_;
	cv::Size input_geometry_;
	int num_channels_;
	int region_size_padding;
};

#endif