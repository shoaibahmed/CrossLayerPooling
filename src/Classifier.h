#ifndef CLASSIFIER_H_
#define CLASSIFIER_H_

#include "stdafx.h"
#include "Utils.h"


class Classifier
{
public:
	Classifier();
	void trainClassifier(cv::Mat& X, cv::Mat& y);
	void trainClassifier(cv::Ptr<cv::ml::TrainData> train_data);
	void generatePredictions(cv::Mat& X, cv::Mat& y_pred);
	float computeAccuracy(cv::Mat& X_test, cv::Mat& y_test);
	bool saveClassifier();
	bool loadClassifier();

private:
	// cv::ml::SVM::CvSVMParams* params;
	cv::Ptr<cv::ml::SVM> svm;
};

#endif