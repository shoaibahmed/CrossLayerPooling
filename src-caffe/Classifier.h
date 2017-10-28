#ifndef CLASSIFIER_H_
#define CLASSIFIER_H_

#include "stdafx.h"
#include "Utils.h"


class Classifier
{
public:
	Classifier(int featureSize, int numClasses);
	// void trainClassifier(cv::Mat& X, cv::Mat& y);
	// void trainClassifier(cv::Ptr<cv::ml::TrainData> train_data);
	void trainSVM(cv::Mat& X, cv::Mat& y);
	void trainMLP(const std::vector<tiny_dnn::vec_t>& data, const std::vector<tiny_dnn::label_t>& labels);
	// void generatePredictions(cv::Mat& X, cv::Mat& y_pred);
	void generatePredictionsSVM(cv::Mat& X, cv::Mat& y_pred);
	void generatePredictionsMLP(const tiny_dnn::vec_t& data, tiny_dnn::label_t& label);
	float computeAccuracy(cv::Mat& X_test, cv::Mat& y_test);
	bool saveClassifier();
	bool loadClassifier();

private:
	cv::Ptr<cv::ml::SVM> svm;
	// tiny_dnn::network<tiny_dnn::sequential> net;
	// tiny_dnn::adam opt;

	// Auto member variable has to be static and const
	// static const std::shared_ptr<tiny_dnn::network<tiny_dnn::sequential> > net;
	// static const std::shared_ptr<tiny_dnn::adam> optimizer;
	tiny_dnn::network<tiny_dnn::sequential> net;
	tiny_dnn::adam optimizer;
};

#endif