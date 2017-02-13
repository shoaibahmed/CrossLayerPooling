#include "Classifier.h"

Classifier::Classifier()
{
	svm = cv::ml::SVM::create();

	// Specify SVM params
	svm->setType(cv::ml::SVM::C_SVC);
	svm->setKernel(cv::ml::SVM::LINEAR);
	svm->setC(10);
	svm->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER, 100, 1e-6));
}

void Classifier::trainClassifier(cv::Mat& X, cv::Mat& y)
{
	// cv::Ptr<cv::ml::TrainData> tempData = cv::ml::TrainData::create(X, cv::ml::ROW_SAMPLE, y);
	// Assign the SVM parameters to the most accurate result
	// svm->trainAuto(tempData);

	// Train classifier
	// Each training sample occupies a column of samples
	svm->train(X, cv::ml::ROW_SAMPLE, y);

	// Save trained classifier
	saveClassifier();
}

void Classifier::trainClassifier(cv::Ptr<cv::ml::TrainData> train_data)
{
	// cv::Ptr<cv::ml::TrainData> tempData = cv::ml::TrainData::create(X, cv::ml::ROW_SAMPLE, y);
	// Assign the SVM parameters to the most accurate result
	// svm->trainAuto(tempData);

	// Train classifier
	// Each training sample occupies a column of samples
	svm->train(train_data);

	// Save trained classifier
	saveClassifier();
}

void Classifier::generatePredictions(cv::Mat& X, cv::Mat& y_pred)
{
	svm->predict(X, y_pred);
}

float Classifier::computeAccuracy(cv::Mat& X_test, cv::Mat& y_test)
{
	// Generate predictions
	cv::Mat y_pred;
	generatePredictions(X_test, y_pred);
	// std::cout << "Prediction: " << y_pred.type() << " (" << y_pred.rows << ", " << y_pred.cols << ")" << std::endl;
	// std::cout << "GT: " << y_test.type() << " (" << y_test.rows << ", " << y_test.cols << ")" << std::endl;

	// Convert the predictions from CV_32F to CV_32S
	y_pred.convertTo(y_pred, CV_32S); 
	// std::cout << "Updated Prediction: " << y_pred.type() << " (" << y_pred.rows << ", " << y_pred.cols << ")" << std::endl;

	// Compute difference between y_test and y_pred
	int correctPredictions = cv::countNonZero(y_test == y_pred);
	float accuracy = (float)correctPredictions / y_test.rows;
	// std::cout << "Accuracy: " << accuracy * 100 << endl;
	return accuracy;
}

bool Classifier::saveClassifier()
{
	svm->save(CLASSIFIER_NAME);
	std::cout << "Classifier saved successfully" << std::endl;
	return true;
}

bool Classifier::loadClassifier()
{
	if(Utils::fileExists(CLASSIFIER_NAME))
	{
		svm->load(CLASSIFIER_NAME);
		// svm = cv::ml::SVM::load<cv::ml::SVM>(CLASSIFIER_NAME);
		std::cout << "Classifier loaded successfully" << std::endl;
		return true;
	}
	else
	{
		std::cout << "Saved classifier instance not found" << std::endl;
		return false;
	}
}