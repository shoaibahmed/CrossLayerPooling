#include "Classifier.h"

Classifier::Classifier(int featureSize, int numClasses) : net(), optimizer()
{
	#if USE_SVM
		svm = cv::ml::SVM::create();

		// Specify SVM params
		svm->setType(cv::ml::SVM::C_SVC);
		svm->setKernel(cv::ml::SVM::LINEAR);
		svm->setC(10);
		svm->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER, 100, 1e-6));
	#else
		// net = tiny_dnn::network<tiny_dnn::sequential>();
		net << tiny_dnn::layers::fc(featureSize, HIDDEN_LAYER_NEURONS) << tiny_dnn::activation::tanh() << 
			   tiny_dnn::layers::fc(HIDDEN_LAYER_NEURONS, numClasses) << tiny_dnn::activation::softmax();
		// optimizer = new tiny_cnn::adam();
		// net = tiny_dnn::make_mlp<tiny_dnn::cross_entropy_multiclass, tiny_dnn::adam, tiny_dnn::activation::tanh>({featureSize, HIDDEN_LAYER_NEURONS, numClasses});
		// net = tiny_dnn::make_mlp<tiny_dnn::activation::softmax>({featureSize, HIDDEN_LAYER_NEURONS, numClasses});
	#endif
}

// void Classifier::trainClassifier(cv::Mat& X, cv::Mat& y)
// void Classifier::trainClassifier(cv::Ptr<cv::ml::TrainData> train_data)
// {
// 	// Train classifier
// 	// Each training sample occupies a column of samples
// 	svm->train(train_data);

// 	// Save trained classifier
// 	saveClassifier();
// }

void Classifier::trainSVM(cv::Mat& X, cv::Mat& y)
{
	// Train classifier
	// Each training sample occupies a column of samples
	svm->train(X, cv::ml::ROW_SAMPLE, y);

	// Save trained classifier
	saveClassifier();
}

void Classifier::trainMLP(const std::vector<tiny_dnn::vec_t>& data, const std::vector<tiny_dnn::label_t>& labels)
{
	// net.fit<tiny_dnn::cross_entropy_multiclass>(opt, in, t, batch_size, epoch);
	// net->train(data, labels); // T == label_t
	net.train<tiny_dnn::cross_entropy>(optimizer, data, labels, 1, 1); // T == label_t
}

void Classifier::generatePredictionsSVM(cv::Mat& X, cv::Mat& y_pred)
{
	svm->predict(X, y_pred);
}

void Classifier::generatePredictionsMLP(const tiny_dnn::vec_t& data, tiny_dnn::label_t& label)
{
	label = net.predict_label(data);
}

float Classifier::computeAccuracy(cv::Mat& X_test, cv::Mat& y_test)
{
	// Generate predictions
	cv::Mat y_pred;
	generatePredictionsSVM(X_test, y_pred);
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
	#if USE_SVM
		svm->save(CLASSIFIER_NAME);
	#else
		net.save(MLP_CLASSIFIER_NAME);
	#endif
	std::cout << "Classifier saved successfully" << std::endl;
	return true;
}

bool Classifier::loadClassifier()
{
	if(Utils::fileExists(CLASSIFIER_NAME))
	{
		#if USE_SVM
			svm->load<cv::ml::SVM>(CLASSIFIER_NAME);	
		#else
			net.load(MLP_CLASSIFIER_NAME);
		#endif
		std::cout << "Classifier loaded successfully" << std::endl;
		return true;
	}
	else
	{
		std::cout << "Saved classifier instance not found" << std::endl;
		return false;
	}
}
