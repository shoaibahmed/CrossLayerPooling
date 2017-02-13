#include "CrossLayerPoolingClassifier.h"

CrossLayerPoolingClassifier::CrossLayerPoolingClassifier()
{
	// Initialize classifier
	classifier = new Classifier();
	feature_extractor = new FeatureExtractor(MODEL_FILE, TRAINED_FILE);

	// Load classifier if already trained model is present
	classifier_loaded = classifier->loadClassifier();

#if PERFORM_PCA
	// Initialize PCA object
	pca = new cv::PCA();

	// Load PCA data if already saved
	pca_data_loaded = loadPCAData();
#endif
}

void CrossLayerPoolingClassifier::trainClassifier(std::string data_dir)
{
	// Read files from data directory
	std::string images_file_name = data_dir + "images.txt";
	std::string labels_file_name = data_dir + "image_class_labels.txt";
	std::string train_test_split_file_name = data_dir + "train_test_split.txt";
	std::string classes_file_name = data_dir + "classes.txt";

	// Declare vectors to store data
	std::vector<std::string> image_files;
	std::vector<int> image_labels;
	std::vector<int> train_test_split;
	std::vector<std::string> classes(NUM_CLASSES);
	// std::vector<std::string> classes;
	
	int num_training_examples = 0;
	int num_validation_examples = 0;
	int num_test_examples = 0;

	// Load image file names
	if(Utils::fileExists(images_file_name))
	{
		std::ifstream images_file(images_file_name);
		std::string line, temp;
		
		while(std::getline(images_file, line))
		{
			// Ignore line number
			std::istringstream iss(line);
			iss >> temp; iss >> temp; // Discard first word
			
			#if DEBUG > 2
				std::cout << "File: " << temp << std::endl; 
			#endif
			image_files.push_back(data_dir + "images/" + temp);
		}

		images_file.close();
	}
	else
	{
		std::cout << "Error: Images file not found (" << images_file_name << ")" << std::endl; 
		return;
	}

	// Load image labels
	if(Utils::fileExists(labels_file_name))
	{
		std::ifstream labels_file(labels_file_name);
		std::string line, temp;

		while(std::getline(labels_file, line))
		{
			// Ignore line number
			std::istringstream iss(line);
			iss >> temp; iss >> temp; // Discard first word
			
			#if DEBUG > 2
				std::cout << "Label: " << temp << std::endl; 
			#endif 
			image_labels.push_back(stoi(temp));
		}

		labels_file.close();
	}
	else
	{
		std::cout << "Error: Labels file not found (" << labels_file_name << ")" << std::endl; 
		return;
	}

	// Load train test split
	if(Utils::fileExists(train_test_split_file_name))
	{
		std::ifstream train_test_split_file(train_test_split_file_name);
		std::string line, temp;

		while(std::getline(train_test_split_file, line))
		{
			// Ignore line number
			std::istringstream iss(line);
			iss >> temp; iss >> temp; // Discard first word
			
			#if DEBUG > 2
				std::cout << "Split: " << temp << std::endl; 
			#endif 

			int set_assignment = stoi(temp);
			train_test_split.push_back(set_assignment);

			if (set_assignment == TRAIN_EXAMPLE)
				num_training_examples++;
			else if (set_assignment == VALIDATION_EXAMPLE)
				num_validation_examples++;
			else if (set_assignment == TEST_EXAMPLE)
				num_test_examples++;
			else
			{
				std::cout << "Error: Example doesn't belong to train, test or validation set" << std::endl; 
				return;
			}
		}

		train_test_split_file.close();
	}
	else
	{
		std::cout << "Error: Train test split file not found (" << train_test_split_file_name << ")" << std::endl; 
		return;
	}

	// Load label class names
	if(Utils::fileExists(classes_file_name))
	{
		std::ifstream classes_file(classes_file_name);
		std::string line, temp;

		while(std::getline(classes_file, line))
		{
			// Ignore line number
			std::istringstream iss(line);
			iss >> temp; 
			int class_index = stoi(temp) - 1;
			iss >> temp;
			
			#if DEBUG > 2
				std::cout << "Class name: " << temp << std::endl; 
			#endif 
			classes[class_index] = temp;
			// classes.insert(classes.begin() + class_index, temp);
		}

		classes_file.close();
	}
	else
	{
		std::cout << "Error: Class names file not found (" << classes_file_name << ")" << std::endl; 
		return;
	}

	// Print data stats
	std::cout << "Data loaded successfully" << std::endl; 
	std::cout << "==================================" << std::endl; 
	std::cout << "Training examples: " << num_training_examples << std::endl; 
	std::cout << "Validation examples: " << num_validation_examples << std::endl; 
	std::cout << "Test examples: " << num_test_examples << std::endl; 
	std::cout << "==================================" << std::endl; 

	std::cout << "Classes: " << std::endl;
	for(int i = 0; i < classes.size(); i++)
		std::cout << classes[i] << std::endl;
	std::cout << "==================================" << std::endl; 

	// Load feature extractor for each of the images
// #if PERFORM_PCA
// 	int dim = feature_extractor->getCrossPooledFeaturesSize(NUM_COMP_PCA);
// #else
// 	int dim = feature_extractor->getCrossPooledFeaturesSize(0);
// #endif
	int dim = feature_extractor->getCrossPooledFeaturesSize(0);

	// cv::Mat data(cv::Size(num_training_examples + num_validation_examples + num_test_examples, dim), CV_32F);
	cv::Mat data_train(num_training_examples, dim, CV_32F);
	cv::Mat label_train(num_training_examples, 1, CV_32S);
	cv::Mat data_val(num_validation_examples, dim, CV_32F);
	cv::Mat label_val(num_validation_examples, 1, CV_32S);
	cv::Mat data_test(num_test_examples, dim, CV_32F);
	cv::Mat label_test(num_test_examples, 1, CV_32S);

	int index_train = 0, index_val = 0, index_test = 0;

	for (int i = 0; i < image_files.size(); i++)
	{
		std::string file_name = image_files[i];
		std::cout << "Processing file: " << file_name << std::endl;

		// Read image
		cv::Mat img = cv::imread(file_name);

		if(img.empty())
		{
			std::cout << "Error: Unable to read file (" << file_name << ")" << std::endl;
			return;
		}

		#if DEBUG
			cv::imshow("Image", img);
			char c = cv::waitKey(100);
			if (c == 'q')
				break;
		#endif

		// Extract features
		cv::Mat feature_vector;
		feature_extractor->extractFeatures(img, feature_vector);

		cv::Mat data_row;
		if (train_test_split[i] == TRAIN_EXAMPLE)
		{
			data_row = data_train.row(index_train);
			label_train.at<int>(index_train) = image_labels[i];
			index_train++;
		}
		else if (train_test_split[i] == VALIDATION_EXAMPLE)
		{
			data_row = data_val.row(index_val);
			label_val.at<int>(index_val) = image_labels[i];
			index_val++;
		}
		else
		{
			data_row = data_test.row(index_test);
			label_test.at<int>(index_test) = image_labels[i];
			index_test++;
		}
		
		// Copy extracted features to the feature mat
		feature_vector.copyTo(data_row);
	}

	// Write extracted features to file
	// cv::FileStorage file("features.txt", cv::FileStorage::WRITE);
	// file << "f" << data;
	// std::cout << "Features written to file" << std::endl;

#if PERFORM_PCA
	if(!pca_data_loaded)
	{
		std::cout << "Computing PCA matrix" << std::endl;

		// Compute principle components
		// pca = pca(data_train, // pass the data
		// 	cv::Mat(), // we do not have a pre-computed mean vector,
		// 	// so let the PCA engine to compute it
		// 	cv::PCA::DATA_AS_ROW, // indicate that the vectors
		// 	// are stored as matrix rows
		// 	// (use PCA::DATA_AS_COL if the vectors are
		// 	// the matrix columns)
		// 	NUM_COMP_PCA // specify, how many principal components to retain
		// );

		PCA pca(data_train, Mat(), cv::PCA::DATA_AS_ROW, NUM_COMP_PCA);

		// Write PCA matrix to file
		savePCAData();
	}
	
	// Compress data using computed PCA matrix
	std::cout << "Training data: (" << data_train.rows << ", " << data_train.cols << ") => (";
	pca->project(data_train, data_train);
	std::cout << data_train.rows << ", " << data_train.cols << ")" << std::endl;

	std::cout << "Validation data: (" << data_val.rows << ", " << data_val.cols << ") => (";
	pca->project(data_val, data_val);
	std::cout << data_val.rows << ", " << data_val.cols << ")" << std::endl;

	std::cout << "Test data: (" << data_test.rows << ", " << data_test.cols << ") => (";
	pca->project(data_test, data_test);
	std::cout << data_test.rows << ", " << data_test.cols << ")" << std::endl;
#endif

	// Train classifier only if no previously saved model found
	if(!classifier_loaded)
	{
		// Train classifier on top of the these features
		// std::cout << "Creating training data" << std::endl;
		// cv::Ptr<cv::ml::TrainData> train_data_ = cv::ml::TrainData::create(data_train, cv::ml::ROW_SAMPLE, label_train);
		
		std::cout << "Training classifier" << std::endl;
		// classifier->trainClassifier(train_data_);
		classifier->trainClassifier(data_train, label_train);
	}

	// Test accuracy on validation and test set
	std::cout << "Computing classifier's performance" << std::endl;

	float train_acc = classifier->computeAccuracy(data_train, label_train);
	std::cout << "Train set accuracy: " << train_acc * 100 << "\%" << std::endl;

	float val_acc = classifier->computeAccuracy(data_val, label_val);
	std::cout << "Validation set accuracy: " << val_acc * 100 << "\%" << std::endl;

	float test_acc = classifier->computeAccuracy(data_test, label_test);
	std::cout << "Test set accuracy: " << test_acc * 100 << "\%" << std::endl;

	std::cout << "Classifier training completed" << std::endl;
}

int CrossLayerPoolingClassifier::generatePrediction(cv::Mat& img)
{
	// Extract features
	cv::Mat feature_vector, predictions;
	feature_extractor->extractFeatures(img, feature_vector);
#if PERFORM_PCA
	pca->project(feature_vector, feature_vector);
#endif
	classifier->generatePredictions(feature_vector, predictions);
	return predictions.at<int>(0);
}


/********************** Private methods **********************/
void CrossLayerPoolingClassifier::savePCAData()
{
	cv::FileStorage fs(PCA_FILE_NAME, cv::FileStorage::WRITE);
	fs << "mean" << pca->mean;
	fs << "e_vectors" << pca->eigenvectors;
	fs << "e_values" << pca->eigenvalues;
	fs.release();

	std::cout << "PCA data saved successfully" << std::endl;
}

bool CrossLayerPoolingClassifier::loadPCAData()
{
	if(Utils::fileExists(PCA_FILE_NAME))
	{
		cv::FileStorage fs(PCA_FILE_NAME, cv::FileStorage::READ);
		fs["mean"] >> pca->mean ;
		fs["e_vectors"] >> pca->eigenvectors ;
		fs["e_values"] >> pca->eigenvalues ;
		fs.release();

		std::cout << "PCA data loaded successfully" << std::endl;
		return true;
	}
	else
	{
		std::cout << "PCA data instance not found" << std::endl;
		return false;
	}
}