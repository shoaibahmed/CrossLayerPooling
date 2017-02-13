#include "FeatureExtractor.h"

FeatureExtractor::FeatureExtractor(const string& model_file, const string& trained_file) 
{
#if USE_GPU
	Caffe::set_mode(Caffe::GPU);
#else
	Caffe::set_mode(Caffe::CPU);
#endif

	/* Load the network. */
	net_.reset(new Net<float>(model_file, TEST));
	net_->CopyTrainedLayersFrom(trained_file);

	CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
	CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

	Blob<float>* input_layer = net_->input_blobs()[0];
	num_channels_ = input_layer->channels();
	CHECK(num_channels_ == 3 || num_channels_ == 1)
	<< "Input layer should have 1 or 3 channels.";
	input_geometry_ = cv::Size(input_layer->width(), input_layer->height());
	// Blob<float>* output_layer = net_->output_blobs()[0];

	CHECK(net_->has_blob(LOWER_BLOB_NAME)) << "Unknown feature blob name " << LOWER_BLOB_NAME << " in the network" << std::endl;
	CHECK(net_->has_blob(LOWER_BLOB_NAME)) << "Unknown feature blob name " << HIGHER_BLOB_NAME << " in the network" << std::endl;

	region_size_padding = REGION_SIZE / 2; // To iterate over the region
	std::cout << "Region size padding: " << region_size_padding << std::endl;
}

void FeatureExtractor::extractFeatures(const cv::Mat& img, cv::Mat& cross_pooled_features) 
{
	Blob<float>* input_layer = net_->input_blobs()[0];
	input_layer->Reshape(1, num_channels_, input_geometry_.height, input_geometry_.width);
	/* Forward dimension change to all layers. */
	net_->Reshape();

	std::vector<cv::Mat> input_channels;
	wrapInputLayer(&input_channels);

	// std::cout << "Pre-processing image" << std::endl;
	preprocess(img, &input_channels);

	net_->Forward();
	// cudaDeviceSynchronize();

	// Extract features
	// Blob<float>* output_layer = net_->output_blobs()[0];
	boost::shared_ptr<Blob<float> > lower_blob = net_->blob_by_name(LOWER_BLOB_NAME);
	boost::shared_ptr<Blob<float> > higher_blob = net_->blob_by_name(HIGHER_BLOB_NAME);
	const float* begin_lower = lower_blob->cpu_data();
	const float* begin_higher = higher_blob->cpu_data();

	// Index(n, k, h, w) -> ((n * K + k) * H + h) * W + w
	int dim = lower_blob->channels() * higher_blob->channels() * REGION_SIZE * REGION_SIZE;
	// std::cout << "Feature dim: " << dim << std::endl;
	cross_pooled_features = Mat::zeros(1, dim, CV_32FC1);
	float* cross_pooled_features_ptr = cross_pooled_features.ptr<float>(0); // Since there is only a single row
	
	// Iterate over the feature volume
	for (int n = 0; n < NUM_IMAGES; n++)
	{
		// L2 is the higher layer
		for (int kL2 = 0; kL2 < HIGHER_BLOB_CHANNELS; kL2++)
		{
			// L1 is the lower layer
			for (int kL1 = 0; kL1 < LOWER_BLOB_CHANNELS; kL1++)
			{
				for (int h = FEATURE_SPACING; h < BLOB_SIZE - FEATURE_SPACING; h++)
				{

#if REGION_SIZE == 1
					int higher_layer_index = ((n * HIGHER_BLOB_CHANNELS + kL2) * BLOB_SIZE + h) * BLOB_SIZE + FEATURE_SPACING;
					int index = ((n * LOWER_BLOB_CHANNELS + kL1) * BLOB_SIZE + h) * BLOB_SIZE + FEATURE_SPACING;
					// int index = ((n * BLOB_CHANNELS + k) * BLOB_SIZE + h) * BLOB_SIZE + w;

					for (int w = FEATURE_SPACING; w < BLOB_SIZE - FEATURE_SPACING; w++)
					{
						// *cross_pooled_features_ptr++ += *(begin_lower + index) * *(begin_higher + higher_layer_index);
						*cross_pooled_features_ptr += *(begin_lower + index) * *(begin_higher + higher_layer_index);
						index++;
						higher_layer_index++;
					}

#else
					for (int w = FEATURE_SPACING; w < BLOB_SIZE - FEATURE_SPACING; w++)
					{
						float* temp_loc = cross_pooled_features_ptr;
						int higher_layer_index = ((n * HIGHER_BLOB_CHANNELS + kL2) * BLOB_SIZE + h) * BLOB_SIZE + FEATURE_SPACING + w;

						for (int r_h = -region_size_padding; r_h <= region_size_padding; r_h++)
						{
							for (int r_w = -region_size_padding; r_w <= region_size_padding; r_w++)
							{
								int index = ((n * LOWER_BLOB_CHANNELS + kL1) * BLOB_SIZE + (h - r_h)) * BLOB_SIZE + FEATURE_SPACING + (w - r_w);
								// *cross_pooled_features_ptr++ += *(begin_lower + index) * *(begin_higher + higher_layer_index);
								*cross_pooled_features_ptr++ += *(begin_lower + index) * *(begin_higher + higher_layer_index);
								// index++;
								// higher_layer_index++;
								// cross_pooled_features_ptr++;
							}
						}
						cross_pooled_features_ptr = temp_loc;
					}
#endif

				}
				cross_pooled_features_ptr++; // Move to next element
			}
		}
	}

	// Perform normalizations
#if PERFORM_FEATURE_NORMALIZATION
	cv::Mat mean, stdDev;
	// Per element norm
	for(int i = 0; i < HIGHER_BLOB_CHANNELS; i++)
	{
		cv::Mat temp(LOWER_BLOB_CHANNELS, 1, CV_32F);
		for (int j = 0; j < LOWER_BLOB_CHANNELS; j++)
		{
			temp.at<float>(j) = cross_pooled_features.at<float>((i * LOWER_BLOB_CHANNELS) + j);
		}

		cv::meanStdDev(temp, mean, stdDev);
		temp = temp / (1e-7 + norm(temp));
		for (int j = 0; j < LOWER_BLOB_CHANNELS; j++)
		{
			cross_pooled_features.at<float>((i * LOWER_BLOB_CHANNELS) + j) = temp.at<float>(j);
		}
	}

	// 1. Conversion to z-score
	cv::meanStdDev(cross_pooled_features, mean, stdDev);
	// std::cout << "Mean: " << mean << " | Std. Dev.: " << stdDev << std::endl;
	// std::cout << "Mean: " << mean.at<double>(0) << " | Std. Dev.: " << stdDev.at<double>(0) << std::endl;

	cross_pooled_features = (cross_pooled_features - mean.at<double>(0)) / stdDev.at<double>(0);
	// cv::meanStdDev(cross_pooled_features, mean, stdDev);
	// std::cout << "Mean: " << mean << " | Std. Dev.: " << stdDev << std::endl;
	// std::cout << "Mean: " << mean.at<double>(0) << " | Std. Dev.: " << stdDev.at<double>(0) << std::endl;

	// 2. Conversion to unit vector
	cv::Mat temp = Utils::signum(cross_pooled_features);
	cv::sqrt(cv::abs(cross_pooled_features), cross_pooled_features);
	cross_pooled_features = cross_pooled_features.mul(temp);
	// cv::sqrt(cross_pooled_features, cross_pooled_features);

	cross_pooled_features = cross_pooled_features / (1e-7 + norm(cross_pooled_features));
#endif
}

int FeatureExtractor::getCrossPooledFeaturesSize(int num_comp_pca)
{
	if (num_comp_pca > 0)
		return num_comp_pca * HIGHER_BLOB_CHANNELS;
	else
		return REGION_SIZE * REGION_SIZE * LOWER_BLOB_CHANNELS * HIGHER_BLOB_CHANNELS;
} 

/* Wrap the input layer of the network in separate cv::Mat objects
* (one per channel). This way we save one memcpy operation and we
* don't need to rely on cudaMemcpy2D. The last preprocessing
* operation will write the separate channels directly to the input
* layer. */
void FeatureExtractor::wrapInputLayer(std::vector<cv::Mat>* input_channels) 
{
	Blob<float>* input_layer = net_->input_blobs()[0];

	int width = input_layer->width();
	int height = input_layer->height();
	float* input_data = input_layer->mutable_cpu_data();
	for (int i = 0; i < input_layer->channels(); ++i) 
	{
		cv::Mat channel(height, width, CV_32FC1, input_data);
		input_channels->push_back(channel);
		input_data += width * height;
	}
}

void FeatureExtractor::preprocess(const cv::Mat& img, std::vector<cv::Mat>* input_channels) 
{
	cv::Mat sample_resized;
	if (img.size() != input_geometry_)
		cv::resize(img, sample_resized, input_geometry_);
	else
		sample_resized = img;
	//cvtColor(sample_resized, sample_resized, CV_BGR2RGB);

	cv::Mat sample_float;
	sample_resized.convertTo(sample_float, CV_32FC3);

	/* This operation will write the separate BGR planes directly to the
	* input layer of the network because it is wrapped by the cv::Mat
	* objects in input_channels. */
	cv::split(sample_float, *input_channels);

	// cv::subtract(input_channels->at(0), 104.00699, input_channels->at(0));
	// cv::subtract(input_channels->at(1), 116.66877, input_channels->at(1));
	// cv::subtract(input_channels->at(2), 122.67892, input_channels->at(2));

	// Per channel mean subtraction
	cv::Scalar imgMean = cv::mean(img);
	// std::cout << "Image mean: " << imgMean << std::endl;

	input_channels->at(0) -= imgMean[0];
	input_channels->at(1) -= imgMean[1];
	input_channels->at(2) -= imgMean[2];

	// input_channels->at(0) /= varB;
	// input_channels->at(1) /= varG;
	// input_channels->at(2) /= varR;

	// CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
	//       == net_->input_blobs()[0]->cpu_data())
	//   << "Input channels are not wrapping the input layer of the network.";
}
