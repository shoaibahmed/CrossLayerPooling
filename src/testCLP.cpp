#include "FeatureExtractor.h"

int main(int argc, char** argv) 
{
  // if (argc != 3) 
  // {
  //   std::cerr << "Usage: " << argv[0]
  //   << " deploy.prototxt network.caffemodel" << std::endl;
  //   return 1;
  // }

  ::google::InitGoogleLogging(argv[0]);

  // std::string model_file   = argv[1];
  // std::string trained_file = argv[2];
  std::string model_file   = "./model/ResNet-152-deploy.prototxt";
  std::string trained_file = "./model/ResNet-152-model.caffemodel";
  std::string inputImageName = "./extras/test%02d.png";

  FeatureExtractor featureExtractor(model_file, trained_file);

  VideoCapture imageReader;
  imageReader = VideoCapture(inputImageName);

  Mat inputImage;

  std::cout << "Capturing frames" << std::endl;
  // Obtain cross-pooled features
  while(1)
  {
    imageReader >> inputImage;

    // Check for invalid input
    if(! inputImage.data )
    {
      std::cout <<  "Processing completed" << std::endl;
      break;
    }

    imshow("Input Image", inputImage);

    // Pre-processing step built into the feature extractor
    // resize(inputImage, inputImage, Size(224, 224));

    cv::Mat crossPooledFeatures;
    featureExtractor.extractFeatures(inputImage, crossPooledFeatures);
    std::cout << "Size of features: " << crossPooledFeatures.type() << " (" << crossPooledFeatures.rows << ", " << crossPooledFeatures.cols << ")" << std::endl;

    // Write features to file
    cv::FileStorage file("features.txt", cv::FileStorage::WRITE);
    file << "f" << crossPooledFeatures;

    // std::chrono::steady_clock::time_point beginTime = std::chrono::steady_clock::now();
    // fcn.getSegmentationMask(inputImage, outputImage);
    // std::chrono::steady_clock::time_point endTime = std::chrono::steady_clock::now();
    // std::cout << "Graph execution time: " << (std::chrono::duration_cast<std::chrono::milliseconds>(endTime - beginTime).count()) << " ms" << std::endl;

    char c = waitKey(-1);
    if (c == 'q')
    {
      std::cout << "Execution terminated" << std::endl;
      break;
    }
  }

  return 0;
}
