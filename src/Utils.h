#ifndef UTILS_H_
#define UTILS_H_

#include "stdafx.h"

class Utils
{
public:
	static cv::Mat signum(cv::Mat& src)
	{
		cv::Mat z = cv::Mat::zeros(src.size(), src.type()); 
		cv::Mat a = (z < src) & 1;
		cv::Mat b = (src < z) & 1;

		cv::Mat dst;
		cv::addWeighted(a, 1.0, b, -1.0, 0.0, dst, CV_32F);
		return dst;
	}

	static bool fileExists(const std::string& file_name)
	{
		std::ifstream file(file_name.c_str());
		return file.good();
	}


	static void convertMatToVector(cv::Mat& matFeatures, tiny_dnn::vec_t& vecFeatures)
	{
		// Pointer to the 0-th row
		const double* p = matFeatures.ptr<double>(0);

		// Copy data to a vector.  Note that (p + matFeatures.cols) points to the
		// end of the row.
		vecFeatures = tiny_dnn::vec_t(p, p + matFeatures.cols);
		// if (matFeatures.isContinuous()) 
		// {
		// 	vecFeatures.assign(matFeatures.datastart, matFeatures.dataend);
		// } 
		// else 
		// {
		// 	for (int i = 0; i < matFeatures.rows; ++i) 
		// 	{
		// 		vecFeatures.insert(vecFeatures.end(), matFeatures.ptr<float>(i), matFeatures.ptr<float>(i)+matFeatures.cols);
		// 	}
		// }
	}

};

#endif