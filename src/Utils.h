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

};

#endif