#include "CrossLayerPoolingClassifier.h"

int main(int argc, char** argv) 
{
	if (argc < 2) 
	{
		std::cerr << "Usage: " << argv[0]
		<< " <Data directory>" << std::endl;
		return 1;
	}

	::google::InitGoogleLogging(argv[0]);

	std::string data_dir   = argv[1];
	std::string images_dir;
	if (argc > 2)
		images_dir = argv[2];
	else
		images_dir = "images/";
	
	// Create classifier object
	CrossLayerPoolingClassifier clpCl;
	clpCl.trainClassifier(data_dir, images_dir);

	return 0;
}
