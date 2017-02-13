all:
	g++ uwaTest.cpp CrossLayerPoolingClassifier.cpp FeatureExtractor.cpp Classifier.cpp -lcaffe -lglog -lboost_system -lopencv_core -lopencv_imgcodecs -lopencv_videoio -lopencv_video -lopencv_highgui -lopencv_imgproc -lopencv_ml -lm -I /home/cvmr/packages/caffe/include -L /home/cvmr/packages/caffe/build/lib -I /usr/local/cuda/include -L /usr/local/cuda/lib64 -L /usr/local/lib -std=c++11 -O3 -o clp

test:
	g++ testCLP.cpp CrossLayerPoolingClassifier.cpp FeatureExtractor.cpp Classifier.cpp -lcaffe -lglog -lboost_system -lopencv_core -lopencv_imgcodecs -lopencv_videoio -lopencv_video -lopencv_highgui -lopencv_imgproc -lopencv_ml -lm -I /home/cvmr/packages/caffe/include -L /home/cvmr/packages/caffe/build/lib -I /usr/local/cuda/include -L /usr/local/cuda/lib64 -L /usr/local/lib -std=c++11 -O3 -o testCLP

clean:
	rm clp
