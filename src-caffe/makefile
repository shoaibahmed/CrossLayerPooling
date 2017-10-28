all:
	g++ uwaTest.cpp CrossLayerPoolingClassifier.cpp FeatureExtractor.cpp Classifier.cpp -fopenmp -lcaffe -lglog -lboost_system -lopencv_core -lopencv_imgcodecs -lopencv_videoio -lopencv_video -lopencv_highgui -lopencv_imgproc -lopencv_ml -lm -I /opt/caffe/include -I /usr/include/atlas -L /usr/lib64/atlas -L /opt/caffe/lib -I /usr/local/cuda/include  -I ../tiny-dnn/ -L /usr/local/cuda/lib64 -L /usr/local/lib -std=c++11 -O3 -o clp

test:
	g++ testCLP.cpp CrossLayerPoolingClassifier.cpp FeatureExtractor.cpp Classifier.cpp -fopenmp -lcaffe -lglog -lboost_system -lopencv_core -lopencv_imgcodecs -lopencv_videoio -lopencv_video -lopencv_highgui -lopencv_imgproc -lopencv_ml -lm -I /opt/caffe/include -I /usr/include/atlas -L /usr/lib64/atlas -L /opt/caffe/lib -I /usr/local/cuda/include -I ../tiny-dnn/ -L /usr/local/cuda/lib64 -L /usr/local/lib -std=c++11 -O3 -o testCLP

clean:
	rm clp
