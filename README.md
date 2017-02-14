<h1>Cross-layer pooling algorithm</h1> <br/>
This repository provides a basic implementation of Cross-layer pooling algorithm (Liu et al. 2015) in C++ using OpenCV and Caffe. The code uses pretrained ResNet-152 network for its initialization. Refer to the paper for more details on ResNet (He et al. 2015) <b>[https://arxiv.org/abs/1512.03385]</b> and <b>[https://arxiv.org/abs/1411.7466]</b> for details on the cross-pooling method. For training the network on your own dataset, <b>CrossLayerPoolingClassifier</b> class to train a linear SVM on top of features computed through cross-layer pooling strategy or use a pre-trained SVM for predictions.

<br/><b>Code Dependencies: </b>
<ol>
<li>OpenCV-3.2</li>
<li>Caffe</li>
<li>Boost</li>
<li>Pretrained ResNet-152-model.caffemodel (https://github.com/KaimingHe/deep-residual-networks)</li>
</ol>

<br/><b>TODOs: </b>
<ol>
<li>Add support for larger region sizes</li>
<li>Add optimization</li>
<li>Add code profiling</li>
</ol>

<br/><br/> Email: <b>12bscsssiddiqui@seecs.edu.pk</b>