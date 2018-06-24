function convLayersUWA(varargin)

setup;
global alexNet;
alexNet = false;

global isDAG;
isDAG = true;

global localRegionSize;
localRegionSize = 3;
localRegionSizePadding = floor(localRegionSize / 2);

USE_LIBLINEAR = false;

% Setup options
if isDAG
    opts.model = 'models/imagenet-resnet-152-dag.mat';
    opts.previousLayerName = 'res4b15';
    opts.nextLayerName = 'res4b20x';
    % opts.previousLayerName = 'res5a';
    % opts.nextLayerName = 'res5cx';
elseif alexNet
    opts.model = 'models/imagenet-caffe-alex.mat';
    opts.layersConv4 = 12;  % Conv-4
    opts.layersConv5 = 14;  % Conv-5
else
    opts.model = 'models/imagenet-vgg-verydeep-19.mat';
    %opts.layersConv4 = 34;  % Conv-5_3
    opts.layersConv4 = 30;  % Conv-5_1
    %opts.layersConv4 = 28;  % Pool_4
    opts.layersConv5 = 36;  % Conv-5_4
end
opts.dataDir = '../data/MFW/';
%opts.dataDir = '../data/MFW-woOther/';
opts.tempDir = '../TempFile_PCA/';
opts.LFDir = '../LocalFeature_PCA/';
opts.calculateCustomAvgImg = false;
opts.numCompPCA = 512;

% Average image opts
opts.avgImg.dataDir = opts.dataDir;
opts.avgImg.imageSize = [224, 224] ;
opts.avgImg.border = [0, 0] ;
opts.avgImg.interpolation = 'bilinear' ;
opts.avgImg.keepAspect = false;

[opts, varargin] = vl_argparse(opts, varargin) ;

create_dirs({opts.tempDir, opts.LFDir})

% Load UWA image database
imdb = fishes_get_database(opts.dataDir);
numImages = length(imdb.images.name);

% Calculate average image
if opts.calculateCustomAvgImg
    averageImage = calculateAverageImage(imdb.images.name(imdb.images.set == 1), opts.avgImg);
end

% Configuration
step1 = 'do';   % extract level 4 local features
step2 = 'do';   % extract level 5 local features
step3 = 'do';	% compress level 5 features
step4 = 'do';	% generate image representations
step5 = 'do';   % classifier training/test
SPM_Config = [1;1];

% Load the model
if isDAG
	net = dagnn.DagNN.loadobj(load(opts.model)) ;
	net.mode = 'test' ;
else
	net = load(opts.model);
	net = vl_simplenn_tidy(net); % makes the model format compatible
end

% Remove the fully connected layer of the network
netFull = net;

if isDAG
    net.vars(net.getVarIndex(opts.previousLayerName)).precious = true;
    net.vars(net.getVarIndex(opts.nextLayerName)).precious = true;
else
    net.layers = net.layers(1 : opts.layersConv4);  % Conv-4
end

if isDAG
    previousLayerFeatures = 1024;
    nextLayerFeatures = 1024;
    % previousLayerFeatures = 2048;
    % nextLayerFeatures = 2048;
elseif alexNet
    previousLayerFeatures = 384;
    nextLayerFeatures = 256;
else
    previousLayerFeatures = 512;
    nextLayerFeatures = 512;
end

if strcmp(step1,'do')
    disp('Starting extraction of features from previous layer');
    L4_dim = localRegionSize * localRegionSize * previousLayerFeatures;
    Mu = zeros(1, L4_dim);
    cnt = 0;
    for i = 1 : numImages
        tic
        % read image
	    %im = imread([dir_image,num2str(i,'%05d'),'.jpg']);
        imageName = fullfile(opts.dataDir, 'images', imdb.images.name{i});
        im = imread(imageName);
        if size(im,3) == 1
            im = repmat(im,[1,1,3]);
        end
        
        if opts.calculateCustomAvgImg
            im = prepare_image(im, averageImage);
        else
            im = prepare_image(im, net.meta.normalization.averageImage);
        end
        
        % Obtain final layer activations from CNN
        if isDAG
			net.eval({'data', im});

            activation = net.vars(net.getVarIndex(opts.previousLayerName)).value;
			activation = squeeze(gather(activation));
		else
	        res = vl_simplenn(net, im);
	        activation = squeeze(gather(res(end).x));
    	end

        [LF_L2, Location] = ExtractLocalFeatures(activation, localRegionSize, false);
        
        Mu = Mu + sum(LF_L2);       % calculate mean vector of L4
        cnt = cnt + size(LF_L2,1);
        save([opts.LFDir, 'LF_L4_', num2str(i,'%05d'), '.mat'], 'LF_L2', 'Location')
        toc
    end
    Mu = Mu/cnt;
    save([opts.tempDir,'Mu.mat'],'Mu')
end

if ~isDAG
    % Remove the fully connected layer of the network
    net = netFull;
    net.layers = net.layers(1 : opts.layersConv5);  % Conv-5
end

if strcmp(step2,'do')
    disp('Starting extraction of features from last layer');
    for i = 1 : numImages
        tic
        % read image
	    %im = imread([dir_image,num2str(i,'%05d'),'.jpg']);
        imageName = fullfile(opts.dataDir, 'images', imdb.images.name{i});
        im = imread(imageName);
        if size(im,3) == 1
            im = repmat(im,[1,1,3]);
        end

        if opts.calculateCustomAvgImg
            im = prepare_image(im, averageImage);
        else
            im = prepare_image(im, net.meta.normalization.averageImage);
        end
        
        if isDAG
			net.eval({'data', im});

			%activation = net.vars(net.getVarIndex('res4b35x')).value;
            activation = net.vars(net.getVarIndex(opts.nextLayerName)).value;
			activation = squeeze(gather(activation));
		else
	        res = vl_simplenn(net, im);
	        activation = squeeze(gather(res(opts.layersConv5).x));
    	end

        [LF_L2, Location] = ExtractLocalFeatures(activation, localRegionSize, true);
        
        save([opts.LFDir, 'LF_L5_', num2str(i,'%05d'), '.mat'], 'LF_L2', 'Location')
        toc
    end

end

% Perform PCA on level 4 features
% PCA projection matrix which will be applied to every level 5 feature afterwards
if strcmp(step3,'do')
    disp('Loading level 4 features for PCA');
    tic
    %numberOfLocalFeatures = 7 * 7;
    %numberOfLocalFeatures = 14 * 14;
    numberOfLocalFeatures = (14 - (2 * localRegionSizePadding)) * (14 - (2 * localRegionSizePadding));
    trainIndices = find(imdb.images.set == 1);
    pcaMatrix = zeros(length(trainIndices) * numberOfLocalFeatures, localRegionSize * localRegionSize * previousLayerFeatures);
    iterator = 1;
    for i = trainIndices
        load([opts.LFDir, 'LF_L4_', num2str(i,'%05d'), '.mat'], 'LF_L2')

        indices = 1 : numberOfLocalFeatures;
        for index = indices
            pcaMatrix(iterator, :) = LF_L2(index, :);
            iterator = iterator + 1;
        end
    end
    toc
    
    % Perform PCA
    disp('Performing PCA on level 4 features');
    tic
    Mpca = pcaCustom(pcaMatrix, opts.numCompPCA);
    save([opts.tempDir, 'Mpca.mat'], 'Mpca')
    toc
    
    % Compress local features using the calculated Mpca
    disp('Compressing features using the calculated PCA projection matrix');
    tic
    pcaMatrix = [];
    for i = 1 : numImages 
        load([opts.LFDir, 'LF_L4_', num2str(i,'%05d'), '.mat'], 'LF_L2', 'Location')
        
        LF_L2_Comp = zeros(numberOfLocalFeatures, opts.numCompPCA);
        for index = 1 : numberOfLocalFeatures
            LF_L2_Comp(index, :) = LF_L2(index, :) * Mpca;
        end
        LF_L2 = LF_L2_Comp;
        
        % Save new local feature vector
        save([opts.LFDir, 'LF_L4_comp_', num2str(i,'%05d'), '.mat'], 'LF_L2', 'Location')
    end
    toc
end

if strcmp(step4,'do')
    disp('Starting cross-layer pooling');
    
    % Initialize number of features in the level 4 layer
    previousLayerFeatures = opts.numCompPCA;

    load([opts.tempDir, 'Mu.mat'], 'Mu')
    %fdim = previousLayerFeatures * (localRegionSize * localRegionSize * nextLayerFeatures)
    fdim =  previousLayerFeatures * nextLayerFeatures;
    Data = zeros(numImages, fdim);
    
    cnt  = 1;
    for i = 1 : numImages
        tic
        load([opts.LFDir,'LF_L4_comp_', num2str(i,'%05d'),'.mat'], 'LF_L2', 'Location')
        LF_L2_4 = LF_L2;
        load([opts.LFDir,'LF_L5_', num2str(i,'%05d'),'.mat'], 'LF_L2')
        LF_L2_5 = LF_L2;
        x_h = Location(:,1); y_h = Location(:,2);
        option.pooling = 'standard';
        option.normalize = 'true';
        
        if isDAG
            spacing = 3;
            option.leftSpacing = spacing;
            option.rightSpacing = spacing;
            option.topSpacing = spacing;
            option.bottomSpacing = spacing;
        else
            option.leftSpacing = 0;
            option.rightSpacing = 0;
            option.topSpacing = 0;
            option.bottomSpacing = 0;
        end

        z = FisherVectorSC_Pooling(SPM_Config, x_h, y_h, LF_L2_4, LF_L2_5, option);
        z = (z - mean(z(:))) / std(z(:));
        z = sqrt(abs(z)) .* sign(z);
        Data(cnt,:) = z / (1e-7 + norm(z));
        cnt = cnt + 1;
        toc
    end
    
    % disp('Saving cross-layer pooling results');
    % tic
    % save([opts.tempDir, 'Data.mat'], 'Data', '-v7.3')
    % toc
end

if strcmp(step5,'do')
    disp('Training SVM');
    % load([opts.tempDir,'Data.mat'], 'Data')
    
    Train_data = Data(imdb.images.set == 1, :);
    Train_label = imdb.images.label(imdb.images.set == 1);

    Val_data = Data(imdb.images.set == 2, :);
    Val_label = imdb.images.label(imdb.images.set == 2);
    
    Test_data = Data(imdb.images.set == 3, :);
    Test_label = imdb.images.label(imdb.images.set == 3);
    
    % compute kernel matrices between every pairs of (train,train) and
    % (test,train) instances and include sample serial number as first column
    numTrain = length(Train_label);
    numTest = length(Test_label);
    numVal = length(Val_label);
    
    if USE_LIBLINEAR
        model = train(Train_label, sparse(Train_data), '-q -c 10');
    else
        % RBF Kernels    
        % radial basis function: exp(-gamma*|u-v|^2)
    %     sigma = 1e-5;
    %     rbfKernel = @(X,Y) exp(-sigma .* pdist2(X,Y,'euclidean').^2);
    % 
    %     trainKernel =  [ (1:numTrain)' , rbfKernel(Train_data, Train_data) ];
    %     valKernel = [ (1:numVal)'  , rbfKernel(Val_data, Train_data)  ];
    %     testKernel = [ (1:numTest)'  , rbfKernel(Test_data, Train_data)  ];

        % Linear Kernels
        trainKernel =  [ (1:numTrain)' , Train_data * Train_data' ];
        valKernel = [ (1:numVal)'  , Val_data * Train_data' ];
        testKernel = [ (1:numTest)'  , Test_data * Train_data' ];

        model = svmtrain(Train_label', trainKernel, '-q -c 10 -t 4');
    end

    disp('Saving SVM Model');
    tic
    save([opts.tempDir, 'SVM.mat'], 'model', '-v7.3');
    toc

    disp('Train data results:');
    if USE_LIBLINEAR
        [predicted_label_train, accuracyTrain, decValsTrain] = predict(Train_label, sparse(Train_data), model);
    else
        [predicted_label_train, accuracyTrain, decValsTrain] = svmpredict(Train_label', trainKernel, model);
    end
    C = confusionmat(Train_label, predicted_label_train)    % Confusion matrix

    disp('Validation data results:');
    if USE_LIBLINEAR
        [predicted_label_val, accuracyVal, decValsVal] = predict(Val_label, sparse(Val_data), model);
    else
        [predicted_label_val, accuracyVal, decValsVal] = svmpredict(Val_label', valKernel, model);
    end
    C = confusionmat(Val_label, predicted_label_val)    % Confusion matrix

    disp('Test data results:');
    if USE_LIBLINEAR
        [predicted_label_test, accuracyTest, decValsTest] = predict(Test_label, sparse(Test_data), model);
    else
        [predicted_label_test, accuracyTest, decValsTest] = svmpredict(Test_label', testKernel, model);
    end
    C = confusionmat(Test_label, predicted_label_test)   % Confusion matrix

    disp('Saving Test Set Results');
    tic
    testFileNames = imdb.images.name(imdb.images.set == 3);
    save([opts.tempDir, 'testSetStats.mat'], 'Test_label', 'predicted_label_test', 'testFileNames');
    toc
end

% prepare oversampled input

%% ========================== Subfunctions ================================
function create_dirs(dir_names)
for i = 1:length(dir_names)
    if ~isdir(dir_names{i})
        mkdir(dir_names{i})
    end
end


function im = prepare_image(im, averageImage)
global alexNet;
if alexNet
    IMAGE_DIM = 227;
else
    IMAGE_DIM = 224;
end

% resize to fixed input size
im = single(im);
im = imresize(im, [IMAGE_DIM IMAGE_DIM], 'bilinear');
% permute from RGB to BGR (IMAGE_MEAN is already BGR)
%im = im(:,:,[3 2 1]) - IMAGE_MEAN;
averageImage = mean(mean(im));

im = bsxfun(@minus, im, averageImage);
%im = permute(im,[2,1,3]);
%im = imresize(im, [CROPPED_DIM CROPPED_DIM], 'bilinear');
%cnt = 1;

% Convert image to YCbCr
%im = rgb2ycbcr(double(im));
% histEqY = histeq(im(:, :, 1));
% im(:, :, 1) = histEqY;

function [LF_L2,Location] = ExtractLocalFeatures(f, localRegionSize, isNextLayer)
regionSizePadding = floor(localRegionSize / 2);

if isNextLayer
    localRegionDim = 1;
else
    localRegionDim = localRegionSize * localRegionSize;
end
LF_L2 = zeros((size(f, 1) - (2 * regionSizePadding)) * (size(f, 2) - (2 * regionSizePadding)), localRegionDim * size(f, 3));
cnt = 1;
Location = zeros((size(f, 1) - (2 * regionSizePadding)) * (size(f, 2) - (2 * regionSizePadding)), 2);
for x = (1 + regionSizePadding) : (size(f, 1) - regionSizePadding)
    for y = (1 + regionSizePadding) : (size(f, 2) - regionSizePadding)
        
        regionCounter = 1;
        if isNextLayer
            currentRegion = zeros(1, size(f, 3));
            currentRegion(regionCounter, :) = f(x, y, :);
        else
            currentRegion = zeros(localRegionSize * localRegionSize, size(f, 3));
            for i = -regionSizePadding : regionSizePadding
                for j = -regionSizePadding : regionSizePadding
                    currentRegion(regionCounter, :) = f(x + i, y + j, :);
                    regionCounter = regionCounter + 1;
                end
            end
        end

        % Linearize the current region matrix
        currentRegion = currentRegion';
        currentRegion = currentRegion(:);
        
        LF_L2(cnt, :) = currentRegion;
        Location(cnt, :) = [x, y];
        cnt = cnt + 1;
    end
end
