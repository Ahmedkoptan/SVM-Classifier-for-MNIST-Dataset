clear all
close all
clc

%accessing all the necessary files in order to begin reading them
trainImgs = fopen('train-images-idx3-ubyte','r','b');
testImgs = fopen('t10k-images-idx3-ubyte','r','b');
trainLabels = fopen('train-labels-idx1-ubyte','r','b');
testLabels = fopen('t10k-labels-idx1-ubyte','r','b');

%preparing the metadata for the training images set to be read
trainmagicnum = fread(trainImgs,1,'int32');
trainCount = fread(trainImgs,1,'int32');
trainW = fread(trainImgs,1,'int32');
trainH = fread(trainImgs,1,'int32');

%preparing the metadata for the test images set to be read
testmagicnum = fread(testImgs,1,'int32');
testCount = fread(testImgs,1,'int32');
testW = fread(testImgs,1,'int32');
testH = fread(testImgs,1,'int32');

%preparing the metadata for the training label set to be read
trainlabelmagicnum = fread(trainLabels,1,'int32');
trainlabelcount = fread(trainLabels,1,'int32');

%preparing the metadata for the test label set to be read
testlabelmagicnum = fread(testLabels,1,'int32');
testLabelCount = fread(testLabels,1,'int32');


toBeTrained=10000; %Only using the first 10000 images for training
toBeTested=100; %only using the first 100 images for testing

% arranging the set of training images in a 10000 X 784 size matrix and
% training labels in a 10000 X 1 size matrix
imgTrainArray = zeros(toBeTrained,(trainW*trainH));
for i=1:1:toBeTrained
    imgTrainArray(i,:)=fread(trainImgs,[1,trainW*trainH],'uint8');
end
labelTrainArray=fread(trainLabels,[toBeTrained,1],'uint8');

% arranging the set of testing images in a 100 X 784 size matrix and
% training labels in a 100 X 1 size matrix
testLabelArray=fread(testLabels,[toBeTested,1],'uint8'); %label
testImgsArray=zeros(toBeTested,(testW*testH));
for i=1:1:toBeTested
    testImgsArray(i,:)=fread(testImgs,[1,((testW*testH))],'uint8');
end


%We will create a quadratic kernel with SVM model and a regular SVM model
%value of C will be 10
C=10;

%{
creating the quadratic kernel svm model with
- s 0 & -c : Classification of SVM type 1 with equation 0.5 wTw+ C sum(Zetai), with C
- t 1: kernel type set to polynomial
- d 2: degree of polynomial set to 2 (quadratic)
- r 1: coefficient of kernel function set to 1,Therefore K(x,z)=(xz+1)^2
%}
quadString=sprintf('-s 0 -c %d -t 1 -d 2 -r 1',C);
quadraticModel=svmtrain(double(labelTrainArray),double(imgTrainArray),quadString);
%{
outputing predictions and accuracy:
- predict_label: stores the SVM prediction output vector
- accuracy: a vector with accuracy, mean squared error, squared correlation coefficient
- dec_values: probability estimate vector
%}
[quadPredict_label, quadAccuracy, quadDec_values] = svmpredict(double(testLabelArray), double(testImgsArray), quadraticModel); % test the training data




%{
creating the linear svm model with
- s 0 & -c : Classification of SVM type 1 with equation 0.5 wTw+ C sum(Zetai), with C
- t 0: kernel type set to linear
%}
linearString=sprintf('-s 0 -c %d -t 0',C);
linearModel=svmtrain(double(labelTrainArray),double(imgTrainArray), linearString);
%{
outputing predictions and accuracy:
- predict_label: stores the SVM prediction output vector
- accuracy: a vector with accuracy, mean squared error, squared correlation coefficient
- dec_values: probability estimate vector
%}
[linearPredict_label, linearAccuracy, linearDec_values] = svmpredict(double(testLabelArray), double(testImgsArray), linearModel); % test the training data

    




