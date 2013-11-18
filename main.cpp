#include <iostream>
#include "ImageFiltering.hpp"
#include "KohonenNetwork.h"


#define showSingleObjectsWithKeypoints false

vector<vector<Mat> > evaluateDescriptors;
vector<vector<Mat> > evaluateObjects;

struct color{
	vector<double> rgb;
};

vector<color> colors;


string getInputObject(){
	string inputObject;

	//cout << endl << "Write name of wanted object:" << endl;

	//cin >> inputObject;
	inputObject = evaluationObject;

	//cout << "Looking for " + inputObject << endl;

	return inputObject;
}

void makeColors(){
	color red, green, blue, yellow, black, white, purple, grey;

	double redD[] = {1,0,0};
	vector<double> redV(redD, redD+3);
	red.rgb = redV;
	colors.push_back(red);

	double greenD[] = {0.32,0.111,0.98};
	vector<double> greenv(greenD, greenD+3);
	green.rgb = greenv;
	colors.push_back(green);

	double blueD[] = {0,0,1};
	vector<double> blueV(blueD, blueD+3);
	blue.rgb = blueV;
	colors.push_back(blue);

	double yellowD[] = {1,1,0};
	vector<double> yellowV(yellowD, yellowD+3);
	yellow.rgb = yellowV;
	colors.push_back(yellow);

	double blackD[] = {0,0,0};
	vector<double> blackV(blackD, blackD+3);
	black.rgb = blackV;
	colors.push_back(black);

	double whiteD[] = {1,1,1};
	vector<double> whiteV(whiteD, whiteD+3);
	white.rgb = whiteV;
	colors.push_back(white);
/*
	double purpleD[] = {128,0,128};
	vector<double> purpleV(purpleD, purpleD+3);
	purple.rgb = purpleV;
	colors.push_back(purple);

	double greyD[] = {128,128,128};
	vector<double> greyV(greyD, greyD+3);
	grey.rgb = greyV;
	colors.push_back(grey);
	*/
}

void readEvaluationSet(){
	for(int i = 1; i <  evaluationPics ; i++){

		Mat src = getImageEvaluation(i);

		Mat filtered;
		vector<Mat> singleObjects;
		singleObjects = filterSurrounding(src, filtered);
		vector<KeyPoint> keypoints;

		Mat output;
		Mat descriptor;
		vector<Mat> descriptorVec;

		SiftDescriptorExtractor extractor;

		for(int j=0; j<singleObjects.size(); j++){
			getOverlay(src, singleObjects[j]);
			keypoints = getKeypointsFromObject(singleObjects[j]);
			descriptor = getDescriptorsFromObject(singleObjects[j],keypoints, extractor);
			descriptorVec.push_back(descriptor.clone());
		}

		evaluateDescriptors.push_back(descriptorVec);
		evaluateObjects.push_back(singleObjects);

	}
	cout << "Evaluation set read" << endl;
}

void runKohonen(){
	KohonenNetwork KNN(200,6);
	//KNN.printNetwork();

	KNN.showAsImage("Before");

	double tempD[] = {0.3, 0.4, 0.12, 0.2, 0.6};
	vector<double> inVector(tempD, tempD+6);

	cout << "Making kohonen network" << endl;;
	for(int i = 1; i < totalIterations; i++){


		/*if((double) rand() / (RAND_MAX) > 0.5)
			inVector[j] = 1;
		else
			inVector[j] = 0;*/
		//cout <<(((double) i/totalIterations)*100 )<< endl;

		//inVector[j] = (double) rand() / (RAND_MAX);

		Mat src = getImage(i);
		cout << i << endl;

		Mat filtered;
		vector<Mat> singleObjects;
		singleObjects = filterSurrounding(src, filtered);
		vector<KeyPoint> keypoints;

		Mat output;
		Mat descriptor;
		vector<Mat > descriptorVec;

		cout << "New iteration" << endl;

		SiftDescriptorExtractor extractor;

		for(int i=0; i<singleObjects.size(); i++){
			getOverlay(src, singleObjects[i]);
			//keypoints = getKeypointsFromObject(singleObjects[i]);
			//descriptor = getDescriptorsFromObject(singleObjects[i],keypoints, extractor);
			//descriptorVec.push_back(descriptor.clone());
			//if(showSingleObjectsWithKeypoints){
			//	drawKeypoints(singleObjects[i], keypoints, singleObjects[i]);
			//	imshow(NumberToString(i), singleObjects[i]);
			//}
			cout << "Custom desc" << endl;
			vector<double> customDesc =  getCustomObjectDescriptor(singleObjects[i]);
			cout << "kohonen" << endl;
			KNN.adjustWeights(customDesc);
		}

		cout << "end" << endl;
	}

	cout << " done" << endl;

	KNN.showAsImage("After");

	waitKey();
}

int main(int argc, char **argv) {

	runKohonen();

	return 0;

	//makeColors();

	cout << "Program started" << endl;

	readEvaluationSet();

	LearningClassifierSystem LCS;
	int pic = startPic;
	int i = 0;
	while(true){

		string inputObject = "cola";
		Mat src = getImage(i);
		Mat filtered;
		vector<Mat> singleObjects;
		singleObjects = filterSurrounding(src, filtered);
		vector<KeyPoint> keypoints;

		Mat output;
		Mat descriptor;
		vector<Mat > descriptorVec;

		SiftDescriptorExtractor extractor;

		for(int i=0; i<singleObjects.size(); i++){
			getOverlay(src, singleObjects[i]);
			keypoints = getKeypointsFromObject(singleObjects[i]);
			descriptor = getDescriptorsFromObject(singleObjects[i],keypoints, extractor);
			descriptorVec.push_back(descriptor.clone());
			if(showSingleObjectsWithKeypoints){
				drawKeypoints(singleObjects[i], keypoints, singleObjects[i]);
				imshow(NumberToString(i), singleObjects[i]);
			}
		}

		//cout << "Vision time: " <<  ((double)cv::getTickCount() - t)/cv::getTickFrequency() << "s" << endl;

		LCS.learn(descriptorVec, singleObjects, inputObject);

			// Coca-cola
		//int correctObjects[] = {1,2,3,1,2,3,1,1,4};
			//Jagermeister
		//int correctObjects[] = {2,2,2,1,1,3,1,1,0,3};
			// Candle
		int correctObjects[] = {3,0,1,2,0,0,1,1,2,1};

		int correctGuesses = 0;
		if(evaluation){
			for(int i = 0; i <  evaluateDescriptors.size() ; i++){
				int score = LCS.evaluateAll(evaluateDescriptors[i], evaluateObjects[i], inputObject,correctObjects[i-1]);
				correctGuesses += score;
			}

			cout << "Evaluation - Correct percentage: " << (double)correctGuesses/(sizeof(correctObjects)/sizeof(correctObjects[0])) << endl;

		}
	}

	return 0;
}

