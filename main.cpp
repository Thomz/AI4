#include <iostream>
#include <opencv2/opencv.hpp>
#include "ImageFiltering.hpp"
#include "LearningClassifierSystem.h"

#define showSingleObjectsWithKeypoints false

string getInputObject(){
	string inputObject;

	cout << "Write name of wanted object:" << endl;

	cin >> inputObject;

	cout << "Looking for " + inputObject << endl;

	return inputObject;
}

Mat findObjectInDatabase(string inputObject){

	Mat result;

	FileStorage fs("database/" + inputObject + ".yml", FileStorage::READ);
	if (fs.isOpened() == 0){
		return result;
	}

	FileNode kptFileNode = fs["Descriptors"];
	read( kptFileNode, result);
	fs.release();

	return result;
}


int main(int argc, char **argv) {

	LearningClassifierSystem LCS;

	while(true){

		string inputObject = getInputObject();

		Mat src = getImage();

		Mat filtered;
		vector<Mat> singleObjects;
		singleObjects = filterSurrounding(src, filtered);
		vector<KeyPoint> keypoints;

		Mat output;
		Mat descriptor;
		vector<Mat> descriptorVec;

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

		Mat databaseDesc = findObjectInDatabase(inputObject);

		//LCS.learn(databaseDesc, descriptorVec, singleObjects, inputObject);

		LCS.Test();
	}

	return 0;
}

