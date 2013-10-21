#include <iostream>
#include <opencv2/opencv.hpp>
#include "ImageFiltering.hpp"
#include "LearningClassifierSystem.h"

#define showSingleObjectsWithKeypoints false

string getInputObject(){
	string inputObject;

	cout << endl << "Write name of wanted object:" << endl;

	cin >> inputObject;

	cout << "Looking for " + inputObject << endl;

	return inputObject;
}

int main(int argc, char **argv) {

	LearningClassifierSystem LCS;
	int pic = 1;
	while(true){
		destroyAllWindows();

		Mat src = getImage(pic++);

		namedWindow("Picture", CV_GUI_NORMAL);
		waitKey(1000);
		imshow("Picture", src);
		waitKey(1);
		string inputObject = getInputObject();
		destroyAllWindows();

		double t = (double)cv::getTickCount();

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

		cout << "Vision time:" <<  ((double)cv::getTickCount() - t)/cv::getTickFrequency() << endl;

		//Mat databaseDesc = findObjectInDatabase(inputObject);

		LCS.learn(descriptorVec, singleObjects, inputObject);

		//LCS.Test();
	}

	return 0;
}


