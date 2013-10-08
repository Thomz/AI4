#include <iostream>
#include <opencv2/opencv.hpp>
#include "queue"
#include "ImageFiltering.hpp"

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
		cout << "Object not found in database" << endl;
		return result;
	}

	FileNode kptFileNode = fs["Descriptors"];
	read( kptFileNode, result);
	fs.release();

	return result;
}

void saveObjectToDatabase(string objectName, Mat descriptor){
	FileStorage fs("database/" + objectName + ".yml", FileStorage::WRITE);
	write( fs, "Descriptors", descriptor);
	fs.release();
}

/*

Mat findObjectInDatabase(string inputObject){
	Mat desc = imread("database/" + inputObject + ".png", );
	imshow("hej", desc);
}

void saveObjectInDatabase(string objectName, Mat descriptor){
	imwrite("database/" + objectName + ".png", descriptor);
}
*/
int main(int argc, char **argv) {

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

	if( databaseDesc.cols != 0){
		FlannBasedMatcher matcher;
		sortDescriptors(databaseDesc,descriptorVec,matcher, singleObjects);
	}

	for(int i = 0; i < singleObjects.size(); i++)
		imshow(NumberToString(i+10), singleObjects[i]);


	//	cout << keypointsObject.size() << endl;

	//if( keypointsObject.size() < 1 )


	waitKey();

	return 0;
}

