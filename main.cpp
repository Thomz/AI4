#include <iostream>
#include <opencv2/opencv.hpp>
#include "queue"
#include "ImageFiltering.hpp"

string getInputObject(){
	string inputObject;

	cout << "Write name of wanted object:" << endl;

	cin >> inputObject;

	cout << "Looking for " + inputObject << endl;

	return inputObject;
}

Vector<KeyPoint> findObjectInDatabase(string inputObject){

	vector<KeyPoint> result;

	FileStorage fs("test.yml", FileStorage::READ);
	if (fs.isOpened() == 0){
		cout << "Object not found in database" << endl;
		return result;
	}
	FileNode kptFileNode = fs["keypoints"];
	read( kptFileNode, result);
	fs.release();
	cout << result.size() << endl;

	return result;
	;
}

int main(int argc, char **argv) {

	string inputObject = getInputObject();
	vector<KeyPoint> keypointsObject  = findObjectInDatabase(inputObject);


	Mat src = getImage();

	Mat filtered;
	vector<Mat> singleObjects;
	singleObjects = filterSurrounding(src, filtered);
	vector<KeyPoint> keypoints;

	Mat output;

	for(int i=0; i<singleObjects.size(); i++){
		getOverlay(src, singleObjects[i]);
		keypoints = getKeypointsFromObject(singleObjects[i]);
		drawKeypoints(singleObjects[i], keypoints, singleObjects[i]);
		imshow(NumberToString(i), singleObjects[i]);
	}




	waitKey();

	return 0;
}

