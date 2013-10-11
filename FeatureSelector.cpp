/*
 * FeatureSelector.cpp
 *
 *  Created on: Oct 10, 2013
 *      Author: Simon
 */

#include "FeatureSelector.h"

#define maxFeatures 2000



FeatureSelector::FeatureSelector() {
	// TODO Auto-generated constructor stub

}

FeatureSelector::~FeatureSelector() {
	// TODO Auto-generated destructor stub
}

void FeatureSelector::test(){
	cout << "efs" << endl;
}

Mat FeatureSelector::getObjectFeatures(string& inputObject){

	Mat result;

	FileStorage fs("databaseSimon/" + inputObject + ".yml", FileStorage::READ);
	if (fs.isOpened() == 0){
		cout << "Object not found in database" << endl;
		return result;
	}

	FileNode kptFileNode = fs["Descriptors"];
	read( kptFileNode, result);
	fs.release();

	return result;
}

Mat FeatureSelector::getObjectWeights(string& inputObject){

	Mat result;

	FileStorage fs("databaseSimon/" + inputObject + "Weights" +  ".yml", FileStorage::READ);
	if (fs.isOpened() == 0){
		cout << "Object not found in database" << endl;
		return result;
	}

	FileNode kptFileNode = fs["Weights"];
	read( kptFileNode, result);
	fs.release();

	return result;
}

void FeatureSelector::updateWeights(string& objectName, Mat& currentFeatures){
	Mat loadedFeatures = getObjectFeatures(objectName);
	Mat_<double> weights;
	if(loadedFeatures.rows!=0){
		weights = getObjectWeights(objectName);
		FlannBasedMatcher matcher;
		vector< DMatch > matches;
		matcher.match( loadedFeatures, currentFeatures, matches );
		cout << matches.size()<<endl;

		for( int i = 0; i < matches.size(); i++ ){
			if(matches[i].distance<200){
				weights.at<double>(matches[i].queryIdx,0)+=1;
			}
			else{
				loadedFeatures.push_back(currentFeatures.row(matches[i].trainIdx));
				weights.push_back(1.);
			}
		}
		saveObjectFeatures(objectName,loadedFeatures);
		saveFeatureWeights(objectName,weights);
	}
	else {
		cout << "can't update non-existing file --> creating new files" << endl;
		saveObjectFeatures(objectName,currentFeatures);
		for(int i=0; i < currentFeatures.rows; i++){
			weights.push_back(1.);
		}
		saveFeatureWeights(objectName,weights);
	}
}

void FeatureSelector::filterFeaturesAndWeights(Mat& features, Mat& weights){
	if(weights.rows > maxFeatures){
		double highest = 0;
		int highestIndex=0;
		Mat filteredFeatures;
		Mat_<double> filteredWeights;
		for(int i=0; i<2000; i++){
			for(int j=0; j < weights.rows; j++){
				if(weights.at<double>(j,0) > highest)
					highest = weights.at<double>(j,0);
					highestIndex = j;
			}
			filteredFeatures.push_back(filteredFeatures.row(highestIndex));
			filteredWeights.push_back(weights.row(highestIndex));
			weights.at<double>(highestIndex,0)=-1;
		}
		if(filteredWeights.rows < 2000){
			// NOT DONE!!
		}

	}
}

void FeatureSelector::saveObjectFeatures(string& objectName, Mat& currentFeatures){
	FileStorage fs("databaseSimon/" + objectName + ".yml", FileStorage::WRITE);
	write( fs, "Descriptors", currentFeatures);
	fs.release();
}

void FeatureSelector::saveFeatureWeights(string& objectName, Mat& weights){
	FileStorage fs("databaseSimon/" + objectName + "Weights" + ".yml", FileStorage::WRITE);
	write( fs, "Weights", weights);
	fs.release();
}

Mat FeatureSelector::findCorrectObject( string& inputString, vector<Mat>& descriptorsObjects, vector<Mat>& singleObjects, Mat nextImage){
	string userInpt;
	Mat descriptors;
	for(int i = 0; i < singleObjects.size() ; i++){
		namedWindow("Picture", CV_WINDOW_NORMAL);
		namedWindow("Next image", CV_WINDOW_NORMAL);
		waitKey(100);
		imshow("Picture", singleObjects[i]);
		imshow("Next image", nextImage);
		waitKey(1);
		cout << "Please enter if this is " << inputString << endl;
		cin >> userInpt;
		if(userInpt == "yes" ||userInpt == "Yes" || userInpt == "y" ){
			descriptors = descriptorsObjects[i].clone();
			return descriptors;
		}
	}

	return descriptors;

}


