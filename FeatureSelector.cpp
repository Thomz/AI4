/*
 * FeatureSelector.cpp
 *
 *  Created on: Oct 10, 2013
 *      Author: Simon
 */

#include "FeatureSelector.h"

#define MAXFEATURES 1000
#define DEGRADEFACTOR 0.8



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
		matcher.match( currentFeatures, loadedFeatures,  matches );

		for( int i = 0; i < matches.size(); i++ ){
			if(matches[i].distance<100){
				weights.at<double>(matches[i].trainIdx,0)+=1;
			}
			else{
				weights.at<double>(matches[i].trainIdx,0)*=DEGRADEFACTOR;
				loadedFeatures.push_back(currentFeatures.row(matches[i].queryIdx));
				weights.push_back(1.);
			}
		}
		filterFeaturesAndWeights(loadedFeatures,weights);
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
	if(weights.rows > MAXFEATURES*2){
		double highest = 0;
		int highestIndex=0;
		Mat filteredFeatures;
		Mat_<double> filteredWeights;
		for(int i=0; i<MAXFEATURES*2; i++){
			for(int j=0; j < weights.rows; j++){
				if(weights.at<double>(j,0) > highest){
					highest = weights.at<double>(j,0);
					highestIndex = j;
				}
			}
			if(highest==1){
				cout << "no highest" << endl;
				break;
			}
			filteredFeatures.push_back(features.row(highestIndex));
			filteredWeights.push_back(weights.row(highestIndex));
			weights.at<double>(highestIndex,0)=-1;
			highest=0;
			highestIndex=0;
		}
		int i=0;
		while(filteredWeights.rows < MAXFEATURES){
			if(weights.at<double>(weights.rows-1-i,0)!=-1){
				filteredWeights.push_back(weights.at<double>(weights.rows-1-i,0));
				filteredFeatures.push_back(features.row(weights.rows-1-i));
			}
			i++;
		}
		features = filteredFeatures.clone();
		weights = filteredWeights.clone();
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

int FeatureSelector::compareFeatures(Mat& currentFeatures, Mat& loadedFeatures){
	int noOfSimilarFeatures = 0;

	FlannBasedMatcher matcher;
	vector< DMatch > matches;
	matcher.match( currentFeatures, loadedFeatures, matches );

	for( int i = 0; i < matches.size(); i++ ){
		if(matches[i].distance<100){
			noOfSimilarFeatures++;
		}
	}

	return noOfSimilarFeatures;
}

vector<int> FeatureSelector::sortDoubleVector(vector<int> indexVector, vector<int> valueVector){
	vector<int> sortedIndexes;
	int indexOfHighestValue=0;
	int highestValue=0;
	for(int i=0; i<valueVector.size(); i++){
		for(int j=0; j<valueVector.size(); j++){
			if(valueVector[j] > highestValue)
				indexOfHighestValue=j;
				highestValue=valueVector[j];
			}
		sortedIndexes.push_back(indexVector[indexOfHighestValue]);
		indexVector[indexOfHighestValue]=-1;
		valueVector[indexOfHighestValue]=-1;
		indexOfHighestValue=0;
	}
	return sortedIndexes;

}

Mat FeatureSelector::findCorrectObject( string& inputString, vector<Mat>& descriptorsObjects, vector<Mat>& singleObjects, Mat nextImage){
	string userInpt;
	Mat descriptors;
	Mat loadedDescriptors = getObjectFeatures(inputString);
	if(loadedDescriptors.rows!=0){
		vector<int> noOfMatches;
		vector<int> indexVector;
		for(int i = 0; i < descriptorsObjects.size() ; i++){
			noOfMatches.push_back(compareFeatures(descriptorsObjects[i],loadedDescriptors));
			indexVector.push_back(i);
			}

		vector<int> sorted = sortDoubleVector(indexVector,noOfMatches);

		for(int i=0; i<sorted.size(); i++){
			namedWindow("Picture", CV_WINDOW_NORMAL);
			cout << "showing image: " << sorted[i] << " of: " << sorted.size() << endl;
			//namedWindow("Next image", CV_WINDOW_NORMAL);
			waitKey(100);
			imshow("Picture", singleObjects[sorted[i]]);
			//imshow("Next image", nextImage);
			waitKey(1);
			cout << "Please enter if this is " << inputString << endl;
			cin >> userInpt;
			if(userInpt == "yes" ||userInpt == "Yes" || userInpt == "y" ){
				descriptors = descriptorsObjects[sorted[i]].clone();
				return descriptors;
			}
		}


	}
	else{
		descriptors=findCorrectObjectSimple(inputString,descriptorsObjects,singleObjects,nextImage);
	}


	return descriptors;
}

Mat FeatureSelector::findCorrectObjectSimple( string& inputString, vector<Mat>& descriptorsObjects, vector<Mat>& singleObjects, Mat nextImage){
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

bool FeatureSelector::evaluateImg( string& inputString, vector<Mat>& descriptorsObjects, vector<Mat>& singleObjects, int correctImg){
	string userInpt;
	Mat descriptors;
	Mat loadedDescriptors = getObjectFeatures(inputString);
	vector<int> noOfMatches;
	vector<int> indexVector;
	for(int i = 0; i < descriptorsObjects.size() ; i++){
		noOfMatches.push_back(compareFeatures(descriptorsObjects[i],loadedDescriptors));
		indexVector.push_back(i);
		}

	vector<int> sorted = sortDoubleVector(indexVector,noOfMatches);

	if(correctImg==sorted[0]){
		return true;
	}
	return false;
}


