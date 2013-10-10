/*
 * FeatureSelector.cpp
 *
 *  Created on: Oct 10, 2013
 *      Author: Simon
 */

#include "FeatureSelector.h"



FeatureSelector::FeatureSelector() {
	// TODO Auto-generated constructor stub

}

FeatureSelector::~FeatureSelector() {
	// TODO Auto-generated destructor stub
}

void FeatureSelector::test(){
	cout << "efs" << endl;
}

Mat FeatureSelector::getObjectFeatures(string inputObject){

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

Mat FeatureSelector::getObjectWeights(string inputObject){

	Mat result;

	FileStorage fs("database/" + inputObject + "Weights" +  ".yml", FileStorage::READ);
	if (fs.isOpened() == 0){
		cout << "Object not found in database" << endl;
		return result;
	}

	FileNode kptFileNode = fs["Weights"];
	read( kptFileNode, result);
	fs.release();

	return result;
}

void FeatureSelector::updateWeights(string objectName, Mat currentFeatures){
	Mat loadedFeatures = getObjectFeatures(objectName);
	Mat weights;
	if(loadedFeatures.rows!=0){
		weights = getObjectWeights(objectName);

	}
	else {
		cout << "can't update non-existing file --> creating new files" << endl;
		saveObjectFeatures(objectName,currentFeatures);
		for(int i=0; i < currentFeatures.rows; i++){
			weights.push_back(0);
		}
		saveFeatureWeights(objectName,weights);
	}
	//update
}

void FeatureSelector::saveObjectFeatures(string objectName, Mat descriptor){
	FileStorage fs("database/" + objectName + ".yml", FileStorage::WRITE);
	write( fs, "Descriptors", descriptor);
	fs.release();
}

void FeatureSelector::saveFeatureWeights(string objectName, Mat weights){
	FileStorage fs("database/" + objectName + "Weights" + ".yml", FileStorage::WRITE);
	write( fs, "Weights", weights);
	fs.release();
}
