/*
 * LearningClassifierSystem.cpp
 *
 *  Created on: Oct 10, 2013
 *      Author: Thomz
 */

#include "LearningClassifierSystem.h"

LearningClassifierSystem::LearningClassifierSystem(){
	cout << "LCS Created"<< endl;
}
LearningClassifierSystem::~LearningClassifierSystem(){
}

void LearningClassifierSystem::Test(){
	cout << "Test"<< endl;
}

void LearningClassifierSystem::learn(Mat descriptorsDatabase, vector<Mat> descriptorsObjects, vector<Mat> singleObjects, string inputString){
	bool found = false;
	  // Search already known GAs to find desired object
	for(int i = 0; i < geneticAlgorithms.size(); i++){
		if(geneticAlgorithms[i].type == inputString){
			found = true;
		}
	}

	if( found ){
		// Do some AI stuff
		cout << "Object already in database" << endl;
		int i = 0;
	}
	else{
		cout << "Object not found in database" << endl;
		findFirstInstance( descriptorsObjects, singleObjects, inputString);
	}

	return;

}

void saveObjectToDatabase(string objectName, Mat descriptor){
	FileStorage fs("database/" + objectName + ".yml", FileStorage::WRITE);
	write( fs, "Descriptors", descriptor);
	fs.release();
}

void LearningClassifierSystem::findFirstInstance( vector<Mat> descriptorsObjects, vector<Mat> singleObjects, string inputString){
	string userInpt;

	for(int i = 0; i < singleObjects.size() ; i++){
		namedWindow("Picture", CV_WINDOW_NORMAL);
		waitKey(100);
		imshow("Picture", singleObjects[i]);
		waitKey(1);
		cout << "Please enter if this is " << inputString << endl;
		cin >> userInpt;
		if(userInpt == "yes" ||userInpt == "Yes" || userInpt == "y" ){
			Chromosome tempChr = {descriptorsObjects[i],0};
			GA tempGA;
			tempGA.type = inputString;
			tempGA.chromosomes.push_back(tempChr);
			break;
		}
	}

	return;

}
