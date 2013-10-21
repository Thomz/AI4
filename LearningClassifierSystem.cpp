/*
 * LearningClassifierSystem.cpp
 *
 *  Created on: Oct 10, 2013
 *      Author: Thomz
 */

#include "LearningClassifierSystem.h"

LearningClassifierSystem::LearningClassifierSystem(){
	cout << "LCS Created"<< endl;
	id = 1000;
	load();
}
LearningClassifierSystem::~LearningClassifierSystem(){
}

template <typename T>
string NumberToString ( T Number )
  {
     ostringstream ss;
     ss << Number;
     return ss.str();
	//mfskf
  }

void LearningClassifierSystem::learn(vector<Mat> descriptorsObjects, vector<Mat> singleObjects, string inputString){
	double t = (double)cv::getTickCount();

	bool found = false;
	int GAno;
	  // Search already known GAs to find desired object
	for(int i = 0; i < geneticAlgorithms.size(); i++){
		if(geneticAlgorithms[i].type == inputString){
			found = true;
			GAno = i;
		}
	}

	if( found ){
		cout << "Object already in database" << endl;

			// Instantiate votes, to call with function voteForObject
		int votes[descriptorsObjects.size()];

			// Clear votes array
		for(int i = 0; i < sizeof(votes)/sizeof(votes[0]); i++)
			votes[i] = 0;

			// Make all chromosomes for type vote which object should be chosen
		voteForObject(descriptorsObjects,  GAno, votes);

		cout << "Vote time: " << ((double)cv::getTickCount() - t)/cv::getTickFrequency() << endl;
		t = (double)cv::getTickCount();

			// Validate with user input that the correct object is chosen
		int rightObject = validateObject(descriptorsObjects, GAno, votes, singleObjects);

		cout << "Validate time: " << ((double)cv::getTickCount() - t)/cv::getTickFrequency() << endl;
		t = (double)cv::getTickCount();

			// Give scores to all the chromosomes
		scoreGivingGA(GAno, rightObject);

		cout << "Score giving time: " << ((double)cv::getTickCount() - t)/cv::getTickFrequency() << endl;
		t = (double)cv::getTickCount();
	}

		// If objct is not aklready in database
	else{
		cout << "Object not found in database" << endl;

			// Find which object is the one that is searched for and put it in a new GA
		findFirstInstance( descriptorsObjects, singleObjects, inputString);
	}

	save();

	return;
}

void LearningClassifierSystem::scoreGivingGA(int GAno, int rightObject){

}

int LearningClassifierSystem::validateObject(vector<Mat> descriptorsObjects, int gaNo, int * votes, vector<Mat> singleObjects, int* newObjectIDArr){
	// Instantiate highscores for use when sorting votes
	int highscore(0), highscoreObj(0);

	Mat tempDescriptorObject;
	int tempVotes, tempNewObjID;
	Mat tempSingleObjects;

	int newObjectIDArr[descriptorsObjects.size()];

	for(int i = 0; i < descriptorsObjects.size(); i++)
		newObjectIDArr[i] = i;

	for( int i = 0; i < descriptorsObjects.size(); i++){
		for(int j = 0; j < descriptorsObjects.size(); j++){
			if(votes[i] > votes[j]){
				tempNewObjID = newObjectIDArr[i]; newObjectIDArr[i] = newObjectIDArr[j]; newObjectIDArr[j] = tempNewObjID;
				tempDescriptorObject = descriptorsObjects[i].clone(); tempVotes = votes[i]; tempSingleObjects = singleObjects[i];
				descriptorsObjects[i] = descriptorsObjects[j].clone(); votes[i] = votes[j]; singleObjects[i] = singleObjects[j].clone();
				descriptorsObjects[j] = tempDescriptorObject.clone(); votes[j] = tempVotes; singleObjects[j] = tempSingleObjects.clone();
			}
		}
	}
		// Cout votes, to test that they are sorted!
	for(int i = 0; i < descriptorsObjects.size(); i++)
		cout << "Original img " <<  NumberToString(newObjectIDArr[i]) << " got " <<  votes[i] << endl;

		// String for userinput
	string userInpt;

		// Run through all sorted objects and ask the user which one is the right, hopefully the first one will be
	for(int i = 0; i < descriptorsObjects.size() ; i++){
			namedWindow("Picture", CV_WINDOW_NORMAL);
			waitKey(100);
			imshow("Picture", singleObjects[i]);
			waitKey(1);
			cout << "Is this the what you are looking for? (y/n)" << endl;
			cin >> userInpt;
			if(userInpt == "yes" ||userInpt == "Yes" || userInpt == "y" ){
				Chromosome tempChr = {descriptorsObjects[i].clone(),1, id};
				id++;
				geneticAlgorithms[gaNo].chromosomes.push_back(tempChr);
				return i;
			}
		}
}

void LearningClassifierSystem::voteForObject(vector<Mat> descriptorsObjects,  int gaNo, int * votes){
	int tempVotes[descriptorsObjects.size()];

	FlannBasedMatcher matcher;
	std::vector< DMatch > matches;

		// For every chromosome(set of features) with the wanted type
	for(int i = 0; i < geneticAlgorithms[gaNo].chromosomes.size(); i++){

			// Clear tempVotes
		for(int j = 0; j < sizeof(tempVotes)/sizeof(*tempVotes); j++)
			tempVotes[j] = 0;

		// For every object that is in the picture
		for(int j = 0; j < descriptorsObjects.size(); j++){

				// Match current object with current chromosome
			matcher.match( geneticAlgorithms[gaNo].chromosomes[i].features, descriptorsObjects[j], matches );

				// Calculate most fittes object
			for(int h = 0; h < matches.size(); h++)
				if(matches[h].distance < matchesThreshold)
					tempVotes[j]++;
		}
		int highscore(0), highscoreObj(0);

		for(int b = 0; b < descriptorsObjects.size(); b++){
			if(tempVotes[b] > highscore){
				highscore = tempVotes[b];
				highscoreObj = b;
			}
		}
		geneticAlgorithms[gaNo].chromosomes[i].lastVoteID = highscoreObj;
		cout << geneticAlgorithms[gaNo].chromosomes[i].lastVoteID << endl;
		votes[highscoreObj] += 1 * geneticAlgorithms[gaNo].chromosomes[i].score;
	}
}

void LearningClassifierSystem::findFirstInstance( vector<Mat> descriptorsObjects, vector<Mat> singleObjects, string inputString){
	string userInpt;

	for(int i = 0; i < singleObjects.size() ; i++){
		namedWindow("Picture", CV_WINDOW_NORMAL);
		waitKey(100);
		imshow("Picture", singleObjects[i]);
		waitKey(1);
		cout << "Is this the " <<  inputString << " you are looking for? (y/n)" << endl;
		cin >> userInpt;
		if(userInpt == "yes" ||userInpt == "Yes" || userInpt == "y" ){
			Chromosome tempChr = {descriptorsObjects[i].clone(),1, id};
			id++;
			GA tempGA;
			tempGA.type = inputString;
			tempGA.chromosomes.push_back(tempChr);
			geneticAlgorithms.push_back(tempGA);
			break;
		}
	}
	return;
}

void LearningClassifierSystem::saveObjectToDatabase(string objectName, string GAname, Mat descriptor){
	FileStorage fs("databaseLCS/" + GAname + "/" + objectName +  ".yml", FileStorage::WRITE);
	write( fs, "Descriptors", descriptor);
	fs.release();
}

Mat LearningClassifierSystem::findObjectInDatabase(string idObject, string gaType){

	Mat result;

	FileStorage fs("databaseLCS/" + gaType + "/"+ idObject + ".yml", FileStorage::READ);
	if (fs.isOpened() == 0){
		cout << "load fail" << endl;
		return result;
	}

	FileNode kptFileNode = fs["Descriptors"];
	read( kptFileNode, result);
	fs.release();

	return result;
}

void LearningClassifierSystem::load(){
	int highestId = id;
	ifstream typesFile;
	typesFile.open( "databaseLCS/types");
	string tempType;
	int types = 0;
	typesFile >> types;
	cout << "Objects found in database:" << endl;
	geneticAlgorithms.clear();
	for(int i = 0; i < types; i++){
		typesFile >> tempType;
		cout << tempType;
		if( i != types-1)
			cout << " - ";

		GA tempGA;
		tempGA.type = tempType;

		ifstream resultFile;
		string tempResult = "databaseLCS/" + tempType + "/result";
		resultFile.open( tempResult.c_str());
		int objects;
		resultFile >> objects;
		int tempID, tempScore;
		for(int j = 0; j < objects; j++){
			Chromosome tempChromo;
			resultFile >> tempID;
			if(tempID > highestId)
				highestId = tempID;
			resultFile >> tempScore;
			tempChromo.features = findObjectInDatabase(NumberToString(tempID), tempType).clone();
			//cout << endl << tempID << " - " << tempScore << endl;
			tempChromo.id = tempID;
			tempChromo.score = tempScore;
			tempGA.chromosomes.push_back(tempChromo);
		}

		geneticAlgorithms.push_back(tempGA);

	}

	id = highestId+1;
}

void LearningClassifierSystem::save(){
	ofstream typesFile;
	string tempTxtTypes = "databaseLCS/types";
	typesFile.open (tempTxtTypes.c_str());
	typesFile << NumberToString(geneticAlgorithms.size()) + "\n";
	for(int i = 0; i < geneticAlgorithms.size(); i++){
		if(	!checkDirs(geneticAlgorithms[i].type)){
			string tempString = "mkdir databaseLCS/" + geneticAlgorithms[i].type;
			int ret =  system(tempString.c_str());
		}

		typesFile << geneticAlgorithms[i].type + "\n" ;

		ofstream myfile;
		string tempTxt = "databaseLCS/" + geneticAlgorithms[i].type + "/result";
		myfile.open (tempTxt.c_str());
		myfile << NumberToString(geneticAlgorithms[i].chromosomes.size())<< endl;

		for(int j = 0; j < geneticAlgorithms[i].chromosomes.size(); j++){
			saveObjectToDatabase(NumberToString(geneticAlgorithms[i].chromosomes[j].id), geneticAlgorithms[i].type, geneticAlgorithms[i].chromosomes[j].features);
			myfile << NumberToString(geneticAlgorithms[i].chromosomes[j].id) + "\n" + NumberToString(geneticAlgorithms[i].chromosomes[j].score) +  "\n";
		}
		myfile.close();

	}
	typesFile.close();

}

bool LearningClassifierSystem::checkDirs(string objectName){
	string tempOpen = "databaseLCS/" + objectName;
	DIR* dir = opendir(tempOpen.c_str());
	if (dir){
		string tempDel = "rm databaseLCS/" + objectName + "/*";
		//int ret = system(tempDel.c_str());
		closedir(dir);
		return true;
	}
	else{
		return false;
	}
}
