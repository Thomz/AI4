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
		//cout << "Object already in database" << endl;

			// Instantiate votes, to call with function voteForObject
		double votes[descriptorsObjects.size()];

			// Clear votes array
		for(int i = 0; i < sizeof(votes)/sizeof(votes[0]); i++)
			votes[i] = 0;

			// Make all chromosomes for type vote which object should be chosen
		voteForObject(descriptorsObjects,  GAno, votes);

		//for(int i = 0; i < descriptorsObjects.size(); i++)
			//cout << votes[i] << endl;

		//cout << "Vote time: " << ((double)cv::getTickCount() - t)/cv::getTickFrequency() << "s"<< endl;
		t = (double)cv::getTickCount();

		int newObjectIDArr[descriptorsObjects.size()];

			// Validate with user input that the correct object is chosen
		int rightObject = validateObject(descriptorsObjects, GAno, votes, singleObjects, newObjectIDArr, false);

		//cout << "Validate time: " << ((double)cv::getTickCount() - t)/cv::getTickFrequency() << "s" << endl;
		t = (double)cv::getTickCount();

		if(rightObject != -1){
			// Give scores to all the chromosomes
			scoreGivingGA(GAno, rightObject, newObjectIDArr, descriptorsObjects);
			updateGA(GAno);
		}
		else
			cout << "No object validated" << endl;

		//cout << "Score giving time: " << ((double)cv::getTickCount() - t)/cv::getTickFrequency() << endl;
		t = (double)cv::getTickCount();
	}

		// If object is not aklready in database
	else{
		cout << "Object not found in database" << endl;

			// Find which object is the one that is searched for and put it in a new GA
		findFirstInstance( descriptorsObjects, singleObjects, inputString);
	}

	save();

	return;
}

void LearningClassifierSystem::updateGA(int GAno){
	if(geneticAlgorithms[GAno].chromosomes.size() > maxChromosomes){
		cout << "Chromosomes reached maximum, updating" << endl;

		Chromosome tmpChr;

		for(int i = 0; i < geneticAlgorithms[GAno].chromosomes.size(); i++){
			for(int j = 0; j < geneticAlgorithms[GAno].chromosomes.size(); j++){
				if(geneticAlgorithms[GAno].chromosomes[i].score > geneticAlgorithms[GAno].chromosomes[j].score){
					tmpChr = geneticAlgorithms[GAno].chromosomes[i];
					geneticAlgorithms[GAno].chromosomes[i] = geneticAlgorithms[GAno].chromosomes[j];
					geneticAlgorithms[GAno].chromosomes[j] = tmpChr;
				}
			}
		}

		while(geneticAlgorithms[GAno].chromosomes.size() > chrCutdown)
			geneticAlgorithms[GAno].chromosomes.pop_back();

	}
}

void LearningClassifierSystem::scoreGivingGA(int GAno, int rightObject, int* newObjectIDArr, vector<Mat> descriptorsObjects){
	/*for(int i = 0; i < descriptorsObjects.size(); i++)
		cout << newObjectIDArr[i] << endl;*/

	// newObject[XX]: XX = original place

	for(int i =0 ; i < geneticAlgorithms[GAno].chromosomes.size(); i++){
		if(newObjectIDArr[geneticAlgorithms[GAno].chromosomes[i].lastVoteID] == rightObject ){
			if( geneticAlgorithms[GAno].chromosomes[i].score < upperThres )
				geneticAlgorithms[GAno].chromosomes[i].score += rightUpCnt;
		}
		else
			geneticAlgorithms[GAno].chromosomes[i].score *= degFactor;
	}
}

int LearningClassifierSystem::validateObject(vector<Mat> descriptorsObjects, int gaNo, double * votes, vector<Mat> singleObjects, int* newObjectIDArr, bool eval){
	// Instantiate highscores for use when sorting votes
	int highscore(0), highscoreObj(0);

	Mat tempDescriptorObject;
	double tempVotes, tempNewObjID;
	Mat tempSingleObjects;

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

		// String for userinput
	string userInpt;

	if(!eval){
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
	else{
		return newObjectIDArr[0];
	}

	return -1;
}

void LearningClassifierSystem::voteForObject(vector<Mat> descriptorsObjects,  int gaNo, double * votes){
	double tempVotes[descriptorsObjects.size()];

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

			// Cout for printing what the different chromosomes voted for
		//cout << geneticAlgorithms[gaNo].chromosomes[i].lastVoteID << endl;

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

int LearningClassifierSystem::evaluateAll(vector<Mat> descriptorsObjects, vector<Mat> singleObjects, string inputString, int correctObject){

		// GA Stuff from learn algorithms
	bool found = false;
	int GAno;
	for(int i = 0; i < geneticAlgorithms.size(); i++){
		if(geneticAlgorithms[i].type == inputString){
			found = true;
			GAno = i;
		}
	}


		// Instantiate votes, to call with function voteForObject
	double votes[descriptorsObjects.size()];

		// Clear votes array
	for(int i = 0; i < sizeof(votes)/sizeof(votes[0]); i++)
	votes[i] = 0;

	voteForObject(descriptorsObjects,  GAno, votes);

	int newObjectIDArr[descriptorsObjects.size()];

	int rightObject = validateObject(descriptorsObjects, GAno, votes, singleObjects, newObjectIDArr, true);

	if(rightObject == correctObject)
		return 1;

	return 0;

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
		double tempID, tempScore;
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

	cout << endl;
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
		int ret = system(tempDel.c_str());
		closedir(dir);
		return true;
	}
	else{
		return false;
	}
}
