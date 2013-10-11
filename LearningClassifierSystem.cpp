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

template <typename T>
  string NumberToString ( T Number )
  {
     ostringstream ss;
     ss << Number;
     return ss.str();
	//mfskf
  }

void LearningClassifierSystem::learn(vector<Mat> descriptorsObjects, vector<Mat> singleObjects, string inputString){
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

		validateObject(descriptorsObjects, GAno, votes, singleObjects);

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

void LearningClassifierSystem::validateObject(vector<Mat> descriptorsObjects, int gaNo, int * votes, vector<Mat> singleObjects){
	// Instantiate highscores for use when sorting votes
	int highscore(0), highscoreObj(0);

	Mat tempDescriptorObject;
	int tempVotes;
	Mat tempSingleObjects;

	for( int i = 0; i < descriptorsObjects.size(); i++){
		for(int j = 0; j < descriptorsObjects.size(); j++){
			if(votes[i] > votes[j]){
				tempDescriptorObject = descriptorsObjects[i].clone(); tempVotes = votes[i]; tempSingleObjects = singleObjects[i];
				descriptorsObjects[i] = descriptorsObjects[j].clone(); votes[i] = votes[j]; singleObjects[i] = singleObjects[j].clone();
				descriptorsObjects[j] = tempDescriptorObject.clone(); votes[j] = tempVotes; singleObjects[j] = tempSingleObjects.clone();
			}
		}
	}
		// Cout votes, to test that they are sorted!
	for(int i = 0; i < descriptorsObjects.size(); i++)
		cout << votes[i] << endl;

		// String for userinput
	string userInpt;

		// Run through all sorted objects and ask the user which one is the right, hopefully the first one will be
	for(int i = 0; i < descriptorsObjects.size() ; i++){
			namedWindow("Highscore", CV_WINDOW_NORMAL);
			waitKey(100);
			imshow("Highscore", singleObjects[i]);
			waitKey(1);
			cout << "Is this the what you are looking for? (y/n)" << endl;
			cin >> userInpt;
			if(userInpt == "yes" ||userInpt == "Yes" || userInpt == "y" ){
				Chromosome tempChr = {descriptorsObjects[i].clone(),0};
				geneticAlgorithms[gaNo].chromosomes.push_back(tempChr);
				break;
			}
		}

		// Show highest votes object
	//imshow("highscore", singleObjects[0]);
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
		votes[highscoreObj]++;
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
			Chromosome tempChr = {descriptorsObjects[i].clone(),0};
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
	FileStorage fs("database/" + GAname + "/" + objectName +  ".yml", FileStorage::WRITE);
	write( fs, "Descriptors", descriptor);
	fs.release();
}

void LearningClassifierSystem::load(){

}

void LearningClassifierSystem::save(){
	for(int i = 0; i < geneticAlgorithms.size(); i++){
		if(	!checkDirs(geneticAlgorithms[i].type)){
			string tempString = "mkdir database/" + geneticAlgorithms[i].type;
			int ret =  system(tempString.c_str());
		}

/*
		ofstream myfile;
		string tempTxt = "database/" + geneticAlgorithms[i].type + "result.txt";
		myfile.open (tempTxt.c_str());
		myfile << "Writing this to a file.\n";
		myfile.close();
		*/

		for(int j = 0; j < geneticAlgorithms[i].chromosomes.size(); j++){
			saveObjectToDatabase(NumberToString(j), geneticAlgorithms[i].type, geneticAlgorithms[i].chromosomes[j].features);
		}
	}
}

bool LearningClassifierSystem::checkDirs(string objectName){
	string tempOpen = "database/" + objectName;
	DIR* dir = opendir(tempOpen.c_str());
	if (dir){
		string tempDel = "rm database/" + objectName + "/*";
		int ret = system(tempDel.c_str());
		closedir(dir);
		return true;
	}
	else{
		return false;
	}
}
