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
	int* votes;
	int highscore(0), highscoreObj(0);
	if( found ){
		// Do some AI stuff
		cout << "Object already in database" << endl;
		votes = voteForObject(descriptorsObjects,  GAno);
		for(int b = 0; b < sizeof(votes)/sizeof(*votes); b++){
			cout << votes[b] << endl;
			if(votes[b] > highscore){
				highscore = votes[b];
				highscoreObj = b;
			}
		}

		imshow("highscore", singleObjects[highscoreObj]);
	}
	else{
		cout << "Object not found in database" << endl;
		findFirstInstance( descriptorsObjects, singleObjects, inputString);
	}

	return;

}

int* LearningClassifierSystem::voteForObject(vector<Mat> descriptorsObjects,  int gaNo){
	int votes[descriptorsObjects.size()];
	int tempVotes[descriptorsObjects.size()];

	for(int i = 0; i < sizeof(votes)/sizeof(*votes); i++)
		votes[i] = 0;

//	for(int i = 0; i < sizeof(votes)/sizeof(*votes); i++)
//		cout << votes[i] << endl;

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

		for(int b = 0; b < sizeof(votes)/sizeof(*votes); b++){
			if(tempVotes[b] > highscore){
				highscore = tempVotes[b];
				highscoreObj = b;
			}
		}
		votes[highscoreObj]++;
	}

	return votes;

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
