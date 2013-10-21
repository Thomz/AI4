/*
 * LearningClassifierSystem.h
 *
 *  Created on: Oct 10, 2013
 *      Author: linuxthomz
 */

#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <sys/types.h>
#include <sys/stat.h>
#include <dirent.h>
#include <fstream>
#include <istream>
#include <sys/time.h>


#define matchesThreshold 200

using namespace cv;
using namespace std;

#ifndef LEARNINGCLASSIFIERSYSTEM_H_
#define LEARNINGCLASSIFIERSYSTEM_H_

class LearningClassifierSystem {
public:
	// Constructor and destructor
	LearningClassifierSystem();
	virtual ~LearningClassifierSystem();

	void learn(vector<Mat> descriptorsObjects,
				 	  vector<Mat> singleObjects,
				 	  string inputString);


private:
	struct Chromosome{
		Mat         features;
		double   score;
		int id;
		int lastVoteID;
	};
	struct GA{
		string                              type;
		vector <Chromosome>   chromosomes;
	};

	int id;

	vector<GA> geneticAlgorithms;

	void voteForObject      (vector<Mat> descriptorsObjects,
										   int gaNo,
										   int * votes);

	void findFirstInstance      ( vector<Mat> descriptorsObjects,
			                                      vector<Mat> singleObjects,
			                                      string inputString);

	int validateObject(vector<Mat> descriptorsObjects,
										int gaNo,
										int * votes,
										vector<Mat> singleObjects,
										int* newObjectIDArr);

	void scoreGivingGA(int GAno, int rightObject );


	void saveObjectToDatabase(string objectName,
													string GAname,
													Mat descriptor);

	Mat findObjectInDatabase(string idObject,
												string gaType);


	bool checkDirs(string objectName);

	void load();

	void save();

};

#endif /* LEARNINGCLASSIFIERSYSTEM_H_ */
