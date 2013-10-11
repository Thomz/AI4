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
	};
	struct GA{
		string                              type;
		vector <Chromosome>   chromosomes;
	};

	vector<GA> geneticAlgorithms;

	void voteForObject      (vector<Mat> descriptorsObjects,
										   int gaNo,
										   int * votes);

	void findFirstInstance      ( vector<Mat> descriptorsObjects,
			                                      vector<Mat> singleObjects,
			                                      string inputString);

	void validateObject(vector<Mat> descriptorsObjects,
										int gaNo,
										int * votes,
										vector<Mat> singleObjects);

	void saveObjectToDatabase(string objectName,
													string GAname,
													Mat descriptor);

	bool checkDirs(string objectName);

	void load();

	void save();

};

#endif /* LEARNINGCLASSIFIERSYSTEM_H_ */
