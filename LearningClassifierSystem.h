/*
 * LearningClassifierSystem.h
 *
 *  Created on: Oct 10, 2013
 *      Author: linuxthomz
 */

#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

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

	int* voteForObject          (vector<Mat> descriptorsObjects,
											  int gaNo);

	void findFirstInstance      ( vector<Mat> descriptorsObjects,
			                                      vector<Mat> singleObjects,
			                                      string inputString);
};

#endif /* LEARNINGCLASSIFIERSYSTEM_H_ */
