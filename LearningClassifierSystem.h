/*
 * LearningClassifierSystem.h
 *
 *  Created on: Oct 10, 2013
 *      Author: linuxthomz
 */

#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

#ifndef LEARNINGCLASSIFIERSYSTEM_H_
#define LEARNINGCLASSIFIERSYSTEM_H_

class LearningClassifierSystem {
public:
	// Constructor and destructor
	LearningClassifierSystem();
	virtual ~LearningClassifierSystem();

	void Test();
	void learn(Mat descriptorsDatabase, vector<Mat> descriptorsObjects, vector<Mat> singleObjects, string inputString);

private:
	struct Chromosome{
		vector<Mat> features;
		double score;
	};
	struct GA{
		string type;
		vector <Chromosome> chromosomes;
	};

	vector<GA> geneticAlgorithms;

	void findFirstInstance( vector<Mat> descriptorsObjects, vector<Mat> singleObjects, string inputString);
};

#endif /* LEARNINGCLASSIFIERSYSTEM_H_ */
