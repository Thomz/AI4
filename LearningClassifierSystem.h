/*
 * LearningClassifierSystem.h
 *
 *  Created on: Oct 10, 2013
 *      Author: linuxthomz
 */

#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <sys/types.h>
#include <sys/stat.h>
#include <dirent.h>
#include <fstream>
#include <istream>
#include <sys/time.h>


#define matchesThreshold 200
#define rightUpCnt 0.1
#define degFactor 0.8
#define upperThres 2
#define maxChromosomes 30
#define chrCutdown 20
#define startPic 30
#define evaluationPics 11
#define evaluation true
#define evaluationObject "candle"


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

	int evaluateAll(vector<Mat> descriptorsObjects,
							 vector<Mat> singleObjects,
							 string inputString,
							 int correctObject);


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
										   double * votes);

	void findFirstInstance      ( vector<Mat> descriptorsObjects,
			                                      vector<Mat> singleObjects,
			                                      string inputString);

	int validateObject(vector<Mat> descriptorsObjects,
										int gaNo,
										double * votes,
										vector<Mat> singleObjects,
										int* newObjectIDArr,
										bool eval);

	void scoreGivingGA( int GAno,
									 int rightObject,
									 int* newObjectIDArr,
									 vector<Mat> descriptorsObjects);


	void saveObjectToDatabase(string objectName,
													string GAname,
													Mat descriptor);

	Mat findObjectInDatabase(string idObject,
												string gaType);

	void updateGA(int GAno);

	bool checkDirs(string objectName);

	void load();

	void save();

};

#endif /* LEARNINGCLASSIFIERSYSTEM_H_ */
