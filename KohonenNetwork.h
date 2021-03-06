/*
 * KohonenNetwork.h
 *
 *  Created on: Nov 13, 2013
 *      Author: linuxthomz
 */

#include <iostream>
#include <vector>
#include <stdlib.h>
#include <math.h>
#include <iomanip>
#include "opencv2/opencv.hpp"
#include <fstream>

#define totalIterations 131
#define startLearningRate 1
#define showBMUimage true
#define classificationPics 14
#define maxThreshold 1.
#define minThreshold 0.
#define startPicKoh 1


using namespace std;
using namespace cv;

#ifndef KOHONENNETWORK_H_
#define KOHONENNETWORK_H_

struct pointStr{
	int x,  y;
};

class KohonenNetwork {
public:
	struct node{
		vector<double> weights;
		string object;
		Point point;
	};

	string classObjects[classificationPics];

	vector<vector<node> > knnMap;
	vector<vector<double> > BMUcount;
	vector<node> BMUs;

	KohonenNetwork(int size,  int sizeWeights);
	virtual ~KohonenNetwork();
	void printNetwork();
	void printBMUcount();

	pointStr findBestMatch(vector<double> inputVector);
	void adjustWeights(vector<double> inputVector);
	void showAsImage(string windowName);
	void getBMUs();
	void classifyBMU(vector<double> inputWieght, string objectName);
	void loadMap();
	void loadBmuMap();
	void load();
	void saveMap();
	void saveBMUs();
	int getObject(vector<vector<double> >descriptor, string object);
	void saveClassifiers();
	void loadClassifiers();
	void printBMUObjects();
	void showAmplifiedImage(string windowName, int amplification,  bool showObjects);

private:
	int mapSize;
	int weightSize;
	double knnMapRadius;
	double timeConstant;
	double iterationCounter;
	pointStr firstBMU;
};

#endif /* KOHONENNETWORK_H_ */
