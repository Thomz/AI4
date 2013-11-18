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

#define totalIterations 35
#define startLearningRate 0.1
#define showBMUimage true


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
	};

	vector<vector<node> > knnMap;
	vector<vector<double> > BMUcount;

	KohonenNetwork(int size,  int sizeWeights);
	virtual ~KohonenNetwork();
	void printNetwork();
	void printBMUcount();

	pointStr findBestMatch(vector<double> inputVector);
	void adjustWeights(vector<double> inputVector);
	void showAsImage(string windowName);





private:
	int mapSize;
	int weightSize;
	double knnMapRadius;
	double timeConstant;
	double iterationCounter;
	pointStr firstBMU;
};

#endif /* KOHONENNETWORK_H_ */
