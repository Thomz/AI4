/*
 * FeatureSelector.h
 *
 *  Created on: Oct 10, 2013
 *      Author: Simon
 */

#ifndef FEATURESELECTOR_H_
#define FEATURESELECTOR_H_

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/nonfree.hpp>
//#include "ImageFiltering.hpp"

using namespace std;
using namespace cv;

class FeatureSelector {
public:
	FeatureSelector();
	virtual ~FeatureSelector();
	void test();
	Mat getObjectFeatures(string);
	Mat getObjectWeights(string);
	void updateWeights(string, Mat);
	void saveObjectFeatures(string, Mat);
	void saveFeatureWeights(string, Mat);

};

#endif /* FEATURESELECTOR_H_ */
