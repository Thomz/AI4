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

using namespace std;
using namespace cv;

class FeatureSelector {
public:
	FeatureSelector();
	virtual ~FeatureSelector();
	void test();
	Mat getObjectFeatures(string& inputObject);
	Mat getObjectWeights(string& inputObject);
	void updateWeights(string& objectName, Mat& currentFeatures);
	void filterFeaturesAndWeights(Mat& features, Mat& weights);
	void saveObjectFeatures(string& objectName, Mat& currentFeatures);
	void saveFeatureWeights(string& objectName, Mat& weights);
	Mat findCorrectObject( string& inputString, vector<Mat>& descriptorsObjects, vector<Mat>& singleObjects, Mat nextImage);

};

#endif /* FEATURESELECTOR_H_ */
