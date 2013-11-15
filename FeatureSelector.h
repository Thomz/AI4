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
	vector<int> sortDoubleVector(vector<int> indexVector, vector<int> valueVector);
	int compareFeatures(Mat& featuresImg1, Mat& featuresImg2);
	Mat findCorrectObject( string& inputString, vector<Mat>& descriptorsObjects, vector<Mat>& singleObjects, Mat nextImage);
	Mat findCorrectObjectSimple( string& inputString, vector<Mat>& descriptorsObjects, vector<Mat>& singleObjects, Mat nextImage);
	bool evaluateImg( string& inputString, vector<Mat>& descriptorsObjects, vector<Mat>& singleObjects, int correctImg);


};

#endif /* FEATURESELECTOR_H_ */
