/*
 * KohonenNetwork.cpp
 *
 *  Created on: Nov 13, 2013
 *      Author: linuxthomz
 */

#include "KohonenNetwork.h"

KohonenNetwork::KohonenNetwork(int sizeMap, int sizeWeights) {
	mapSize  = sizeMap;
	weightSize = sizeWeights;
	knnMapRadius = sizeMap/2;
	timeConstant = totalIterations/log(knnMapRadius);
	iterationCounter = 0;

    knnMap.resize(sizeMap);
    for (int i = 0; i < sizeMap; i++)
        knnMap[i].resize(sizeMap);

    BMUcount.resize(sizeMap);
    for (int i = 0; i < sizeMap; i++)
    	BMUcount[i].resize(sizeMap);


    srand (time(NULL));

	for(int i = 0; i < knnMap.size(); i++){
		for(int j = 0; j < knnMap.size(); j++){
			for(int k = 0; k < sizeWeights; k++){
				knnMap[i][j].weights.push_back(((double) rand() / (RAND_MAX)));
				BMUcount[i][j] = 0;
			}
		}
	}

	cout << "Kohonen network created" << endl;
}

KohonenNetwork::~KohonenNetwork() {
	// TODO Auto-generated destructor stub
}

void KohonenNetwork::printNetwork(){
	for(int i = 0; i < knnMap.size(); i++){
		for(int j = 0; j < knnMap.size(); j++){
			cout << "[" <<  knnMap[i][j].weights.size() << "]";
		}
		cout << endl;
	}
}

void KohonenNetwork::printBMUcount(){
	for(int i = 0; i < BMUcount.size(); i++){
		for(int j = 0; j < BMUcount.size(); j++){
			cout << "[" <<  BMUcount[i][j]<< "]";
		}
		cout << endl;
	}
}

pointStr KohonenNetwork::findBestMatch(vector<double> inputVector){
	double distance, bestDistance(999999), bestI, bestJ;

	for(int i = 0; i < knnMap.size(); i++){
		for(int j = 0; j < knnMap.size(); j++){

			for(int k = 0; k < weightSize; k++)
				distance += pow(inputVector[k] - knnMap[i][j].weights[k],2);

			distance = sqrt(distance);

			if(distance < bestDistance){
				bestDistance = distance;
				bestI = i;
				bestJ = j;
			}
			distance = 0;
		}
	}

	pointStr ret = {bestI, bestJ};

	return ret;
}

void KohonenNetwork::adjustWeights(vector<double> inputVector){
	pointStr BMR = findBestMatch(inputVector);

	BMUcount[BMR.x][BMR.y]++;

	iterationCounter++;

	if(iterationCounter == 1){
		firstBMU = BMR;
	}

	/*
	for(int i = 0; i < weightSize; i++)
		cout << "First BMU : " << knnMap[firstBMU.x][firstBMU.y].weights[i] << endl;
*/

	double neighbourhoodRadius = knnMapRadius * exp(-iterationCounter/timeConstant);
	double influence = 0;
	double learningRate = 0;

	for(int i = 0; i < knnMap.size(); i++){
		for(int j = 0; j < knnMap.size(); j++){

			double distFromBMR = sqrt(pow(BMR.x - i,2) + pow(BMR.y - j,2));

			if(distFromBMR < neighbourhoodRadius){
				influence = exp( -(pow(distFromBMR,2) / (2*neighbourhoodRadius  )  ));
				learningRate = startLearningRate * exp(-(double)iterationCounter/totalIterations);
				for(int k = 0; k < weightSize; k++){
					knnMap[i][j].weights[k] = knnMap[i][j].weights[k] + influence * learningRate * (inputVector[k] - knnMap[i][j].weights[k]);
					if(knnMap[i][j].weights[k] < 0.0001 )
						knnMap[i][j].weights[k] = 0;
				}
			}
		}
	}

}

void KohonenNetwork::showAsImage(string windowName){

	Mat image = Mat::ones(Size(mapSize,mapSize), CV_8UC3);
	Mat bmuIMG = Mat::ones(Size(mapSize,mapSize), CV_8UC3);

	for(int i = 0; i < knnMap.size(); i++){
		for(int j = 0; j < knnMap.size(); j++){
			//double colors[] = { knnMap[i][j].weights[0], knnMap[i][j].weights[1], knnMap[i][j].weights[2] };
			image.at<cv::Vec3b>(j,i)[0] = knnMap[i][j].weights[0] * 255.0;
			image.at<cv::Vec3b>(j,i)[1] = knnMap[i][j].weights[1] * 255.0;
			image.at<cv::Vec3b>(j,i)[2] = knnMap[i][j].weights[2] * 255.0;
			if(showBMUimage){
				if(BMUcount[i][j] > totalIterations/20)
					circle(bmuIMG, Point(i,j), 2, Scalar(image.at<cv::Vec3b>(j,i)[0],image.at<cv::Vec3b>(j,i)[1] ,image.at<cv::Vec3b>(j,i)[2]),2);
			}
		}
	}

	imshow("BMU Image",bmuIMG);



	imshow(windowName, image);
}
