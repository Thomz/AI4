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

	string classificationObjects[] = {"bicycletube", "cacao", "candle", "cillitbang", "cola", "controller", "curryketchup", "jaeger", "kleenex","lacoste","powerball","vestfyen"};

	for(int i = 0; i < classificationPics; i++)
		classObjects[i] = classificationObjects[i];

    knnMap.resize(sizeMap);
    for (int i = 0; i < sizeMap; i++)
        knnMap[i].resize(sizeMap);

    BMUcount.resize(sizeMap);
    for (int i = 0; i < sizeMap; i++)
    	BMUcount[i].resize(sizeMap);


    srand (time(NULL));

    double diff = maxThreshold - minThreshold;

	for(int i = 0; i < knnMap.size(); i++){
		for(int j = 0; j < knnMap.size(); j++){
			for(int k = 0; k < sizeWeights; k++){
				knnMap[i][j].weights.push_back( (((double) rand() / RAND_MAX) * diff) + minThreshold);
				knnMap[i][j].point = Point(i,j);
				knnMap[i][j].object = "";
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
			cout << "[" <<  knnMap[i][j].weights[0] << "]";
		}
		cout << endl;
	}
}

void KohonenNetwork::printBMUcount(){
	for(int i = 0; i < BMUcount.size(); i++){
		for(int j = 0; j < BMUcount.size(); j++){
			cout << "[" << knnMap[j][i].object << "]";
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
			if(showBMUimage && iterationCounter>1){
				if(BMUcount[i][j] > 0)
					circle(bmuIMG, Point(i,j), 2, Scalar(255,255,255),2);
					//circle(bmuIMG, Point(i,j), 2, Scalar(image.at<cv::Vec3b>(j,i)[0],image.at<cv::Vec3b>(j,i)[1] ,image.at<cv::Vec3b>(j,i)[2]),2);
			}
		}
	}

	imshow("BMU Image",bmuIMG);



	imshow(windowName, image);
}

void KohonenNetwork::classifyBMU(vector<double> inputWieght, string objectName){

	cout << "Classifying BMU" << endl;

	double bestDistance(INFINITY), distance(0);
	int bestBMU = 0;

	for(int i = 0; i < BMUs.size(); i++){

		for(int j = 0; j < weightSize; j++)
			distance += pow(BMUs[i].weights[j] - inputWieght[j],2);

		distance = sqrt(distance);

		if(distance < bestDistance){
			bestDistance = distance;
			bestBMU = i;
			cout << "BestBMU: "  << bestBMU << endl;
		}

		distance = 0;
	}

	knnMap[BMUs[bestBMU].point.x][BMUs[bestBMU].point.y].object = objectName;

	Mat bmuIMG = Mat::ones(Size(mapSize,mapSize), CV_8UC3);
	circle(bmuIMG, Point(BMUs[bestBMU].point.x,BMUs[bestBMU].point.y), 2, Scalar(255,255,255),2);
	imshow("Candle", bmuIMG);

}

void KohonenNetwork::getBMUs(){
	for(int i = 0; i < knnMap.size(); i++)
		for(int j = 0; j < knnMap.size(); j++)
			if(BMUcount[i][j] > totalIterations/35)
				BMUs.push_back(knnMap[i][j]);

}

void KohonenNetwork::loadMap(){
	ifstream myReadFile;
	myReadFile.open("knn.txt");

	string temp;
	double tempD;

	myReadFile >> temp;
	mapSize = atoi(temp.c_str());

	myReadFile >> temp;
	weightSize = atoi(temp.c_str());

	 knnMap.resize(mapSize);
	 for (int i = 0; i < mapSize; i++)
		 knnMap[i].resize(mapSize);

	for(int i = 0; i < mapSize; i++){
		for(int j = 0; j <mapSize; j++){
			cout << i << " - " << j << endl;
			knnMap[i][j].weights.resize(weightSize);
			for(int k = 0; k < weightSize; k++){
				myReadFile >> temp;
				knnMap[i][j].weights[k] = atof(temp.c_str());
				knnMap[i][j].point = Point(i,j);
				knnMap[i][j].object = "";
			}
		}
	}

	iterationCounter = 2;

	myReadFile.close();

	printNetwork();

}

void KohonenNetwork::load(){
	loadMap();
	loadBmuMap();
}

void KohonenNetwork::loadBmuMap(){
	ifstream myReadFile;
	myReadFile.open("bmus.txt");

	string temp;
	int bmuMapSize;

	myReadFile >> temp;
	bmuMapSize = atoi(temp.c_str());

	 BMUcount.resize(bmuMapSize);
	 for (int i = 0; i < bmuMapSize; i++)
		 BMUcount[i].resize(bmuMapSize);

	for(int i = 0; i < bmuMapSize; i++){
		for(int j = 0; j <bmuMapSize; j++){
			myReadFile >> temp;
			BMUcount[i][j] =atoi(temp.c_str());
		}
	}

	myReadFile.close();

	printNetwork();

}
void KohonenNetwork::saveMap(){
	ofstream myfile;
	myfile.open ("map.txt");
	myfile << mapSize << " " << weightSize << endl;

	for(int i=0; i<mapSize; i++){
		for(int j=0; j<mapSize; j++){
			for(int k=0; k<weightSize; k++){
				myfile << knnMap[i][j].weights[k];
				myfile << " ";
			}
			myfile << endl;
		}
	}
	myfile.close();
}

void KohonenNetwork::saveBMUs(){
	ofstream myfile;
	myfile.open ("bmus.txt");
	myfile << mapSize << endl;

	for(int i=0; i<mapSize; i++){
		for(int j=0; j<mapSize; j++){
				myfile << BMUcount[i][j];
				myfile << " ";
		}
		myfile << endl;
	}
	myfile.close();
}


