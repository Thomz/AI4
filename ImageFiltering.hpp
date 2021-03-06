#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "queue"
#include <cv.h>
using namespace cv;
using namespace std;

#define GRIDWIDTH 3
#define GRIDHEIGHT 2


#include "LearningClassifierSystem.h"

using namespace cv;
using namespace std;

template <typename T>
  string NumberToString ( T Number )
  {
     ostringstream ss;
     ss << Number;
     return ss.str();
	//mfskf
  }

vector<Mat> filterSurrounding(Mat& input, Mat& output){
	Mat gray;
	cvtColor(input, gray, CV_RGB2GRAY);
	Mat_<float> floatImg = gray.clone();

	Mat_<uchar> smooth = floatImg.clone();
	Mat_<uchar> edges;
	Mat edges2;
	Canny(smooth,edges,50,200,3,1);
	//imshow("edges",edges);
	dilate(edges,edges,Mat(),Point(-1,-1),2);
	//imshow("dilated",edges);
	//erode(edges,edges,Mat(),Point(-1,-1),2);
	//imshow("eroded",edges);

	copyMakeBorder(edges,edges,2,2,2,2,BORDER_CONSTANT,0); // makes frame of black pixels

	Mat mask;
	Canny(edges, mask, 100, 200);
	copyMakeBorder(mask, mask, 1, 1, 1, 1, BORDER_REPLICATE);
	Mat_<uchar> temp;
	int paintedWhite = 0;
	for(int x=0; x<edges.cols;x+=5)		// fills all encapsulated areas with white
		for(int y=0; y<edges.rows;y+=5){
			temp = edges.clone();
			paintedWhite = floodFill(temp,mask,Point(x,y),255);
			if(paintedWhite*10 < 4*temp.cols*temp.rows){
				edges=temp;
			}
		}

	dilate(edges,edges,Mat(),Point(-1,-1),2);	// expand to close holes and make full bodies
	erode(edges,edges,Mat(),Point(-1,-1),8);	// erodes to small edges
	dilate(edges,edges,Mat(),Point(-1,-1),4);	// expands to maintain object size

	Mat blobs = edges.clone();
	//imshow("filtered", blobs);

	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	findContours( blobs, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE );

	/// Draw contours
	Mat drawing = Mat::zeros( edges.size(), CV_8UC3 );
	int idx = 0;
	vector<Mat> singleImgs;

	for( ; idx >= 0; idx = hierarchy[idx][0] )
	{
		Scalar color( 255, 255, 255 );
		drawContours( drawing, contours, idx, color, CV_FILLED, 8, hierarchy );
		dilate(drawing,drawing,Mat(),Point(-1,-1),5);
		if(contours[idx].size()>40)
			singleImgs.push_back(drawing.clone());
		drawing = Mat::zeros( edges.size(), CV_8UC3 );
	}


	return singleImgs;

}

void getOverlay(Mat& orig, Mat& filtered){
	for(int x=0; x<filtered.cols;x++)
		for(int y=0; y<filtered.rows;y++){
			if(filtered.at<Vec3b>(y,x)[0] == 255)
			{
				filtered.at<cv::Vec3b>(y,x)[0] = orig.at<cv::Vec3b>(y,x)[0];
				filtered.at<cv::Vec3b>(y,x)[1] = orig.at<cv::Vec3b>(y,x)[1];
				filtered.at<cv::Vec3b>(y,x)[2] = orig.at<cv::Vec3b>(y,x)[2];
			}
		}
}

Mat getImage(int i){
	string temp = evaluationObject;
	if(!evaluation)
		return imread("pics/evaluation/" + temp + "/" + NumberToString(i) + ".jpg", CV_LOAD_IMAGE_UNCHANGED);
	else
		return imread("pics/training/0" + NumberToString(i) + ".jpg", CV_LOAD_IMAGE_UNCHANGED);
}

// i is set to: 0=evaluation and 1=training
Mat getImageKohonen(int i, int task, string object){
	string temp = object;
	if(task==0){
		Mat img = imread("pics/evaluation/" + temp + "/" + NumberToString(i) + ".jpg", CV_LOAD_IMAGE_UNCHANGED);
		cvtColor(img, img, CV_BGR2HSV);
		// make brighter
		for(int y=0; y<img.rows; y++ ){
			for(int x=0; x<img.cols; x++){
				if(img.at<Vec3b>(y,x)[2]>5){
					int tempVal = img.at<Vec3b>(y,x)[2];
					tempVal+=100;
					if(tempVal>255){
						img.at<Vec3b>(y,x)[2]=255;
					}
					else{
						img.at<Vec3b>(y,x)[2]=tempVal;
					}
				}
			}
		}
		cvtColor(img, img, CV_HSV2BGR);
		return img;
	}
	else{
		Mat img = imread("pics/training/0" + NumberToString(i) + ".jpg", CV_LOAD_IMAGE_UNCHANGED);
		cvtColor(img, img, CV_BGR2HSV);
		// make brighter
		for(int y=0; y<img.rows; y++ ){
			for(int x=0; x<img.cols; x++){
				if(img.at<Vec3b>(y,x)[2]>5){
					int tempVal = img.at<Vec3b>(y,x)[2];
					tempVal+=0;
					if(tempVal>255){
						img.at<Vec3b>(y,x)[2]=255;
					}
					else{
						img.at<Vec3b>(y,x)[2]=tempVal;
					}
				}
			}
		}
		cvtColor(img, img, CV_HSV2BGR);
		return img;
	}
}

Mat getImageEvaluation(int i){
	string temp = evaluationObject;
	if(evaluation)
		return imread("pics/evaluation/" + temp + "/" + NumberToString(i) + ".jpg", CV_LOAD_IMAGE_UNCHANGED);
	else
		return imread("pics/training/0" + NumberToString(i) + ".jpg", CV_LOAD_IMAGE_UNCHANGED);
}

Mat getImageClassification(int i, string object){
	Mat img = imread("pics/classification/" + object +  ".jpg", CV_LOAD_IMAGE_UNCHANGED);
	cvtColor(img, img, CV_BGR2HSV);
	// make brighter
	for(int y=0; y<img.rows; y++ ){
		for(int x=0; x<img.cols; x++){
			if(img.at<Vec3b>(y,x)[2]>5){
				int tempVal = img.at<Vec3b>(y,x)[2];
				tempVal+=100;
				if(tempVal>255){
					img.at<Vec3b>(y,x)[2]=255;
				}
				else{
					img.at<Vec3b>(y,x)[2]=tempVal;
				}
			}
		}
	}
	cvtColor(img, img, CV_HSV2BGR);
	return img;
}

Mat getDescriptorsFromObject(Mat src, vector<KeyPoint> keypoints,	SiftDescriptorExtractor extractor){
	Mat descriptor;
	extractor.compute( src, keypoints, descriptor );
	return descriptor;
}

void sortDescriptors(Mat databaseDesc, vector<Mat>& objects, 	FlannBasedMatcher matcher,vector< Mat> & images){
	std::vector< DMatch > matches;

	vector<double> dists;
	double dist = 0;

	for(int i = 0; i < objects.size(); i++){
		matcher.match( databaseDesc, objects[i], matches );
		for(int j = 0; j < 15; j++){
			dist += matches[i].distance;
		}
		dist /= matches.size();
		dists.push_back(dist);
		dist = 0;
	}

	Mat tempDesc;
	double tempDist;
	Mat tempImg;

	for( int i = 0; i < objects.size(); i++){
		for(int j = 0; j < objects.size(); j++){
			if(dists[i] < dists[j]){
				tempDist = dists[i]; tempDesc=objects[i].clone(); tempImg = images[i].clone();
				dists[i] = dists[j]; objects[i] = objects[j].clone(); images[i] = images[j].clone();
				dists[j] = tempDist; objects[j] = tempDesc.clone(); images[j] = tempImg.clone();
			}
		}
	}
}

vector<KeyPoint> getKeypointsFromObject(Mat src){
	SiftFeatureDetector detector;
	// Find keypoints in the picture
	vector<KeyPoint> keypoints;
	detector.detect(src, keypoints);
	// Draw keypoints on new pic
	///drawKeypoints(src, keypoints, output);

	return keypoints;
}

void rotate(Mat& src, double angle, Mat& dst)
{
    int len = max(src.cols, src.rows);
    Point2f pt(len/2., len/2.);
    Mat r = getRotationMatrix2D(pt, angle, 1.0);

    warpAffine(src, dst, r, Size(len, len));
}

vector<double> getCustomObjectDescriptor(Mat input){
	copyMakeBorder(input,input,200,200,200,200,BORDER_CONSTANT, Scalar(0,0,0));
	Mat gray;
	vector<double> avgs;
	cvtColor(input, gray, CV_BGR2GRAY);
	Mat_<float> floatImg = gray.clone();

	Mat_<uchar> smooth = floatImg.clone();
	Mat_<uchar> edges;
	Canny(smooth,edges,50,200,3,1);
	dilate(edges,edges,Mat(),Point(-1,-1),2);


	copyMakeBorder(edges,edges,2,2,2,2,BORDER_CONSTANT,0); // makes frame of black pixels

	Mat mask;
	Canny(edges, mask, 100, 200);
	copyMakeBorder(mask, mask, 1, 1, 1, 1, BORDER_REPLICATE);
	Mat_<uchar> temp;
	int paintedWhite = 0;
	for(int x=0; x<edges.cols;x+=5)		// fills all encapsulated areas with white
		for(int y=0; y<edges.rows;y+=5){
			temp = edges.clone();
			paintedWhite = floodFill(temp,mask,Point(x,y),255);
			if(paintedWhite*10 < 4*temp.cols*temp.rows){
				edges=temp;
			}
		}

	dilate(edges,edges,Mat(),Point(-1,-1),2);	// expand to close holes and make full bodies
	erode(edges,edges,Mat(),Point(-1,-1),8);	// erodes to small edges
	dilate(edges,edges,Mat(),Point(-1,-1),4);	// expands to maintain object size
	Mat rotated = edges.clone();


	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	findContours( edges, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE );


	RotatedRect rectangle1;
	rectangle1 = minAreaRect(Mat(contours[0]));

	rotate(rotated, rectangle1.angle, rotated);
	Mat rotatedInput;
	rotate(input,rectangle1.angle, rotatedInput);

	vector<vector<Point> > contours1;
	vector<Vec4i> hierarchy1;
	findContours( rotated, contours1, hierarchy1, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE );

	Rect rect;
	rect = boundingRect(contours1[0]);

	Mat ROI = rotatedInput(rect);
	Mat hsv;
	cvtColor(ROI, hsv, CV_BGR2HSV);
	// make brighter
	/*for(int y=0; y<hsv.rows; y++ ){
		for(int x=0; x<hsv.cols; x++){
			if(hsv.at<Vec3b>(y,x)[2]>5){
				int tempVal = hsv.at<Vec3b>(y,x)[2];
				tempVal+=100;
				if(tempVal>255){
					hsv.at<Vec3b>(y,x)[2]=255;
				}
				else{
					hsv.at<Vec3b>(y,x)[2]=tempVal;
				}
			}
		}
	}*/
	Mat bgr;
	cvtColor(hsv, bgr, CV_HSV2BGR);

	for(int k=0; k<GRIDWIDTH; k++){
		for(int l=0; l<GRIDHEIGHT; l++){
			double currentSumR=0;
			double currentSumG=0;
			double currentSumB=0;
			int pixels= hsv.rows/GRIDHEIGHT * (hsv.cols/GRIDWIDTH);

			for(int y=0+(l*(bgr.rows/GRIDHEIGHT)); y<bgr.rows/GRIDHEIGHT+(l*(bgr.rows/GRIDHEIGHT)); y++ ){
				for(int x=0+(k*(bgr.cols/GRIDWIDTH)); x<bgr.cols/GRIDWIDTH+(k*(bgr.cols/GRIDWIDTH)); x++){
					if(hsv.at<Vec3b>(y,x)[2]>5){
						currentSumB+=bgr.at<Vec3b>(y,x)[0];
						currentSumG+=bgr.at<Vec3b>(y,x)[1];
						currentSumR+=bgr.at<Vec3b>(y,x)[2];
					}
					else{
						pixels--;
					}
				}
			}
			avgs.push_back((currentSumB/pixels)/255);
			avgs.push_back((currentSumG/pixels)/255);
			avgs.push_back((currentSumR/pixels)/255);
		}
	}


	return avgs;
}







