#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/dnn.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include "AvgBackground.h"
#include "cv_yuv_codebook.h"
#include "datastructure.h"
#include "cvblob.h"
#include <string>
#include <iostream>
#include <cstdlib>
#include <ctype.h>
#include <string.h>
#include "opencv2/opencv.hpp"
#include "BackgroundSubtract.h"
#include "Detector.h"
#include <opencv2/highgui/highgui_c.h>
#include "Ctracker.h"
#include "doorDet.h"
using namespace cv;
using namespace cv::dnn;
using namespace cvb;
using namespace std;
#define TIMEOUT_SEC(buflen,baud) (buflen*20/baud+2)  //接收超时
#define TIMEOUT_USEC 0
//VARIABLES for CODEBOOK METHOD:
codeBook *cB;   //This will be our linear model of the image, a vector 
				//of lengh = height*width
int maxMod[CHANNELS];	//Add these (possibly negative) number onto max 
						// level when code_element determining if new pixel is foreground
int minMod[CHANNELS]; 	//Subract these (possible negative) number from min 
						//level code_element when determining if pixel is foreground
unsigned cbBounds[CHANNELS]; //Code Book bounds for learning
bool ch[CHANNELS];		//This sets what channels should be adjusted for background bounds
int nChannels = CHANNELS;
int imageLen = 0;
uchar *pColor; //YUV pointer
/*parameters for erosion and dilation*/
Mat src, erosion_dst, dilation_dst;

int erosion_elem = 0;
int erosion_size = 0;  //originally 0
int dilation_elem = 0;
int dilation_size = 15; //originally 15
int const max_elem = 2;
int const max_kernel_size = 21;

int in = 0;
int out = 0;
const char* In;
const char* Out;
/** @function People Counter*/

/** @function Erosion */
void Erosion( IplImage* src, IplImage* output )
{
  int erosion_type;
  if( erosion_elem == 0 ){ erosion_type = MORPH_RECT; }
  else if( erosion_elem == 1 ){ erosion_type = MORPH_CROSS; }
  else if( erosion_elem == 2) { erosion_type = MORPH_ELLIPSE; }

  Mat element = getStructuringElement( erosion_type,
                                       Size( 2*erosion_size + 1, 2*erosion_size+1 ),
                                       Point( erosion_size, erosion_size ) );

  /// Apply the erosion operation
  cv::Mat m0 = cv::cvarrToMat(src);
  cv::Mat m1 = cv::cvarrToMat(output);
  erode( m0, m1, element );
  //imshow( "Erosion Demo", erosion_dst );
}

/** @function Dilation */
void Dilation( IplImage* src, IplImage* output )
{
  int dilation_type;
  if( dilation_elem == 0 ){ dilation_type = MORPH_RECT; }
  else if( dilation_elem == 1 ){ dilation_type = MORPH_CROSS; }
  else if( dilation_elem == 2) { dilation_type = MORPH_ELLIPSE; }

  Mat element = getStructuringElement( dilation_type,
                                       Size( 2*dilation_size + 1, 2*dilation_size+1 ),
                                       Point( dilation_size, dilation_size ) );
  /// Apply the dilation operation
  cv::Mat m0 = cv::cvarrToMat(src);
  cv::Mat m1 = cv::cvarrToMat(output);

  dilate( m0, m1, element );
  //imshow( "Dilation Demo", dilation_dst );
}
/*****************************************/


void help() {
	printf("\nLearn background and find foreground using simple average and average difference learning method:\n"
		"\nUSAGE:\n  ch9_background startFrameCollection# endFrameCollection# [movie filename, else from camera]\n"
		"If from AVI, then optionally add HighAvg, LowAvg, HighCB_Y LowCB_Y HighCB_U LowCB_U HighCB_V LowCB_V\n\n"
		"***Keep the focus on the video windows, NOT the consol***\n\n"
		"INTERACTIVE PARAMETERS:\n"
		"\tESC,q,Q  - quit the program\n"
		"\th	- print this help\n"
		"\tp	- pause toggle\n"
		"\ts	- single step\n"
		"\tr	- run mode (single step off)\n"
		"=== AVG PARAMS ===\n"
		"\t-    - bump high threshold UP by 0.25\n"
		"\t=    - bump high threshold DOWN by 0.25\n"
		"\t[    - bump low threshold UP by 0.25\n"
		"\t]    - bump low threshold DOWN by 0.25\n"
		"=== CODEBOOK PARAMS ===\n"
		"\ty,u,v- only adjust channel 0(y) or 1(u) or 2(v) respectively\n"
		"\ta	- adjust all 3 channels at once\n"
		"\tb	- adjust both 2 and 3 at once\n"
		"\ti,o	- bump upper threshold up,down by 1\n"
		"\tk,l	- bump lower threshold up,down by 1\n"
		);
}

// 0 left-right   1 up-down
int DETECT_DIRECTION = 1;

#define SHOW_TRAJTORY 0
#define SHOW_FOREGROUND 0
#define SHOW_CONTOUR 0

#define MOG_BACKGROUND 0

#define OPTICAL_FLOW 0

using namespace cv;
using namespace std;


void drawOptFlowMap(const Mat& flow, Mat& cflowmap, int step, const Scalar& color) {

		for (int y = 0; y < cflowmap.rows; y += step)

			for (int x = 0; x < cflowmap.cols; x += step)
			{
					const Point2f& fxy = flow.at< Point2f>(y, x);

					line(cflowmap, Point(x, y), Point(cvRound(x + fxy.x), cvRound(y + fxy.y)), color);

					circle(cflowmap, Point(cvRound(x + fxy.x), cvRound(y + fxy.y)), 1, color, -1);
			}
}


//
//USAGE:  background startFrameCollection# endFrameCollection# [movie filename, else from camera]
//If from AVI, then optionally add HighAvg, LowAvg, HighCB_Y LowCB_Y HighCB_U LowCB_U HighCB_V LowCB_V
//
int main(int argc, char** argv)
{
    CTracker tracker(0.2,0.5,60.0,10,10);  //for Kalman filter analysis
    Scalar Colors[]={Scalar(255,0,0),Scalar(0,255,0),Scalar(0,0,255),Scalar(255,255,0),Scalar(0,255,255),Scalar(255,0,255),Scalar(255,127,255),Scalar(127,0,255),Scalar(127,0,127)};
    IplImage* rawImage = 0, *yuvImage = 0; //yuvImage is for codebook method
    IplImage *ImaskAVG = 0,*ImaskAVGCC = 0;
    IplImage *ImaskCodeBook = 0,*ImaskCodeBookCC = 0, *kfdisp =0;
	IplImage *labelImg;
	Ptr<BackgroundSubtractor> pMOG2; //MOG2 Background subtractor
	Mat fgMaskMOG2; //fg mask fg mask generated by MOG2 method
    Mat display;
	Mat prvs, flow;
	Mat imagRGB;
	Mat im_gray;

	std::cerr << "argc" << argc;
	//! [Create the importer of TensorFlow model]
	Ptr<dnn::Importer> importer_arrow;
	try                                     //Try to import TensorFlow AlexNet model
	{
		importer_arrow = dnn::createTensorflowImporter("frozen_model_arrow.pb"); // frozen_model_arrow.pb");
	}
	catch (const cv::Exception &err)        //Importer can throw errors, we will catch them
	{
		std::cerr << err.msg << std::endl;
	}

	if (!importer_arrow)
	{
		std::cerr << "Can't load network by using the mode file: " << std::endl;
		std::cerr << "frozen_model_arrow.pb" << std::endl;
		exit(-1);
	}

	Ptr<dnn::Importer> importer_floor;
	try                                     //Try to import TensorFlow AlexNet model
	{
		importer_floor = dnn::createTensorflowImporter("frozen_model_floor.pb");
	}
	catch (const cv::Exception &err)        //Importer can throw errors, we will catch them
	{
		std::cerr << err.msg << std::endl;
	}

	if (!importer_floor)
	{
		std::cerr << "Can't load network by using the mode file: " << std::endl;
		std::cerr << "frozen_model_floor.pb" << std::endl;
		exit(-1);
	}


	String inBlobName = "data_node";
	String outBlobName = "Softmax_1";

	//! [Initialize network]
	dnn::Net net_arrow;
	importer_arrow->populateNet(net_arrow);
	importer_arrow.release();                     //We don't need importer anymore
	for (String l : net_arrow.getLayerNames())
	{
		cout << l << " ";
	}

	dnn::Net net_floor;
	importer_floor->populateNet(net_floor);
	importer_floor.release();                     //We don't need importer anymore

	//CascadeClassifier hs_cascade;
	//CascadeClassifier upper_cascade;
	//CascadeClassifier down_cascade;
	//CascadeClassifier full_cascade;
	//CascadeClassifier head_cascade;

	vector<Rect> detected_hs;
	Rect door_prev;
	Rect door_current;
	Rect door;
	door_prev.width = 100;
	door_prev.height = 200;
	door_current.width = 100;
	door_current.height = 200;
	door.width = 100;
	door.height = 200;

	int door_constant = 0;
	int door_found = 0;
	int door_status = 0;

	Rect roi_arrow(213, 1, 14, 12);
	Rect roi_floor(213, 12, 14, 13);

	//Mat image_noarrow = imread("noarrow.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	//imshow("image_noarrow", image_noarrow);

	//Mat image_arrowdown = imread("arrowdown.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	//imshow("image_arrowdown", image_arrowdown);

	//Mat image_arrowup = imread("arrowup.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	//imshow("image_arrowup", image_arrowup);

	//Mat image_floor[25];

	//for (int i = 0; i < 25; i++) {
	//	char fn[64];
	//	sprintf(fn, "%d.jpg\0", i);
	//	image_floor[i] = imread(fn, CV_LOAD_IMAGE_GRAYSCALE);
	//}

	int elevator_status = 0;
	int elevator_status_record[32] = { 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 };
	int elevator_status_wptr = 0;

	int elevator_floor = 0;
	int elevator_floor_backup = 0;
	int elevator_record[32] = { 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 };
	int elevator_record_wptr = 0;


	//vector<Rect> detected_upper;
	//vector<Rect> detected_down;
	//vector<Rect> detected_full;
	//vector<Rect> detected_head;

	//if (!hs_cascade.load("HS.xml")) {
	//	cout << "Could not load the cascade of face detection" << std::endl;
	//	return -1;
	//}
	//if (!upper_cascade.load("haarcascade_upperbody.xml")) {
	//	cout << "Could not load the cascade of face detection" << std::endl;
	//	return -1;
	//}
	//if (!down_cascade.load("haarcascade_lowerbody.xml")) {
	//	cout << "Could not load the cascade of face detection" << std::endl;
	//	return -1;
	//}
	//if (!full_cascade.load("haarcascade_fullbody.xml")) {
	//	cout << "Could not load the cascade of face detection" << std::endl;
	//	return -1;
	//}

	//if (!head_cascade.load("cascadeH5.xml")) {
	//	cout << "Could not load the cascade of head detection" << std::endl;
	//	return -1;
	//}

    cv::Mat output;
	VideoCapture capture(0);
	
    CvTracks tracks;
	int startcapture = 1;
	int endcapture = 20;
    int c,n;
    CvFont font;
    cvInitFont(&font, CV_FONT_HERSHEY_COMPLEX, 0.5, 0.5, 1, 2, 8);
	Mat element = getStructuringElement(MORPH_RECT, Size(7, 7), Point(3, 3));
    vector<TrackPoint> prePoint, curPoint; //previous Point and Current Point
    int middle; //half of the height of the video
    //std::vector < std::vector<cv::Point2i > > blobs;
    //sprintf(in, "%d", In);
    vector<Point2d> centers; //for Kalman filter

    /** **********************************************8  */
    maxMod[0] = 74;  //Set color thresholds to default values
    minMod[0] = 32;
    maxMod[1] = 72;
    minMod[1] = 23;
    maxMod[2] = 72;
    minMod[2] = 23;
	float scalehigh = HIGH_SCALE_NUM;
	float scalelow = LOW_SCALE_NUM;
	
	

	if(argc == 3){
		printf("Capture from Camera\n");
		DETECT_DIRECTION = 0;
				
	}
	else {
		printf("Capture from file %s\n",argv[3]);
		bool readsuccess = capture.open(argv[3]);
		if(!readsuccess) { printf("Couldn't open %s\n",argv[3]); return -1;}

	}
	if(isdigit(argv[1][0])) { //Start from of background capture
		startcapture = atoi(argv[1]);
		printf("startcapture = %d\n",startcapture);
	}
	if(isdigit(argv[2][0])) { //End frame of background capture
		endcapture = atoi(argv[2]);
        printf("endcapture = %d\n", endcapture);
	}
	if(argc > 4){ //See if parameters are set from command line
		//FOR AVG MODEL
		if(argc >= 5){
			if(isdigit(argv[4][0])){
				scalehigh = (float)atoi(argv[4]);
			}
		}
		if(argc >= 6){
			if(isdigit(argv[5][0])){
				scalelow = (float)atoi(argv[5]);
			}
		}
		//FOR CODEBOOK MODEL, CHANNEL 0
		if(argc >= 7){
			if(isdigit(argv[6][0])){
				maxMod[0] = atoi(argv[6]);
			}
		}
		if(argc >= 8){
			if(isdigit(argv[7][0])){
				minMod[0] = atoi(argv[7]);
			}
		}
		//Channel 1
		if(argc >= 9){
			if(isdigit(argv[8][0])){
				maxMod[1] = atoi(argv[8]);
			}
		}
		if(argc >= 10){
			if(isdigit(argv[9][0])){
				minMod[1] = atoi(argv[9]);
			}
		}
		//Channel 2
		if(argc >= 11){
			if(isdigit(argv[10][0])){
				maxMod[2] = atoi(argv[10]);
			}
		}
		if(argc >= 12){
			if(isdigit(argv[11][0])){
				minMod[2] = atoi(argv[11]);
			}
		}
	}


	//MAIN PROCESSING LOOP:
	bool pause = false;
	bool singlestep = false;

	pMOG2 = createBackgroundSubtractorMOG2(); //MOG2 approach

	if (capture.isOpened())
	{
        cvNamedWindow( "Video", 1 );
#if SHOW_FOREGROUND
        cvNamedWindow( "ForegroundCodeBook",1);
#endif
#if SHOW_CONTOUR
		cvNamedWindow( "CodeBook_ConnectComp",1);
#endif

#if SHOW_TRAJTORY
        namedWindow("Trajetory_Analysis");
#endif
        int i = -1;
        /*add trackbar for tuning erosion and dilation*/
        createTrackbar( "EElement:\n 0: Rect \n 1: Cross \n 2: Ellipse", "Video",
                          &erosion_elem, max_elem,
                          0 );


          /// Create Dilation Trackbar
        createTrackbar( "DElement:\n 0: Rect \n 1: Cross \n 2: Ellipse", "CodeBook_ConnectComp",
                          &dilation_elem, max_elem,
                          0 );


        /************************************************************/
        for(;;)
        {
    			if(!pause){

					//rawImage = cvQueryFrame(capture);
					bool readsuccess = capture.read(imagRGB);

					//char filename[64];
					//sprintf(filename, "jpg\%d.jpg\0", i+1);
					//imwrite(filename, imagRGB);
					//i++;
					//printf("%d frame\n", i);
					//continue;

					if (!readsuccess)
						break;
					cvtColor(imagRGB, im_gray, CV_BGR2GRAY);
					rawImage = cvCloneImage(&(IplImage)imagRGB);

					// detect arrow
					cv::Mat croppedArrowImage = im_gray(roi_arrow);
					cv::Mat croppedArrowImage_f;
					resize(croppedArrowImage, croppedArrowImage, cv::Size(28, 28));

					croppedArrowImage.convertTo(croppedArrowImage_f, CV_32F);
					croppedArrowImage_f = (croppedArrowImage_f - (255 / 2.0)) / 255;
					imshow("Video", croppedArrowImage);
					Mat inputBlobArrow = blobFromImage(croppedArrowImage_f, 1, Size(28, 28));   //Convert Mat to image batch
					net_arrow.setInput(inputBlobArrow, inBlobName);
					Mat result_arrow = net_arrow.forward(outBlobName);

					{
						double min, max, min1, max1;
						int min_loc[8];
						int max_loc[8];
						int min_loc1[8];
						int max_loc1[8];
						minMaxIdx(result_arrow, &min, &max, min_loc, max_loc);
						//cout << "result_arrow = " << endl << " " << result_arrow << endl << endl;
						elevator_status = max_loc[1];
					}


					//double error_noarrow = norm(croppedImage, image_noarrow, CV_L2);
					//double error_arrowdown = norm(croppedImage, image_arrowdown, CV_L2);
					//double error_arrowup = norm(croppedImage, image_arrowup, CV_L2);

					//// stop
					//if (error_noarrow < error_arrowdown && error_noarrow < error_arrowup) {
					//	elevator_status = 0;
					//}

					//// up
					//if (error_arrowup < error_noarrow && error_arrowup < error_arrowdown) {
					//	elevator_status = 1;
					//}

					//// down
					//if (error_arrowdown < error_noarrow && error_arrowdown < error_arrowup) {
					//	elevator_status = 2;
					//}

#define ABS_INT(a) ((a) < 0 ? (-(a)) : a)

					//if(elevator_status == 2)
					//{
					//	int dist0 = 0, dist1 = 0;
					//	for (int i = 1; i < 4; i++) {
					//		dist0 += ABS_INT(1 - elevator_record[(elevator_record_wptr - i) % 32]);
					//		dist1 += ABS_INT(2 - elevator_record[(elevator_record_wptr - i) % 32]);
					//	}

					//	if (dist0 < dist1)
					//		elevator_status = 1;
					//}

					//elevator_status_record[elevator_status_wptr++] = elevator_status;
					//elevator_status_wptr = elevator_status_wptr & 31;


					// detect floor
					cv::Mat croppedFloorImage = im_gray(roi_floor);
					cv::Mat croppedFloorImage_f;
					resize(croppedFloorImage, croppedFloorImage, cv::Size(28, 28));

					//char filename[64];
					//sprintf(filename, "dump_roi\\%d.jpg", i+1);
					//imwrite(filename, croppedFloorImage);

					croppedFloorImage.convertTo(croppedFloorImage_f, CV_32F);
					croppedFloorImage_f = (croppedFloorImage_f - (255 / 2.0)) / 255;

					

					Mat inputBlob = blobFromImage(croppedFloorImage_f, 1, Size(28,28));   //Convert Mat to image batch
					
					net_floor.setInput(inputBlob, inBlobName);
					Mat result_floor = net_floor.forward(outBlobName);

					double min, max, min1, max1;
					int min_loc[8];
					int max_loc[8];
					int min_loc1[8];
					int max_loc1[8];
					minMaxIdx(result_floor, &min, &max, min_loc, max_loc);
					//cout << "result_floor = " << endl << " " << result_floor << endl << endl;
					elevator_floor = max_loc[1];
					result_floor.at<float>(elevator_floor) = min;
					minMaxIdx(result_floor, &min1, &max1, min_loc1, max_loc1);
					elevator_floor_backup = max_loc1[1];
					//cout << "floor = " << elevator_floor << "posibility=" << max << "floor = " << elevator_floor_backup << "posibility=" << max1 << endl << endl;

					int elevator_floor_possible = elevator_record[(elevator_record_wptr - 1) & 31];
					if (elevator_record[(elevator_record_wptr - 1)&31] != 0) {
						if (elevator_status == 1) {
							elevator_floor_possible = elevator_record[(elevator_record_wptr - 1) & 31] + 1;
						}
						else if (elevator_status == 2) {
							elevator_floor_possible = elevator_record[(elevator_record_wptr - 1) & 31] - 1;
						}

						if (ABS_INT(elevator_floor_backup - elevator_floor_possible) < ABS_INT(elevator_floor - elevator_floor_possible)) {
							elevator_floor = elevator_floor_backup;
						}


					}


					// check again based on the direction of move
					if (elevator_status == 1 && elevator_floor < elevator_record[(elevator_record_wptr - 1) % 32] && elevator_floor_backup >= elevator_record[(elevator_record_wptr - 1) % 32]) {
						// elevator is moving up
						elevator_floor = elevator_floor_backup;
					}

					if (elevator_status == 1 && elevator_floor >= elevator_record[(elevator_record_wptr - 1) % 32] && elevator_floor_backup >= elevator_record[(elevator_record_wptr - 1) % 32] && elevator_floor >= elevator_floor_backup) {
						// elevator is moving up
						elevator_floor = elevator_floor_backup;
					}

					if (elevator_status == 2 && elevator_floor > elevator_record[(elevator_record_wptr - 1) % 32] && elevator_floor_backup <= elevator_record[(elevator_record_wptr - 1) % 32]) {
						// elevator is moving down
						elevator_floor = elevator_floor_backup;
					}

					if (elevator_status == 2 && elevator_floor <= elevator_record[(elevator_record_wptr - 1) % 32] && elevator_floor_backup <= elevator_record[(elevator_record_wptr - 1) % 32] && elevator_floor <= elevator_floor_backup) {
						// elevator is moving down
						elevator_floor = elevator_floor_backup;
					}

					elevator_record[elevator_record_wptr++] = elevator_floor;
					elevator_record_wptr = elevator_record_wptr & 31;

					//double error_min = DBL_MAX;
					//int floor_idx = 0;
					//for (int i = 0; i < 25; i++) {
					//	double error_floor = norm(croppedFloorImage, image_floor[i], CV_L2);
					//	if (error_floor < error_min) {
					//		error_min = error_floor;
					//		floor_idx = i;
					//	}
					//}
					//elevator_floor = floor_idx;



					//detectDoor(im_gray, i);


					//hs_cascade.detectMultiScale(im_gray, detected_hs, 1.1, 3, 0 | CV_HAAR_SCALE_IMAGE, cv::Size(50, 50));
					//upper_cascade.detectMultiScale(im_gray, detected_upper, 1.1, 3, 0 | CV_HAAR_SCALE_IMAGE, cv::Size(50, 50));
					//down_cascade.detectMultiScale(im_gray, detected_down, 1.1, 3, 0 | CV_HAAR_SCALE_IMAGE, cv::Size(50, 50));
					//full_cascade.detectMultiScale(im_gray, detected_full, 1.1, 3, 0 | CV_HAAR_SCALE_IMAGE, cv::Size(50, 50));
					//head_cascade.detectMultiScale(im_gray, detected_head, 1.1, 4, 0 | 1, Size(40, 40), Size(100, 100));

#if 1
					blur(imagRGB, imagRGB, Size(4, 4));

					pMOG2->apply(imagRGB, fgMaskMOG2);

					//morphologyEx(fgMaskMOG2, fgMaskMOG2, CV_MOP_ERODE, element);
					morphologyEx(fgMaskMOG2, fgMaskMOG2, CV_MOP_CLOSE, element);
					//morphologyEx(fgMaskMOG2, testImg, CV_MOP_OPEN, element);

					threshold(fgMaskMOG2, fgMaskMOG2, 128, 255, CV_THRESH_BINARY);

					//imshow("fgMaskMOG2", fgMaskMOG2);

#endif

					if (DETECT_DIRECTION == 1) {
						middle = (rawImage->height) / 2;
					}
				else {
					middle = (rawImage->width) / 2;
				}
                
				++i;//count it

			}
			if(singlestep){
				pause = true;
			}
			//First time:
			if(0 == i) {
				printf("\n . . . wait for it . . .\n"); //Just in case you wonder why the image is white at first
				//AVG METHOD ALLOCATION
				AllocateImages(rawImage);
				scaleHigh(scalehigh);
				scaleLow(scalelow);

				//CODEBOOK METHOD ALLOCATION:
				yuvImage = cvCloneImage(rawImage);
				ImaskCodeBook = cvCreateImage( cvGetSize(rawImage), IPL_DEPTH_8U, 1 );
				ImaskCodeBookCC = cvCreateImage( cvGetSize(rawImage), IPL_DEPTH_8U, 1 );

				cvSet(ImaskCodeBook,cvScalar(255));
				imageLen = rawImage->width*rawImage->height;
				cB = new codeBook [imageLen];
				for(int f = 0; f<imageLen; f++)
				{
 					cB[f].numEntries = 0;
				}
				for(int nc=0; nc<nChannels;nc++)
				{
					cbBounds[nc] = 10; //Learning bounds factor
				}
				ch[0] = true; //Allow threshold setting simultaneously for all channels
				ch[1] = true;
				ch[2] = true;
				// trajectory image
				kfdisp = cvCreateImage(cvGetSize(rawImage), IPL_DEPTH_8U, 3);
				display = cv::cvarrToMat(kfdisp);

				// label image
				labelImg = cvCreateImage(cvGetSize(ImaskCodeBookCC), IPL_DEPTH_LABEL, 1);

#if OPTICAL_FLOW
				prvs = im_gray.clone();
#endif
			}
			else {
#if OPTICAL_FLOW
				calcOpticalFlowFarneback(prvs, im_gray, flow, 0.5, 3, 15, 3, 5, 1.2, 0);

				Mat cflow;
				cvtColor(prvs, cflow, CV_GRAY2BGR);
				drawOptFlowMap(flow, cflow, 10, CV_RGB(0, 255, 0));
				imshow("OpticalFlowFarneback", cflow);

				prvs = im_gray.clone();

#endif



			}
			//If we've got an rawImage and are good to go:                
        	if( rawImage )
        	{
                cvCvtColor( rawImage, yuvImage, CV_BGR2YCrCb );//YUV For codebook method
				//This is where we build our background model
				if( !pause && i >= startcapture && i < endcapture  ){
					//LEARNING THE AVERAGE AND AVG DIFF BACKGROUND
					accumulateBackground(rawImage);
					//LEARNING THE CODEBOOK BACKGROUND
					pColor = (uchar *)((yuvImage)->imageData);
					for(int c=0; c<imageLen; c++)
					{
						cvupdateCodeBook(pColor, cB[c], cbBounds, nChannels);
						pColor += 3;
					}
				}
				//When done, create the background model
				if(i == endcapture){
					createModelsfromStats();
				}
				//Find the foreground if any
				if (i >= endcapture) {
					//FIND FOREGROUND BY CODEBOOK METHOD
#if MOG_BACKGROUND
					ImaskCodeBook = cvCloneImage(&(IplImage)fgMaskMOG2);
#else
					
					curPoint.clear();
					uchar maskPixelCodeBook;
					pColor = (uchar *)((yuvImage)->imageData); //3 channel yuv image
					uchar *pMask = (uchar *)((ImaskCodeBook)->imageData); //1 channel image
					for (int c = 0; c < imageLen; c++)
					{
						maskPixelCodeBook = cvbackgroundDiff(pColor, cB[c], nChannels, minMod, maxMod);
						*pMask++ = maskPixelCodeBook;
						pColor += 3;
					}
					//This part just to visualize bounding boxes and centers if desired
					cv::Mat masktmp = cv::cvarrToMat(ImaskCodeBook);
					min(fgMaskMOG2, masktmp, masktmp);
					//threshold(masktmp, masktmp, 128, 255, CV_THRESH_BINARY);
					ImaskCodeBook = cvCloneImage(&(IplImage)masktmp);
#endif
					cvCopy(ImaskCodeBook, ImaskCodeBookCC);
					cvconnectedComponents(ImaskCodeBookCC);
					Erosion(ImaskCodeBookCC, ImaskCodeBookCC);
					Dilation(ImaskCodeBookCC, ImaskCodeBookCC);

					CvBlobs blobs;
					unsigned int result = cvLabel(ImaskCodeBookCC, labelImg, blobs);

					cvFilterByArea(blobs, 0, 100000);

					cvUpdateTracks(blobs, tracks, 5., 10);

					// find the maximum track block

					if (i == endcapture && door_found == 0) {
						for (CvTracks::const_iterator it = tracks.begin(); it != tracks.end(); ++it)
						{
							if ((it->second->maxx - it->second->minx) > door_prev.width && (it->second->maxy - it->second->miny) > door_prev.height) {
								door_prev.width = it->second->maxx - it->second->minx;
								door_prev.height = it->second->maxy - it->second->miny;
								door_prev.x = it->second->minx;
								door_prev.y = it->second->miny;
							}
						}
					}
					else if (i > endcapture && door_found == 0) {
						for (CvTracks::const_iterator it = tracks.begin(); it != tracks.end(); ++it)
						{
							if ((it->second->maxx - it->second->minx) > door_current.width && (it->second->maxy - it->second->miny) > door_current.height) {
								door_current.width = it->second->maxx - it->second->minx;
								door_current.height = it->second->maxy - it->second->miny;
								door_current.x = it->second->minx;
								door_current.y = it->second->miny;
							}
						}

						if (door_current.width == door_prev.width && door_current.height == door_prev.height && door_current.x == door_prev.x && door_current.y == door_prev.y && !(door_current.width == 100 && door_current.height == 200 && door_current.x == 0 && door_current.y == 0)) {
							door_constant++;
						}
						else {
							door_constant = 0;
						}

						door_prev.width = door_current.width;
						door_prev.height = door_current.height;
						door_prev.x = door_current.x;
						door_prev.y = door_current.y;
					}



					for (CvTracks::const_iterator it = tracks.begin(); it != tracks.end(); ++it)
					{
						//if (mode&CV_TRACK_RENDER_ID)
						if (!it->second->inactive)
						{
							curPoint.push_back(trackPoint(it->second->id, (int)it->second->centroid.x, (int)it->second->centroid.y));
							centers.push_back(Point2d(it->second->centroid.x, it->second->centroid.y));
						}
					}

					if (DETECT_DIRECTION == 1) {
						if (prePoint.size() == curPoint.size())
						{
							for (int kk = 0; kk < prePoint.size(); kk++)
							{
								if (curPoint[kk].id == prePoint[kk].id)
								{
									if (curPoint[kk].y <= middle&&prePoint[kk].y > middle) //moving up
									{
										out++;
									}
									if (curPoint[kk].y >= middle&&prePoint[kk].y < middle) //moving down
									{
										in++;
									}
								}
							}
						}
					}
					else {
						if (prePoint.size() == curPoint.size())
						{
							for (int kk = 0; kk < prePoint.size(); kk++)
							{
								//line(display,prePoint[klm],curPoint[klm],);
								if (curPoint[kk].id == prePoint[kk].id)
								{
									if (curPoint[kk].x <= middle&&prePoint[kk].x > middle) //moving up
									{
										out++;
									}
									if (curPoint[kk].x >= middle&&prePoint[kk].x < middle) //moving down
									{
										in++;
									}
								}
							}
						}
					}

					if (centers.size() > 0)
					{
						tracker.Update(centers);

						if (prePoint.size() == curPoint.size())
						{
							for (int j = 0; j < centers.size(); j++)
							{
								line(display, Point(prePoint[j].x, prePoint[j].y), Point(curPoint[j].x, curPoint[j].y), Scalar(255, 0, 0), 1, 8, 0);
								cvCircle(kfdisp, Point(curPoint[j].x, curPoint[j].y), 2, Scalar(255, 255, 0), 1, 8, 0);
							}
						}

					}
					prePoint.clear();
					for (int blobid = 0; blobid < curPoint.size(); blobid++)
					{
						prePoint.push_back(curPoint[blobid]);
					}

					cvRenderBlobs(labelImg, blobs, ImaskCodeBookCC, rawImage, CV_BLOB_RENDER_CENTROID | CV_BLOB_RENDER_BOUNDING_BOX);
					cvRenderTracks(tracks, ImaskCodeBookCC, rawImage, CV_TRACK_RENDER_ID | CV_TRACK_RENDER_BOUNDING_BOX);

				}

                centers.clear();
				//if (DETECT_DIRECTION == 1) {
				//	cvPutText(rawImage, "Door", cvPoint(20, (rawImage->height) / 2), &font, CV_RGB(255, 0, 0));
				//	cvLine(rawImage, cvPoint(0, (rawImage->height) / 2), cvPoint(rawImage->width, (rawImage->height) / 2), CV_RGB(255, 0, 0), 1, 8, 0);
				//}
				//else {
				//	cvPutText(rawImage, "Door", cvPoint((rawImage->width) / 2-40, 30), &font, CV_RGB(255, 0, 0));
				//	cvLine(rawImage, cvPoint((rawImage->width) / 2, 0), cvPoint((rawImage->width) / 2, rawImage->height), CV_RGB(255, 0, 0), 1, 8, 0);
				//}
                
                stringstream ssin;
                stringstream ssout;
                ssin << in; ssout<<out;
                string innumber = ssin.str();
                string outnumber = ssout.str();


    //            cvPutText(rawImage, innumber.c_str(), cvPoint(60,40), &font, CV_RGB(255,0,0));
				//cvPutText(rawImage, "In: ", cvPoint(20,40), &font, CV_RGB(255,0,0));
				//cvPutText(rawImage, "Out: ", cvPoint(180,40), &font, CV_RGB(255,0,0));
    //            cvPutText(rawImage, outnumber.c_str(), cvPoint(230,40), &font, CV_RGB(255,0,0));


				if (elevator_status == 0) {
					cvPutText(rawImage, "Move: ", cvPoint(180, 40), &font, CV_RGB(255, 0, 0));
					cvPutText(rawImage, "Stop", cvPoint(235, 40), &font, CV_RGB(0, 0, 0));
				}
				else if (elevator_status == 1) {
					cvPutText(rawImage, "Move: ", cvPoint(180, 40), &font, CV_RGB(255, 0, 0));
					cvPutText(rawImage, "Up", cvPoint(235, 40), &font, CV_RGB(0, 0, 0));
				}
				else if (elevator_status == 2) {
					cvPutText(rawImage, "Move: ", cvPoint(180, 40), &font, CV_RGB(255, 0, 0));
					cvPutText(rawImage, "Down", cvPoint(235, 40), &font, CV_RGB(0, 0, 0));
				}

				if (elevator_floor > 0) {
					char buffer[16];
					cvPutText(rawImage, "Floor: ", cvPoint(180, 60), &font, CV_RGB(255, 0, 0));
					cvPutText(rawImage, itoa(elevator_floor, buffer, 10), cvPoint(235, 60), &font, CV_RGB(0, 0, 0));
				}
			
				cv::Mat imagtmp = cv::cvarrToMat(rawImage);

				//for (int i = 0; i < detected_hs.size(); i++) {

				//	rectangle(imagtmp, detected_hs[i].tl(), detected_hs[i].br(), Scalar(0, 0, 0), 2, 8, 0);

				//}

				//for (int i = 0; i < detected_upper.size(); i++) {

				//	rectangle(imagtmp, detected_upper[i].tl(), detected_upper[i].br(), Scalar(0, 0, 0), 2, 8, 0);

				//}

				//for (int i = 0; i < detected_down.size(); i++) {

				//	rectangle(imagtmp, detected_down[i].tl(), detected_down[i].br(), Scalar(0, 0, 0), 2, 8, 0);

				//}

				//for (int i = 0; i < detected_full.size(); i++) {
				
				//	rectangle(imagtmp, detected_full[i].tl(), detected_full[i].br(), Scalar(0, 0, 0), 2, 8, 0);

				//}

				//for (int i = 0; i < detected_head.size(); i++) {

				//	rectangle(imagtmp, detected_head[i].tl(), detected_head[i].br(), Scalar(0, 0, 0), 2, 8, 0);

				//}

				if (door_constant > 10) {
					if (door_found == 0) {
						door.x = door_current.x;
						door.y = door_current.y;
						door.width = door_current.width;
						door.height = door_current.height;

						door_found = 1;
					}
					
					door_constant = 0;

					door_prev.width = 100;
					door_prev.height = 200;
					door_current.width = 100;
					door_current.height = 200;

					
				}

				if (door_found == 1) {
					rectangle(imagtmp, door.tl(), door.br(), Scalar(255, 255, 255), 0.5, 8, 0);
					rawImage = cvCloneImage(&(IplImage)imagtmp);
					cvPutText(rawImage, "Door", cvPoint(door.x, door.y + door.height), &font, CV_RGB(255, 0, 0));
					//if (door_status == 0) {
					//	cvPutText(rawImage, "Open", cvPoint(door.x + door.width - 40, door.y + door.height), &font, CV_RGB(0, 0, 0));
					//}
					//else {
					//	cvPutText(rawImage, "Close", cvPoint(door.x + door.width - 40, door.y + door.height), &font, CV_RGB(0, 0, 0));
					//}

					if (tracks.size() == 0) {
						door_status = (door_status == 0) ? 1 : 0;
					}
				}


                // cvShowImage( "Video", rawImage );
#if SHOW_FOREGROUND
 				cvShowImage( "ForegroundCodeBook",ImaskCodeBook);
#endif
#if SHOW_CONTOUR
                cvShowImage( "CodeBook_ConnectComp",ImaskCodeBookCC);
#endif

#if SHOW_TRAJTORY
                imshow( "Trajetory_Analysis",display);
#endif
                //display.release();
                //USER INPUT:
                c = cvWaitKey(30)&0xFF;
				//End processing on ESC, q or Q
				if(c == 27 || c == 'q' | c == 'Q')
					break;
				//Else check for user input
				switch(c)
				{
					case 'h':
						help();
						break;
					case 'p':
						pause ^= 1;
						break;
					case 's':
						singlestep = 1;
						pause = false;
						break;
					case 'r':
						pause = false;
						singlestep = false;
						break;
					//AVG BACKROUND PARAMS
					case '-':
						if(i > endcapture){
							scalehigh += 0.25;
							printf("AVG scalehigh=%f\n",scalehigh);
							scaleHigh(scalehigh);
						}
						break;
					case '=':
						if(i > endcapture){
							scalehigh -= 0.25;
							printf("AVG scalehigh=%f\n",scalehigh);
							scaleHigh(scalehigh);
						}
						break;
					case '[':
						if(i > endcapture){
							scalelow += 0.25;
							printf("AVG scalelow=%f\n",scalelow);
							scaleLow(scalelow);
						}
						break;
					case ']':
						if(i > endcapture){
							scalelow -= 0.25;
							printf("AVG scalelow=%f\n",scalelow);
							scaleLow(scalelow);
						}
						break;
				//CODEBOOK PARAMS
                case 'y':
                case '0':
                        ch[0] = 1;
                        ch[1] = 0;
                        ch[2] = 0;
                        printf("CodeBook YUV Channels active: ");
                        for(n=0; n<nChannels; n++)
                                printf("%d, ",ch[n]);
                        printf("\n");
                        break;
                case 'u':
                case '1':
                        ch[0] = 0;
                        ch[1] = 1;
                        ch[2] = 0;
                        printf("CodeBook YUV Channels active: ");
                        for(n=0; n<nChannels; n++)
                                printf("%d, ",ch[n]);
                        printf("\n");
                        break;
                case 'v':
                case '2':
                        ch[0] = 0;
                        ch[1] = 0;
                        ch[2] = 1;
                        printf("CodeBook YUV Channels active: ");
                        for(n=0; n<nChannels; n++)
                                printf("%d, ",ch[n]);
                        printf("\n");
                        break;
                case 'a': //All
                case '3':
                        ch[0] = 1;
                        ch[1] = 1;
                        ch[2] = 1;
                        printf("CodeBook YUV Channels active: ");
                        for(n=0; n<nChannels; n++)
                                printf("%d, ",ch[n]);
                        printf("\n");
                        break;
                case 'b':  //both u and v together
                        ch[0] = 0;
                        ch[1] = 1;
                        ch[2] = 1;
                        printf("CodeBook YUV Channels active: ");
                        for(n=0; n<nChannels; n++)
                                printf("%d, ",ch[n]);
                        printf("\n");
                        break;
				case 'i': //modify max classification bounds (max bound goes higher)
					for(n=0; n<nChannels; n++){
						if(ch[n])
							maxMod[n] += 1;
						printf("%.4d,",maxMod[n]);
					}
					printf(" CodeBook High Side\n");
					break;
				case 'o': //modify max classification bounds (max bound goes lower)
					for(n=0; n<nChannels; n++){
						if(ch[n])
							maxMod[n] -= 1;
						printf("%.4d,",maxMod[n]);
					}
					printf(" CodeBook High Side\n");
					break;
				case 'k': //modify min classification bounds (min bound goes lower)
					for(n=0; n<nChannels; n++){
						if(ch[n])
							minMod[n] += 1;
						printf("%.4d,",minMod[n]);
					}
					printf(" CodeBook Low Side\n");
					break;
				case 'l': //modify min classification bounds (min bound goes higher)
					for(n=0; n<nChannels; n++){
						if(ch[n])
							minMod[n] -= 1;
						printf("%.4d,",minMod[n]);
					}
					printf(" CodeBook Low Side\n");
					break;
				}
				
            }
		}		
		capture.release();
		cvDestroyWindow("Video");
#if SHOW_FOREGROUND
		cvDestroyWindow("ForegroundCodeBook");
#endif
#if SHOW_CONTOUR
		cvDestroyWindow("CodeBook_ConnectComp");
#endif

#if SHOW_TRAJTORY
		cvDestroyWindow("Trajetory_Analysis");
#endif



        
		cvDestroyWindow( "ForegroundAVG" );
		cvDestroyWindow( "ForegroundCodeBook");
		cvDestroyWindow( "CodeBook_ConnectComp");
		DeallocateImages();
		if(yuvImage) cvReleaseImage(&yuvImage);
		if(ImaskAVG) cvReleaseImage(&ImaskAVG);
		if(ImaskAVGCC) cvReleaseImage(&ImaskAVGCC);
		if(ImaskCodeBook) cvReleaseImage(&ImaskCodeBook);
		if(ImaskCodeBookCC) cvReleaseImage(&ImaskCodeBookCC);

		delete [] cB;
    }
	else{ printf("\n\nDarn, Something wrong with the parameters\n\n"); help();
	}
    return 0;
}



