#include "Utilities.h"
#include <iostream>
#include <fstream>

#define ANGLE_TOL 20
float getAngle(Vec4i line) {
	float x1 = line.val[0], y1 = line.val[1], x2 = line.val[2], y2 = line.val[3];
	return atan2(y2 - y1, x2 - x1) * 180.0 / CV_PI;
}

double calcAngle(Vec4i line) {
	double x1 = line.val[0], y1 = line.val[1], x2 = line.val[2], y2 = line.val[3];
	double x_dist = x1 - x2;
	double y_dist = y1 - y2;
	double x = pow(x_dist, 2), y = pow(y_dist, 2);
	double d = abs(sqrt(x + y));
	double radians = atan2(y, x);

	// return value in degrees
	return radians * 180 / CV_PI;
}

bool isHorizontal(float angle) {
	return (angle > -ANGLE_TOL && angle < ANGLE_TOL) || (angle > (180- ANGLE_TOL) && angle < (180+ ANGLE_TOL));
}

bool isVertical(float angle) {
	return (angle > (90- ANGLE_TOL) && angle < (90+ ANGLE_TOL)) || (angle > -(90+ ANGLE_TOL) && angle < -(90- ANGLE_TOL));
}

bool isSkewed(float angle) {
	return (angle > -45 && angle < 0);
}

// check if two points are within 25 pixels
bool tolerance(Point p1, Point p2) {
	double x_dist = pow(p1.x - p2.x, 2);
	double y_dist = pow(p1.y - p2.y, 2);
	return abs(sqrt(x_dist + y_dist)) < 25;
}

Point interpolate(Point p1, Point p2) {
	return Point((p1.x + p2.x) / 2, (p1.y + p2.y) / 2);
}

void equaliseLighting(Mat image, bool showOutput = false) {
	// equalise the colours during light changes
	cvtColor(image, image, CV_BGR2YCrCb);

	vector<Mat> channels;
	split(image, channels);
	equalizeHist(channels[0], channels[0]);
	merge(channels, image);
	cvtColor(image, image, CV_YCrCb2BGR);

	if (showOutput)
		imshow("Equalised", image);
}

void detectDoor(Mat &im_gray, int frame_count) {
	Mat src, output;


	// generate lines
	vector<Vec4i> lines, horizontalLines, verticalLines, skewedLines;

	// Canny for getting outlines
	Mat grey;
	im_gray.copyTo(grey);
	Canny(grey, grey, 50, 255);
	imshow("Canny", grey);

	// detect hough lines probabilistically
	HoughLinesP(grey, lines, 1, CV_PI / 180, 50, 200, 50);

	for (int i = 0; i < lines.size(); i++) {
		Vec4i l = lines[i];
		float ratio = l[1] / grey.rows;
		double angle = getAngle(l);

		if (isHorizontal(angle))
			horizontalLines.push_back(l);
		else if (isVertical(angle))
			verticalLines.push_back(l);
		else if (isSkewed(angle))
			skewedLines.push_back(l);
	}

	Mat edges;
	im_gray.copyTo(edges);
	//for (Vec4i line : horizontalLines) {
	//	Point point1(line[0], line[1]);
	//	Point point2(line[2], line[3]);
	//	cv::line(edges, point1, point2, Scalar(0, 0, 255), 2);
	//}

	for (Vec4i line : verticalLines) {
		Point point1(line[0], line[1]);
		Point point2(line[2], line[3]);
		cv::line(edges, point1, point2, Scalar(255, 0, 0), 2);
	}

	//for (Vec4i line : skewedLines) {
	//	Point point1(line[0], line[1]);
	//	Point point2(line[2], line[3]);
	//	cv::line(edges, point1, point2, Scalar(255, 255, 0), 2);

	//	const int DELAY = 50;

	//	if (skewedLines.size() > 0) {

	//	}
	//}

	vector<Point> doorPoints;

	for (int h = 0; h < horizontalLines.size(); h++) {
		for (int v = 0; v < verticalLines.size(); v++) {
			Vec4i h_line = horizontalLines[h];
			Vec4i v_line = verticalLines[v];

			Point h_p1 = Point(h_line[0], h_line[1]);
			Point h_p2 = Point(h_line[2], h_line[3]);

			Point v_p1 = Point(v_line[0], v_line[1]);
			Point v_p2 = Point(v_line[2], v_line[3]);

			// check if points are within tolerance of one another
			if (tolerance(h_p1, v_p1)) {
				doorPoints.push_back(interpolate(h_p1, v_p1));
			}
			if (tolerance(h_p1, v_p2)) {
				doorPoints.push_back(interpolate(h_p1, v_p2));
			}
			if (tolerance(h_p2, v_p1)) {
				doorPoints.push_back(interpolate(h_p2, v_p1));
			}
			if (tolerance(h_p2, v_p2)) {
				doorPoints.push_back(interpolate(h_p2, v_p2));
			}
		}
	}

	for (int c = 0; c < doorPoints.size(); c++) {
		cv::circle(edges, doorPoints[c], 5, Scalar(0, 255, 0), 2);
	}

	// print frame count to mat
	string frame_text("Frame " + to_string(frame_count));
	putText(edges, frame_text, cvPoint(30, 30), FONT_HERSHEY_COMPLEX_SMALL, 0.5, cvScalar(0, 0, 255), 1, CV_AA);

	imshow("Edges", edges);


}