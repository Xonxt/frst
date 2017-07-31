#include "frst.h"

#include <iostream>
#include <string.h>

int main(int argc, char* argv[]) {

	cv::Mat image;

	if (argc > 1) {
		image = cv::imread(argv[1]);
	}
	else {
		image = cv::imread("image.jpeg");
	}

	if (!image.data) {
		std::cout << "Could not open or find the image" << std::endl;
		return -1;
	}

	cv::namedWindow("Display window", cv::WINDOW_AUTOSIZE);

	// lose the Alpha channel
	if (image.channels() == 4) {
		cv::cvtColor(image, image, CV_BGRA2BGR);
	}

	// convert to grayscale
	cv::Mat grayImg;
	cv::cvtColor(image, grayImg, CV_BGR2GRAY);

	// apply FRST
	cv::Mat frstImage;
	frst2d(grayImg, frstImage, 12, 2, 0.1, FRST_MODE_DARK);

	// the frst will have irregular values, normalize them!
	cv::normalize(frstImage, frstImage, 0.0, 1.0, cv::NORM_MINMAX);
	frstImage.convertTo(frstImage, CV_8U, 255.0);

	// the frst image is grayscale, let's binarize it
	cv::Mat markers;
	cv::threshold(frstImage, frstImage, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
	bwMorph(frstImage, markers, cv::MORPH_CLOSE, cv::MORPH_ELLIPSE, 5);

	// the 'markers' image contains dots of different size. Let's vectorize it
	std::vector< std::vector<cv::Point> > contours;
	std::vector<cv::Vec4i> hierarchy;

	contours.clear();
	cv::findContours(markers, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

	// get the moments
	std::vector<cv::Moments> mu(contours.size());
	for (int i = 0; i < contours.size(); i++)
	{
		mu[i] = moments(contours[i], false);
	}

	//  get the mass centers:
	std::vector<cv::Point2f> mc(contours.size());
	for (int i = 0; i < contours.size(); i++)
	{
		mc[i] = cv::Point2f(mu[i].m10 / mu[i].m00, mu[i].m01 / mu[i].m00);
	}

	// draw the point centers	
	for (int i = 0; i< contours.size(); i++)
	{		
		cv::circle(image, mc[i], 2, CV_RGB(0,255,0), -1, 8, 0);
	}
	
	// display the image
	for (;;) {
		cv::imshow("Display window", image);

		char ch = cv::waitKey(10);

		if (char(ch) == 27)
			break;
	}

	return 0;
}

