#include "frst.h"

#include <iostream>
#include <string.h>

int main(int argc, char* argv[]) {
  
  cv::Mat image;
  
  if (argc > 1) {
    image = cv::imread(argv[1]);
  }
  else {
    image = cv::imread("C:\\Users\\kovalenm\\Work\\imgs\\Tumor\\Tumor_004.png");
  }
  
  if(!image.data) {
    std::cout <<  "Could not open or find the image" << std::endl;
    return -1;
  }

  cv::namedWindow("Display window", WINDOW_AUTOSIZE);
  
  if (image.channels() == 4) {
    cv::cvtColor(image, image, CV_BGRA2BGR);
  }
  
  cv::Mat grayImg;
  cv::cvtColor(image, grayImg, CV_BGR2GRAY);
  
  cv::Mat frstImage;
  frst2d(grayImg, frstImage, 12, 2, 0.1, FRST_MODE_DARK);
  
  cv::normalize(frstImage, frstImage, 0.0, 1.0, cv::NORM_MINMAX);
	frstImage.convertTo(frstImage, CV_8U, 255.0); // call the FRST
  
  for(;;) {
    cv::imshow("Display window", frstImage);
    
    char ch = cv::waitKey(10);
    
    if (char(ch) == 27) 
      break;
  }

  return 0;
}