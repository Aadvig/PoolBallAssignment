
#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    std::cout << "OpenCV version: " << CV_VERSION << std::endl;

    // Create a 500x500 image
    cv::Mat image(500, 500, CV_8UC3, cv::Scalar(40, 40, 40));

    // Draw a green circle
    cv::circle(image, cv::Point(250, 250), 100,
               cv::Scalar(0, 255, 0), -1);

    // Display the image
    cv::imshow("OpenCV Test", image);

    std::cout << "OpenCV is working!" << std::endl;

    // Wait for a key press
    cv::waitKey(0);

    return 0;
}
