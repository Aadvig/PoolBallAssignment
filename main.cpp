
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

// Perspective transformation
Mat perspectiveTransformTable(const Mat& image) {

    // Original table corners (x = column, y = row)
    Point2f original[4] = {
        Point2f(337, 35),    // Top left
        Point2f(1570, 21),   // Top right
        Point2f(38, 472),    // Bottom left
        Point2f(1888, 451)   // Bottom right
    };

    // Destination corners
    Point2f transformed[4] = {
        Point2f(0, 0),
        Point2f(1433, 0),
        Point2f(0, 682),
        Point2f(1433, 682)
    };

    // Calculate transformation matrix
    Mat matrix = getPerspectiveTransform(original, transformed);

    // Apply transformation
    Mat result;
    warpPerspective(image, result, matrix, Size(1434, 683));

    return result;
}



Mat detectBackground(const Mat& image) {

    // Convert image from BGR to HSV
    Mat hsv;
    cvtColor(image, hsv, COLOR_BGR2HSV);

    // Define HSV range for green felt
    Scalar lowerGreen(30, 20, 60);
    Scalar upperGreen(150, 90, 255);

    // Create binary mask (white = background)
    Mat mask;
    inRange(hsv, lowerGreen, upperGreen, mask);

    // Create a red overlay
    Mat overlay = image.clone();
    overlay.setTo(Scalar(0, 0, 255), mask);

    // Blend overlay with original image
    Mat highlighted;
    addWeighted(image, 0.5, overlay, 0.5, 0, highlighted);

    // Display results
    imshow("Background Mask", mask);
    imshow("Highlighted Background", highlighted);

    return mask;
}


int main() {

    // Load image
    string folder = "/Users/aadyavig/CLionProjects/PoolBallAssignment/";
    string filename = "Table1.jpeg";

    Mat image = imread(folder + filename);

    if (image.empty()) {
        cout << "Error loading image!" << endl;
        return -1;
    }

    // Apply perspective transformation
    Mat transformed = perspectiveTransformTable(image);

    // Detect green background
    Mat backgroundMask = detectBackground(transformed);

    // Display transformed image
    imshow("Transformed Image", transformed);

    waitKey(0);
    destroyAllWindows();

    // Display results
    imshow("Original Image", image);
    imshow("Transformed Image", transformed);

    // Save transformed image
    imwrite("TransformedTable.jpeg", transformed);

    waitKey(0);
    destroyAllWindows();

    return 0;
}
