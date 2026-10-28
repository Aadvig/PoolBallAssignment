
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>

using namespace cv;
using namespace std;

// =====================================================
// Perspective transformation
// =====================================================
Mat perspectiveTransformTable(const Mat& image) {

    Point2f original[4] = {
        Point2f(337, 35),    // Top left
        Point2f(1570, 21),   // Top right
        Point2f(38, 472),    // Bottom left
        Point2f(1888, 451)   // Bottom right
    };

    Point2f transformed[4] = {
        Point2f(0, 0),
        Point2f(1433, 0),
        Point2f(0, 682),
        Point2f(1433, 682)
    };

    Mat matrix = getPerspectiveTransform(original, transformed);

    Mat result;
    warpPerspective(image, result, matrix, Size(1434, 683));

    return result;
}


// =====================================================
// Generic colour detection
// =====================================================
Mat detectColour(const Mat& image,
                 const string& colourName,
                 Scalar lowerHSV,
                 Scalar upperHSV) {

    // Convert image from BGR to HSV
    Mat hsv;
    cvtColor(image, hsv, COLOR_BGR2HSV);

    // Create binary mask for selected colour
    Mat colourMask;
    inRange(hsv, lowerHSV, upperHSV, colourMask);

    // Remove noise and fill small gaps
    Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));

    morphologyEx(colourMask, colourMask, MORPH_OPEN, kernel);
    morphologyEx(colourMask, colourMask, MORPH_CLOSE, kernel);

    // Highlight detected regions
    Mat overlay = image.clone();
    overlay.setTo(Scalar(255, 50, 255), colourMask);

    Mat highlighted;
    addWeighted(image, 0.6, overlay, 0.4, 0, highlighted);

    // Display results
    imshow(colourName + " Mask", colourMask);
    imshow(colourName + " Highlighted", highlighted);

    return colourMask;
}


// =====================================================
// Main
// =====================================================
int main() {

    // Image directory
    string folder = "/Users/aadyavig/CLionProjects/PoolBallAssignment/";
    string filename = "Table6.jpeg";

    // Load image
    Mat image = imread(folder + filename);

    if (image.empty()) {
        cout << "Error loading image!" << endl;
        return -1;
    }

    // Perspective transformation
    Mat transformed = perspectiveTransformTable(image);

    // =================================================
    // Colour thresholds
    // =================================================

    // Yellow (done thresholds)
    Scalar lowerYellow(20, 200, 200);
    Scalar upperYellow(43, 255, 255);

    // Blue (done thresholds)
    Scalar lowerBlue(116, 130, 65);
    Scalar upperBlue(123, 255, 255);

    // Red (starting thresholds)
    Scalar lowerRed(173, 170, 140);
    Scalar upperRed(179, 255, 255);

    Mat redMask = detectColour(
        transformed, "Red", lowerRed, upperRed
    );

    imshow("Transformed", transformed);
    imshow("red", redMask);

    waitKey(0);
    destroyAllWindows();

    return 0;
}
