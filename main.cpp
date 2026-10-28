
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

// Perspective transformation
Mat perspectiveTransformTable(const Mat& image) {

    Point2f original[4] = {
        Point2f(337, 35),
        Point2f(1570, 21),
        Point2f(38, 472),
        Point2f(1888, 451)
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

// Canny edge detection
Mat detectEdges(const Mat& image) {

    // Convert to grayscale
    Mat gray;
    cvtColor(image, gray, COLOR_BGR2GRAY);

    // Reduce noise
    Mat blurred;
    GaussianBlur(gray, blurred, Size(5, 5), 1);

    // Apply Canny
    Mat edges;
    Canny(blurred, edges, 25, 75);

    return edges;
}

int main() {

    string folder = "/Users/aadyavig/CLionProjects/PoolBallAssignment/";
    string filename = "Table4.jpeg";

    Mat image = imread(folder + filename);

    if (image.empty()) {
        cout << "Error loading image!" << endl;
        return -1;
    }

    // Perspective transformation
    Mat transformed = perspectiveTransformTable(image);

    // Edge detection
    Mat edges = detectEdges(transformed);

    // Display images
    imshow("Transformed Image", transformed);
    imshow("Canny Edges", edges);

    // Save results
    imwrite(folder + "CannyEdges.png", edges);

    waitKey(0);
    destroyAllWindows();

    return 0;
}
