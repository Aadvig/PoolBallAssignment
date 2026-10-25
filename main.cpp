
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <algorithm>
#include <string>

using namespace cv;
using namespace std;

// =====================================================
// PERSPECTIVE TRANSFORMATION
// =====================================================

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


// =====================================================
// MEDIAN BACKGROUND MODEL
// =====================================================

Mat createBackground(const vector<Mat>& images) {

    // All images must have the same dimensions
    Mat background(images[0].size(), CV_8UC3);

    vector<uchar> values(images.size());

    // Process every pixel
    for (int y = 0; y < background.rows; y++) {
        for (int x = 0; x < background.cols; x++) {

            // Process B, G and R separately
            for (int c = 0; c < 3; c++) {

                // Collect pixel values from all images
                for (int i = 0; i < images.size(); i++) {
                    values[i] = images[i].at<Vec3b>(y, x)[c];
                }

                // Sort values
                sort(values.begin(), values.end());

                // Calculate median (average middle two for 6 images)
                int n = values.size();
                uchar median = static_cast<uchar>(
                    (int(values[n / 2 - 1]) +
                     int(values[n / 2])) / 2
                );

                // Assign median to background image
                background.at<Vec3b>(y, x)[c] = median;
            }
        }
    }

    return background;
}


// =====================================================
// MAIN
// =====================================================

int main() {

    string folder = "/Users/aadyavig/CLionProjects/PoolBallAssignment/";

    vector<Mat> transformedImages;

    // Load and transform all six images
    for (int i = 1; i <= 6; i++) {

        string filename = "Table" + to_string(i) + ".jpeg";

        Mat image = imread(folder + filename);

        if (image.empty()) {
            cout << "Error loading: " << filename << endl;
            return -1;
        }

        Mat transformed = perspectiveTransformTable(image);

        transformedImages.push_back(transformed);

        cout << "Processed " << filename << endl;
    }

    // Generate median background
    Mat background = createBackground(transformedImages);

    // Display background
    imshow("Median Background", background);

    // Save background
    imwrite(folder + "MedianBackground.png", background);

    cout << "Background generated successfully!" << endl;

    waitKey(0);
    destroyAllWindows();

    return 0;
}
