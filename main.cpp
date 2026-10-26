
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>

using namespace cv;
using namespace std;

// =====================================================
// ADJUSTABLE PARAMETERS
// =====================================================

const int TABLE_NUMBER = 1;

const string FOLDER =
    "/Users/aadyavig/CLionProjects/PoolBallAssignment/";

// Mean shift
const int SPATIAL_RADIUS = 15;
const int COLOUR_RADIUS = 20;
const int PYRAMID_LEVEL = 1;

// Canny
const int CANNY_LOW = 50;
const int CANNY_HIGH = 100;

// Hough circle detection
const double HOUGH_DP = 1.0;
const double HOUGH_MIN_DISTANCE = 25;
const double HOUGH_PARAM1 = 100;
const double HOUGH_PARAM2 = 25;

const int MIN_CIRCLE_RADIUS = 10;
const int MAX_CIRCLE_RADIUS = 50;

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

    Mat matrix = getPerspectiveTransform(
        original,
        transformed
    );

    Mat result;

    warpPerspective(
        image,
        result,
        matrix,
        Size(1434, 683)
    );

    return result;
}

// =====================================================
// MEAN SHIFT
// =====================================================

Mat applyMeanShift(const Mat& image) {

    Mat result;

    pyrMeanShiftFiltering(
        image,
        result,
        SPATIAL_RADIUS,
        COLOUR_RADIUS,
        PYRAMID_LEVEL
    );

    return result;
}

// =====================================================
// CANNY EDGE DETECTION
// =====================================================

Mat detectEdges(const Mat& image) {

    Mat gray;
    cvtColor(
        image,
        gray,
        COLOR_BGR2GRAY
    );

    Mat blurred;
    GaussianBlur(
        gray,
        blurred,
        Size(5, 5),
        1.5
    );

    Mat edges;
    Canny(
        blurred,
        edges,
        CANNY_LOW,
        CANNY_HIGH
    );

    return edges;
}

// =====================================================
// HOUGH CIRCLE DETECTION
// =====================================================

Mat detectCircles(
    const Mat& image,
    vector<Vec3f>& circles
) {

    Mat gray;
    cvtColor(
        image,
        gray,
        COLOR_BGR2GRAY
    );

    // Reduce image noise
    Mat blurred;

    GaussianBlur(
        gray,
        blurred,
        Size(9, 9),
        2,
        2
    );

    // Detect circles
    HoughCircles(
        blurred,
        circles,
        HOUGH_GRADIENT,
        HOUGH_DP,
        HOUGH_MIN_DISTANCE,
        HOUGH_PARAM1,
        HOUGH_PARAM2,
        MIN_CIRCLE_RADIUS,
        MAX_CIRCLE_RADIUS
    );

    Mat result = image.clone();

    // Draw circles
    for (size_t i = 0; i < circles.size(); i++) {

        Vec3f c = circles[i];

        Point centre(
            cvRound(c[0]),
            cvRound(c[1])
        );

        int radius = cvRound(c[2]);

        // Draw circumference
        circle(
            result,
            centre,
            radius,
            Scalar(0, 255, 0),
            2
        );

        // Draw centre
        circle(
            result,
            centre,
            3,
            Scalar(0, 0, 255),
            FILLED
        );

        cout << "Circle " << i + 1
             << ": Centre = ("
             << centre.x << ", "
             << centre.y << ")"
             << ", Radius = "
             << radius << endl;
    }

    return result;
}

// =====================================================
// MAIN
// =====================================================

int main() {

    string filename =
        "Table" + to_string(TABLE_NUMBER) + ".jpeg";

    Mat image = imread(FOLDER + filename);

    if (image.empty()) {
        cout << "Error loading image: "
             << filename << endl;
        return -1;
    }

    // 1. Perspective transformation
    Mat transformed =
        perspectiveTransformTable(image);

    // 2. Mean shift
    Mat segmented =
        applyMeanShift(transformed);

    // 3. Canny edge detection
    Mat edges =
        detectEdges(segmented);

    // 4. Hough circle transform
    vector<Vec3f> circles;

    Mat result =
        detectCircles(segmented, circles);

    // 5. Print results
    cout << endl;
    cout << "Total detected circles: "
         << circles.size() << endl;

    // 6. Display results
    imshow(
        "Original Image",
        image
    );

    imshow(
        "Perspective Transform",
        transformed
    );

    imshow(
        "Mean Shift",
        segmented
    );

    imshow(
        "Canny Edges",
        edges
    );

    imshow(
        "Hough Circle Detection",
        result
    );

    // 7. Save results
    imwrite(
        FOLDER + "HoughCircles.png",
        result
    );

    imwrite(
        FOLDER + "CannyEdges.png",
        edges
    );

    waitKey(0);
    destroyAllWindows();

    return 0;
}
