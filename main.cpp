#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <vector>

using namespace cv;
using namespace std;

// =====================================================
// ADJUSTABLE PARAMETERS
// =====================================================

const int TABLE_NUMBER = 2;
const string SELECTED_COLOUR = "Yellow";   // Change this only

// Mean shift
const int SPATIAL_RADIUS = 15;
const int COLOUR_RADIUS = 20;
const int PYRAMID_LEVEL = 1;

// Canny
const int CANNY_LOW = 250;
const int CANNY_HIGH = 300;

// Colour mask cleanup
const int MASK_KERNEL_SIZE = 3;
const int EDGE_EXPANSION = 7;

// Circle detection
const int REGION_MERGE_DISTANCE = 5;
const int MIN_CIRCLE_RADIUS = 20;
const int MAX_CIRCLE_RADIUS = 50;

// Folder
const string FOLDER =
    "/Users/aadyavig/CLionProjects/PoolBallAssignment/";

// =====================================================
// COLOUR PARAMETERS
// =====================================================

struct ColourRange {
    string name;
    Scalar lower;
    Scalar upper;
};

vector<ColourRange> getColourRanges() {
    return {
        {"Yellow", Scalar(20, 200, 200), Scalar(43, 255, 255)},
        {"Blue",   Scalar(116, 130, 65), Scalar(123, 255, 255)},
        {"Red",    Scalar(130, 170, 140), Scalar(179, 255, 255)},
        {"Brown",  Scalar(0, 110, 65), Scalar(12, 210, 170)},
        {"Green",  Scalar(40, 100, 60), Scalar(85, 255, 255)},
        {"Orange", Scalar(8, 130, 100), Scalar(20, 255, 255)},
        {"Purple", Scalar(125, 80, 50), Scalar(155, 255, 255)},
        {"Black",  Scalar(0, 0, 0), Scalar(179, 255, 65)},
        {"White",  Scalar(0, 0, 170), Scalar(179, 65, 255)}
    };
}

ColourRange getSelectedColourRange(const string& colourName) {
    vector<ColourRange> colours = getColourRanges();

    for (const auto& colour : colours) {
        if (colour.name == colourName) {
            return colour;
        }
    }

    cout << "Colour not found. Defaulting to Blue." << endl;
    return {"Blue", Scalar(116, 130, 65), Scalar(123, 255, 255)};
}

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

    Point2f destination[4] = {
        Point2f(0, 0),
        Point2f(1433, 0),
        Point2f(0, 682),
        Point2f(1433, 682)
    };

    Mat matrix = getPerspectiveTransform(original, destination);

    Mat result;
    warpPerspective(image, result, matrix, Size(1434, 683));

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
// CANNY
// =====================================================

Mat detectEdges(const Mat& image) {

    Mat gray, edges;

    cvtColor(image, gray, COLOR_BGR2GRAY);
    Canny(gray, edges, CANNY_LOW, CANNY_HIGH);

    return edges;
}

// =====================================================
// COLOUR DETECTION
// =====================================================

Mat detectColour(const Mat& hsv, const ColourRange& colour) {

    Mat mask;

    inRange(hsv, colour.lower, colour.upper, mask);

    Mat kernel = getStructuringElement(
        MORPH_ELLIPSE,
        Size(MASK_KERNEL_SIZE, MASK_KERNEL_SIZE)
    );

    morphologyEx(mask, mask, MORPH_OPEN, kernel);

    return mask;
}

// =====================================================
// COLOUR OVERLAY
// =====================================================

Mat highlightColour(const Mat& image, const Mat& mask) {

    Mat overlay = image.clone();
    overlay.setTo(Scalar(255, 0, 255), mask);   // magenta

    Mat result;
    addWeighted(image, 0.6, overlay, 0.4, 0, result);

    return result;
}

// =====================================================
// COMBINE COLOUR + CANNY
// =====================================================



Mat detectBallCircles(const Mat& image,
                      const Mat& edges,
                      const Mat& colourMask) {

    // =====================================================
    // 1. Find regions of the selected colour
    // =====================================================

    vector<vector<Point>> contours;

    findContours(
        colourMask.clone(),
        contours,
        RETR_EXTERNAL,
        CHAIN_APPROX_SIMPLE
    );

    Mat result = image.clone();

    for (const auto& contour : contours) {

        if (contour.size() < 3)
            continue;

        // =================================================
        // 2. Initial bounding circle from colour region
        // =================================================

        Point2f initialCentre;
        float initialRadius;

        minEnclosingCircle(
            contour,
            initialCentre,
            initialRadius
        );

        // Ignore tiny colour regions
        if (initialRadius < MIN_CIRCLE_RADIUS)
            continue;

        // Avoid regions already too large to be one ball
        if (initialRadius > MAX_CIRCLE_RADIUS)
            continue;

        // =================================================
        // 3. Find nearby Canny edges
        // =================================================

        const int EDGE_SEARCH_DISTANCE = 10;

        Mat searchMask = Mat::zeros(
            image.size(), CV_8U
        );

        // Create a search region around the initial circle
        circle(
            searchMask,
            initialCentre,
            cvRound(initialRadius) + EDGE_SEARCH_DISTANCE,
            Scalar(255),
            FILLED
        );

        // Select Canny edges inside the search region
        Mat nearbyEdges;

        bitwise_and(
            edges,
            searchMask,
            nearbyEdges
        );

        // =================================================
        // 4. Refine circle using colour + nearby edges
        // =================================================

        vector<Point> edgePoints;
        findNonZero(nearbyEdges, edgePoints);

        vector<Point2f> combinedPoints;

        // Include all colour contour points
        for (const Point& p : contour) {
            combinedPoints.push_back(Point2f(p));
        }

        // Include nearby Canny edge points
        for (const Point& p : edgePoints) {
            combinedPoints.push_back(Point2f(p));
        }

        Point2f finalCentre;
        float finalRadius;

        minEnclosingCircle(
            combinedPoints,
            finalCentre,
            finalRadius
        );

        // Reject oversized results
        if (finalRadius < MIN_CIRCLE_RADIUS ||
            finalRadius > MAX_CIRCLE_RADIUS)
            continue;

        // =================================================
        // 5. Draw detected ball
        // =================================================

        circle(
            result,
            finalCentre,
            cvRound(finalRadius),
            Scalar(0, 255, 0),
            2
        );

        circle(
            result,
            finalCentre,
            3,
            Scalar(0, 0, 255),
            -1
        );

        cout << "Ball candidate: ("
             << finalCentre.x << ", "
             << finalCentre.y << "), radius "
             << finalRadius << endl;
    }

    return result;
}




// =====================================================
// MAIN
// =====================================================

int main() {

    string filename = "Table" + to_string(TABLE_NUMBER) + ".jpeg";

    Mat image = imread(FOLDER + filename);

    if (image.empty()) {
        cout << "Error loading: " << filename << endl;
        return -1;
    }

    // 1. Perspective transform
    Mat transformed = perspectiveTransformTable(image);

    // 2. Mean shift
    Mat segmented = applyMeanShift(transformed);

    // 3. Canny
    Mat edges = detectEdges(segmented);

    // 4. Convert to HSV
    Mat hsv;
    cvtColor(segmented, hsv, COLOR_BGR2HSV);

    // 5. Get chosen colour
    ColourRange selected = getSelectedColourRange(SELECTED_COLOUR);

    // 6. Detect selected colour
    Mat colourMask = detectColour(hsv, selected);

    // 7. Overlay colour mask
    Mat colourOverlay = highlightColour(transformed, colourMask);

    // 8. Combine colour + canny
    Mat combined = detectBallCircles(transformed, edges, colourMask);

    // =================================================
    // DISPLAY ONLY FIVE IMAGES
    // =================================================

    imshow("1 - Perspective Transform", transformed);
    imshow("2 - Mean Shift", segmented);
    imshow("3 - Colour Overlay: " + selected.name, colourOverlay);
    imshow("4 - Canny on Mean Shift", edges);
    imshow("5 - Combined Detection: " + selected.name, combined);

    waitKey(0);
    destroyAllWindows();

    return 0;
}