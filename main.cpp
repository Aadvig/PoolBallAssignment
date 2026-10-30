#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

using namespace cv;
using namespace std;

// =====================================================
// ADJUSTABLE PARAMETERS
// =====================================================

const int TABLE_NUMBER = 6;
const string SELECTED_COLOUR = "Black";   // Change this only
const int BORDER_MARGIN = 15;  // Pixels removed from the outer image edge
const int GREEN_BORDER_MARGIN = 100;

// Mean shift
const int SPATIAL_RADIUS = 15;
const int COLOUR_RADIUS = 10;
const int PYRAMID_LEVEL = 1;

// Canny
const int CANNY_LOW = 50;
const int CANNY_HIGH = 100;

// Colour mask cleanup
const int MASK_KERNEL_SIZE = 3;
const int EDGE_EXPANSION = 17;

// Circle detection
const int REGION_MERGE_DISTANCE = 10;
const int MIN_CIRCLE_RADIUS = 10;
const int MAX_CIRCLE_RADIUS = 100;

// Folder
const string FOLDER =
    "/Users/aadyavig/CLionProjects/PoolBallAssignment/";

// =====================================================
// COLOUR PARAMETER
// =====================================================

struct ColourRange {
    string name;
    Scalar lower;
    Scalar upper;
};

vector<ColourRange> getColourRanges() {
    return {
        {"Red",  Scalar(0, 170, 115), Scalar(0, 255, 255)},
        {"Red2", Scalar(173, 170, 115), Scalar(179, 255, 255)},
        {"Orange", Scalar(7, 145, 130), Scalar(13, 255, 255)},
        {"Yellow", Scalar(20, 200, 200), Scalar(39, 255, 255)},
        {"Green",  Scalar(67, 110, 35), Scalar(74, 160, 90)},
        {"Green2", Scalar(58, 95, 38),  Scalar(71, 165, 102)},
        {"Blue",   Scalar(113, 125, 55), Scalar(123, 255, 255)},
        {"Purple", Scalar(125, 65, 30),  Scalar(165, 255, 255)},
        {"Brown", Scalar(0, 120, 110), Scalar(9, 220, 155)},
        {"Black", Scalar(0, 0, 0), Scalar(170, 100, 30)},
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
// COLOUR DETECTION: HSV THRESHOLDS ONLY
// =====================================================

Mat detectColour(const Mat& hsv,
                 const ColourRange& colour) {

    // =====================================================
    // 1. HSV colour detection
    // =====================================================

    Mat mask;
    inRange(hsv, colour.lower, colour.upper, mask);

    // Combine secondary HSV ranges
    string secondaryName;

    if (colour.name == "Red")
        secondaryName = "Red2";

    if (colour.name == "Green")
        secondaryName = "Green2";

    if (colour.name == "Blue")
        secondaryName = "Blue2";

    if (colour.name == "Purple")
        secondaryName = "Purple2";

    if (!secondaryName.empty()) {

        for (const auto& range : getColourRanges()) {

            if (range.name == secondaryName) {

                Mat secondMask;

                inRange(
                    hsv,
                    range.lower,
                    range.upper,
                    secondMask
                );

                bitwise_or(mask, secondMask, mask);
                break;
            }
        }
    }

    // =====================================================
    // 2. Morphological cleanup
    // =====================================================

    Mat kernel = getStructuringElement(
        MORPH_ELLIPSE,
        Size(MASK_KERNEL_SIZE, MASK_KERNEL_SIZE)
    );

    morphologyEx(
        mask,
        mask,
        MORPH_OPEN,
        kernel
    );

    // =====================================================
    // 3. Border exclusion
    // =====================================================

    int margin = (colour.name == "Green")
        ? GREEN_BORDER_MARGIN
        : BORDER_MARGIN;

    margin = max(
        0,
        min(margin, min(mask.rows, mask.cols) / 2)
    );

    if (margin > 0) {

        mask.rowRange(0, margin).setTo(0);

        mask.rowRange(
            mask.rows - margin,
            mask.rows
        ).setTo(0);

        mask.colRange(0, margin).setTo(0);

        mask.colRange(
            mask.cols - margin,
            mask.cols
        ).setTo(0);
    }

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

    Mat result = image.clone();

    // =====================================================
    // WHITE BALL: Largest connected white region only
    // =====================================================

    if (SELECTED_COLOUR == "White") {

        Mat labels, stats, centroids;

        int count = connectedComponentsWithStats(
            colourMask,
            labels,
            stats,
            centroids,
            8,
            CV_32S
        );

        int largestLabel = -1;
        int largestArea = 0;

        // Find the largest white connected component
        for (int i = 1; i < count; i++) {

            int area = stats.at<int>(i, CC_STAT_AREA);

            if (area > largestArea) {
                largestArea = area;
                largestLabel = i;
            }
        }

        if (largestLabel != -1) {

            // Extract only the largest white region
            Mat largestMask = (labels == largestLabel);

            vector<vector<Point>> contours;
            findContours(
                largestMask.clone(),
                contours,
                RETR_EXTERNAL,
                CHAIN_APPROX_SIMPLE
            );

            if (!contours.empty()) {

                Point2f centre;
                float radius;

                minEnclosingCircle(
                    contours[0],
                    centre,
                    radius
                );

                // Draw only the white ball
                circle(result, centre, cvRound(radius),
                       Scalar(0, 255, 0), 2);

                circle(result, centre, 3,
                       Scalar(0, 0, 255), -1);

                cout << "White ball: " << centre
                     << ", radius: " << radius << endl;
            }
        }

        return result;
    }

    // =====================================================
    // ALL OTHER COLOURS: Existing colour + Canny method
    // =====================================================

    vector<vector<Point>> contours;

    findContours(
        colourMask.clone(),
        contours,
        RETR_EXTERNAL,
        CHAIN_APPROX_SIMPLE
    );

    const int EDGE_SEARCH_DISTANCE = 10;

    for (const auto& contour : contours) {

        if (contour.size() < 3)
            continue;

        Point2f initialCentre;
        float initialRadius;

        minEnclosingCircle(
            contour,
            initialCentre,
            initialRadius
        );

        if (initialRadius < MIN_CIRCLE_RADIUS ||
            initialRadius > MAX_CIRCLE_RADIUS)
            continue;

        // Search for Canny edges near the colour region
        Mat searchMask = Mat::zeros(image.size(), CV_8U);

        circle(
            searchMask,
            initialCentre,
            cvRound(initialRadius) + EDGE_SEARCH_DISTANCE,
            Scalar(255),
            FILLED
        );

        Mat nearbyEdges;
        bitwise_and(edges, searchMask, nearbyEdges);

        vector<Point> edgePoints;
        findNonZero(nearbyEdges, edgePoints);

        vector<Point2f> combinedPoints;

        for (const Point& p : contour)
            combinedPoints.emplace_back(p.x, p.y);

        for (const Point& p : edgePoints)
            combinedPoints.emplace_back(p.x, p.y);

        Point2f finalCentre;
        float finalRadius;

        minEnclosingCircle(
            combinedPoints,
            finalCentre,
            finalRadius
        );

        if (finalRadius < MIN_CIRCLE_RADIUS ||
            finalRadius > MAX_CIRCLE_RADIUS)
            continue;

        circle(result, finalCentre, cvRound(finalRadius),
               Scalar(0, 255, 0), 2);

        circle(result, finalCentre, 3,
               Scalar(0, 0, 255), -1);
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

    // 6. Detect selected colour with HSV thresholds only
    Mat colourMask = detectColour(hsv, selected);

    // 8. Overlay colour mask
    Mat colourOverlay = highlightColour(transformed, colourMask);

    // 9. Combine colour + canny
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

