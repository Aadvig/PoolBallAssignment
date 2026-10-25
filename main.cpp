
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>

using namespace cv;
using namespace std;

// =====================================================
// PARAMETERS
// =====================================================

const int TABLE_NUMBER = 1;

const string FOLDER =
    "/Users/aadyavig/CLionProjects/PoolBallAssignment/";

const int SPATIAL_RADIUS = 15;
const int COLOUR_RADIUS = 25;
const int PYRAMID_LEVEL = 1;

const double DISTANCE_THRESHOLD = 0.3;
const int MIN_REGION_AREA = 50;

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
        original, transformed
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
// MEAN-SHIFT SEGMENTATION
// =====================================================

Mat segmentImage(const Mat& image) {

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
// FOREGROUND MASK
// =====================================================

Mat segmentBalls(const Mat& image) {

    Mat lab;
    cvtColor(image, lab, COLOR_BGR2Lab);

    // Estimate felt colour from central table region
    Rect sampleRegion(
        image.cols / 4,
        image.rows / 4,
        image.cols / 2,
        image.rows / 2
    );

    Scalar backgroundColour = mean(
        lab(sampleRegion)
    );

    Mat background(
        lab.size(),
        lab.type(),
        backgroundColour
    );

    Mat difference;
    absdiff(lab, background, difference);

    vector<Mat> channels;
    split(difference, channels);

    Mat distance;
    magnitude(
        channels[1],
        channels[2],
        distance
    );

    Mat mask;
    threshold(
        distance,
        mask,
        25,
        255,
        THRESH_BINARY
    );

    mask.convertTo(mask, CV_8U);

    // Remove noise
    Mat kernel = getStructuringElement(
        MORPH_ELLIPSE,
        Size(5, 5)
    );

    morphologyEx(
        mask,
        mask,
        MORPH_OPEN,
        kernel
    );

    morphologyEx(
        mask,
        mask,
        MORPH_CLOSE,
        kernel
    );

    return mask;
}

// =====================================================
// WATERSHED REGION DETECTION
// =====================================================

vector<Point2f> detectRegions(
    const Mat& image,
    const Mat& binaryMask,
    Mat& watershedDisplay
) {

    vector<Point2f> centres;

    // Distance transform
    Mat distance;

    distanceTransform(
        binaryMask,
        distance,
        DIST_L2,
        5
    );

    normalize(
        distance,
        distance,
        0,
        1.0,
        NORM_MINMAX
    );

    // Identify foreground markers
    Mat foreground;

    threshold(
        distance,
        foreground,
        DISTANCE_THRESHOLD,
        1.0,
        THRESH_BINARY
    );

    Mat foreground8U;
    foreground.convertTo(
        foreground8U,
        CV_8U,
        255
    );

    // Label connected foreground regions
    Mat markers;

    connectedComponents(
        foreground8U,
        markers
    );

    markers += 1;

    // Mark unknown regions
    Mat unknown;
    subtract(
        binaryMask,
        foreground8U,
        unknown
    );

    markers.setTo(0, unknown);

    // Watershed segmentation
    watershed(image, markers);

    watershedDisplay = image.clone();

    double minVal, maxVal;

    minMaxLoc(
        markers,
        &minVal,
        &maxVal
    );

    // Extract each region
    for (int label = 2;
         label <= static_cast<int>(maxVal);
         label++) {

        Mat regionMask = markers == label;

        int area = countNonZero(regionMask);

        if (area < MIN_REGION_AREA)
            continue;

        Moments m = moments(
            regionMask,
            true
        );

        if (m.m00 == 0)
            continue;

        Point2f centre(
            static_cast<float>(m.m10 / m.m00),
            static_cast<float>(m.m01 / m.m00)
        );

        centres.push_back(centre);

        // Draw region boundary
        vector<vector<Point>> contours;

        findContours(
            regionMask.clone(),
            contours,
            RETR_EXTERNAL,
            CHAIN_APPROX_SIMPLE
        );

        drawContours(
            watershedDisplay,
            contours,
            -1,
            Scalar(0, 255, 0),
            2
        );

        // Draw centroid
        circle(
            watershedDisplay,
            centre,
            4,
            Scalar(0, 0, 255),
            FILLED
        );
    }

    // Mark watershed boundaries
    Mat boundaries = markers == -1;

    watershedDisplay.setTo(
        Scalar(255, 0, 0),
        boundaries
    );

    return centres;
}

// =====================================================
// MAIN
// =====================================================

int main() {

    string filename =
        "Table" + to_string(TABLE_NUMBER) + ".jpeg";

    Mat image = imread(
        FOLDER + filename
    );

    if (image.empty()) {
        cout << "Error loading: "
             << filename << endl;
        return -1;
    }

    // 1. Perspective transformation
    Mat transformed =
        perspectiveTransformTable(image);

    // 2. Mean-shift segmentation
    Mat segmented =
        segmentImage(transformed);

    // 3. Foreground segmentation
    Mat binaryMask =
        segmentBalls(segmented);

    // 4. Watershed region detection
    Mat watershedDisplay;

    vector<Point2f> centres =
        detectRegions(
            segmented,
            binaryMask,
            watershedDisplay
        );

    // 5. Output detected centres
    cout << "Detected regions: "
         << centres.size() << endl;

    for (size_t i = 0; i < centres.size(); i++) {

        cout << "Region " << i + 1
             << ": " << centres[i]
             << endl;
    }

    // 6. Display intermediate results
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
        "Binary Mask",
        binaryMask
    );

    imshow(
        "Watershed Segmentation",
        watershedDisplay
    );

    // 7. Save output
    imwrite(
        FOLDER + "WatershedResult.png",
        watershedDisplay
    );

    waitKey(0);
    destroyAllWindows();

    return 0;
}
