#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <cmath>
#include <xlsxwriter.h>

using namespace cv;
using namespace std;

// =====================================================
// ADJUSTABLE PARAMETERS
// =====================================================

const int TABLE_NUMBER = 6;
const string SELECTED_COLOUR = "White";   // Change this only
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
// EXCEL EXPORT DATA
// =====================================================

struct Detection {
    string colour;
    Point2f centre;
};

struct GroundTruth {
    string ball;
    float row;
    float column;
};

vector<Detection> allDetections;

// The source Excel file is supplied as ball_ground_truth.csv.
vector<GroundTruth> loadGroundTruth(const string& imageName) {
    vector<GroundTruth> truth;
    ifstream file(FOLDER + "ball_ground_truth.csv");
    if (!file) {
        cerr << "Could not open ball_ground_truth.csv" << endl;
        return truth;
    }

    string line;
    getline(file, line);  // Skip CSV header
    while (getline(file, line)) {
        stringstream ss(line);
        string image, ball, rowText, columnText;
        getline(ss, image, ',');
        getline(ss, ball, ',');
        getline(ss, rowText, ',');
        getline(ss, columnText, ',');
        if (image != imageName || rowText.empty() || columnText.empty()) continue;
        try {
            truth.push_back({ball, stof(rowText), stof(columnText)});
        } catch (const exception&) { /* Ignore malformed rows */ }
    }
    return truth;
}

// Writes a REAL .xlsx workbook with reference locations and detections.
void exportBallLocations(const string& imageName,
                         const vector<Detection>& detections) {
    vector<GroundTruth> truth = loadGroundTruth(imageName);
    if (truth.empty()) {
        cerr << "No ground truth for " << imageName << endl;
        return;
    }

    string filename = FOLDER + "Ball_Comparison_Table" +
                      to_string(TABLE_NUMBER) + ".xlsx";
    lxw_workbook* workbook = workbook_new(filename.c_str());
    if (!workbook) {
        cerr << "Failed to create Excel workbook" << endl;
        return;
    }
    lxw_worksheet* sheet = workbook_add_worksheet(workbook, "Comparison");
    lxw_worksheet* extras = workbook_add_worksheet(workbook, "Extra Detections");
    lxw_format* header = workbook_add_format(workbook);
    format_set_bold(header);
    format_set_bg_color(header, 0xDCEAF7);
    format_set_border(header, LXW_BORDER_THIN);
    worksheet_set_column(sheet, 0, 0, 12, NULL);
    worksheet_set_column(sheet, 1, 7, 18, NULL);
    worksheet_set_column(extras, 0, 2, 20, NULL);

    const char* headings[] = {"Ball", "Actual Row (Y)", "Actual Column (X)",
        "Detected Row (Y)", "Detected Column (X)", "Detected Colour",
        "Position Error (px)", "Status"};
    for (int i = 0; i < 8; ++i)
        worksheet_write_string(sheet, 0, i, headings[i], header);
    worksheet_write_string(extras, 0, 0, "Detected Colour", header);
    worksheet_write_string(extras, 0, 1, "Row (Y)", header);
    worksheet_write_string(extras, 0, 2, "Column (X)", header);

    // One-to-one nearest-neighbour association, only within 50 pixels.
    // This is a POSITION match, not a confirmation of the ball's identity.
    vector<int> assigned(truth.size(), -1);
    vector<bool> used(detections.size(), false);
    while (true) {
        float best = 50.0f;
        int bestTruth = -1, bestDetection = -1;
        for (int i = 0; i < (int)truth.size(); ++i) {
            if (assigned[i] != -1) continue;
            for (int j = 0; j < (int)detections.size(); ++j) {
                if (used[j]) continue;
                float dx = detections[j].centre.x - truth[i].column;
                float dy = detections[j].centre.y - truth[i].row;
                float distance = sqrt(dx * dx + dy * dy);
                if (distance < best) {
                    best = distance;
                    bestTruth = i;
                    bestDetection = j;
                }
            }
        }
        if (bestTruth < 0) break;
        assigned[bestTruth] = bestDetection;
        used[bestDetection] = true;
    }

    for (int i = 0; i < (int)truth.size(); ++i) {
        int row = i + 1;
        worksheet_write_string(sheet, row, 0, truth[i].ball.c_str(), NULL);
        worksheet_write_number(sheet, row, 1, truth[i].row, NULL);
        worksheet_write_number(sheet, row, 2, truth[i].column, NULL);
        int j = assigned[i];
        if (j == -1) {
            worksheet_write_string(sheet, row, 7, "Not detected", NULL);
            continue;
        }
        const Detection& d = detections[j];
        worksheet_write_number(sheet, row, 3, d.centre.y, NULL);
        worksheet_write_number(sheet, row, 4, d.centre.x, NULL);
        worksheet_write_string(sheet, row, 5, d.colour.c_str(), NULL);
        string errorFormula = "=SQRT((D" + to_string(row + 1) + "-B" +
            to_string(row + 1) + ")^2+(E" + to_string(row + 1) + "-C" +
            to_string(row + 1) + ")^2)";
        worksheet_write_formula(sheet, row, 6, errorFormula.c_str(), NULL);
        worksheet_write_string(sheet, row, 7, "Position matched", NULL);
    }
    int extraRow = 1;
    for (int j = 0; j < (int)detections.size(); ++j) {
        if (used[j]) continue;
        worksheet_write_string(extras, extraRow, 0, detections[j].colour.c_str(), NULL);
        worksheet_write_number(extras, extraRow, 1, detections[j].centre.y, NULL);
        worksheet_write_number(extras, extraRow, 2, detections[j].centre.x, NULL);
        extraRow++;
    }
    worksheet_freeze_panes(sheet, 1, 0);
    lxw_error error = workbook_close(workbook);
    if (error == LXW_NO_ERROR)
        cout << "Excel saved: " << filename << endl;
    else
        cerr << "Excel write error: " << lxw_strerror(error) << endl;
}

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
                      const Mat& colourMask,
                       vector<Detection>* found = nullptr,
                       const string& detectedColour = SELECTED_COLOUR) {

    Mat result = image.clone();

    // =====================================================
    // WHITE BALL: Largest connected white region only
    // =====================================================

    if (detectedColour == "White") {

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

                if (found) found->push_back({"White", centre});

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

        if (found) found->push_back({detectedColour, finalCentre});
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

    // Export all colours, not just the colour selected for display.
    vector<Detection> detectedBalls;
    for (const auto& range : getColourRanges()) {
        // Secondary ranges are included by detectColour() already.
        if (range.name == "Red2" || range.name == "Green2") continue;
        Mat mask = detectColour(hsv, range);
        detectBallCircles(transformed, edges, mask, &detectedBalls, range.name);
    }
    // Remove near-identical detections found under multiple HSV colour names.
    vector<Detection> uniqueDetections;
    for (const auto& item : detectedBalls) {
        bool duplicate = false;
        for (const auto& earlier : uniqueDetections) {
            if (norm(item.centre - earlier.centre) < 15.0f) {
                duplicate = true;
                break;
            }
        }
        if (!duplicate) uniqueDetections.push_back(item);
    }
    exportBallLocations(filename, uniqueDetections);

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

