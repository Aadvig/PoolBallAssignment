
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <deque>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <cmath>
#include <limits>
#include <iomanip>
#include <stdexcept>
#include <xlsxwriter.h>

using namespace cv;
using namespace std;

// =====================================================
// ADJUSTABLE PARAMETERS
// =====================================================

const int TABLE_NUMBER = 1;
// Maximum centre distance for a correct one-to-one detection match.
const double ACCURACY_TOLERANCE_PX = 50.0;
const string SELECTED_COLOUR = "Blue";   // Change this only
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
// EXCEL EXPORT: ONE WORKBOOK, ONE SHEET, ALL TABLES
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

vector<GroundTruth> loadGroundTruth(const string& imageName) {
    vector<GroundTruth> truth;
    ifstream file(FOLDER + "ball_ground_truth_second_correction.csv");
    if (!file) {
        cerr << "Could not open " << FOLDER << "ball_ground_truth_second_correction.csv" << endl;
        return truth;
    }

    string line;
    getline(file, line); // Header: Image,Ball,Row,Column
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
        } catch (const exception&) {
            cerr << "Ignoring invalid ground-truth entry: " << line << endl;
        }
    }
    if (truth.empty()) {
        cerr << "No ground truth found for " << imageName << endl;
    }
    return truth;
}

// All red dots are retained. Only one-to-one pairs within 50 px count as TP.
// A maximum-cardinality, minimum-distance matching prevents a distant dot
// from consuming a ground-truth ball that another detection can match correctly.
struct TableMetrics {
    string image;
    int groundTruth = 0;
    int redDots = 0;
    int accurate = 0;         // TP
    int missing = 0;          // FN
    int falseDetections = 0;  // FP
    double sumError = 0.0;   // Errors of valid (<= tolerance) matches only
    int errorCount = 0;
};

double safeDivide(double a, double b) { return b > 0.0 ? a / b : 0.0; }
double precision(const TableMetrics& m) { return safeDivide(m.accurate, m.redDots); }
double recall(const TableMetrics& m) { return safeDivide(m.accurate, m.groundTruth); }
double f1Score(const TableMetrics& m) {
    return safeDivide(2.0 * m.accurate,
                      2.0 * m.accurate + m.falseDetections + m.missing);
}

string coordinateText(double x, double y) {
    ostringstream ss;
    ss << fixed << setprecision(1) << "(" << x << ", " << y << ")";
    return ss.str();
}

struct MatchEdge {
    int to, rev, capacity;
    double cost;
};

// Successive shortest augmenting paths on a residual network.
// Each augment increases the number of valid matches by one; augmenting
// until no path exists maximizes TP, with minimum total error for that TP.
vector<int> matchWithinTolerance(const vector<GroundTruth>& truth,
                                 const vector<Detection>& detections) {
    const int n = (int)truth.size(), m = (int)detections.size();
    const int source = n + m, sink = source + 1, nodes = sink + 1;
    vector<vector<MatchEdge>> graph(nodes);
    auto addEdge = [&](int u, int v, int cap, double cost) {
        graph[u].push_back({v, (int)graph[v].size(), cap, cost});
        graph[v].push_back({u, (int)graph[u].size() - 1, 0, -cost});
    };
    for (int i = 0; i < n; ++i) addEdge(source, i, 1, 0);
    for (int j = 0; j < m; ++j) addEdge(n + j, sink, 1, 0);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            double d = norm(detections[j].centre -
                            Point2f(truth[i].column, truth[i].row));
            if (d <= ACCURACY_TOLERANCE_PX) addEdge(i, n + j, 1, d);
        }
    }

    while (true) {
        vector<double> dist(nodes, numeric_limits<double>::infinity());
        vector<int> prevNode(nodes, -1), prevEdge(nodes, -1);
        vector<bool> queued(nodes, false);
        deque<int> q;
        dist[source] = 0;
        q.push_back(source);
        queued[source] = true;
        while (!q.empty()) {
            int u = q.front(); q.pop_front(); queued[u] = false;
            for (int k = 0; k < (int)graph[u].size(); ++k) {
                const auto& e = graph[u][k];
                if (e.capacity <= 0 || dist[u] + e.cost >= dist[e.to] - 1e-9)
                    continue;
                dist[e.to] = dist[u] + e.cost;
                prevNode[e.to] = u;
                prevEdge[e.to] = k;
                if (!queued[e.to]) {
                    queued[e.to] = true;
                    q.push_back(e.to);
                }
            }
        }
        if (prevNode[sink] == -1) break;
        for (int v = sink; v != source; v = prevNode[v]) {
            MatchEdge& e = graph[prevNode[v]][prevEdge[v]];
            e.capacity -= 1;
            graph[v][e.rev].capacity += 1;
        }
    }

    vector<int> assignment(n, -1);
    for (int i = 0; i < n; ++i)
        for (const auto& e : graph[i])
            if (e.to >= n && e.to < n + m && e.capacity == 0)
                assignment[i] = e.to - n;
    return assignment;
}

TableMetrics appendComparison(lxw_worksheet* sheet, int& nextRow,
                              const string& imageName,
                              const vector<Detection>& detections) {
    vector<GroundTruth> truth = loadGroundTruth(imageName);
    if (truth.empty()) throw runtime_error("Missing ground truth for " + imageName);
    TableMetrics stats;
    stats.image = imageName;
    stats.groundTruth = (int)truth.size();
    stats.redDots = (int)detections.size();

    vector<int> assigned = matchWithinTolerance(truth, detections);
    vector<bool> used(detections.size(), false);
    for (int i = 0; i < (int)truth.size(); ++i) {
        int row = nextRow++;
        worksheet_write_string(sheet, row, 0, truth[i].ball.c_str(), NULL);
        string actual = coordinateText(truth[i].column, truth[i].row);
        worksheet_write_string(sheet, row, 1, actual.c_str(), NULL);
        int j = assigned[i];
        if (j < 0) {
            worksheet_write_string(sheet, row, 2, "Not detected within tolerance", NULL);
            worksheet_write_string(sheet, row, 4, "FN", NULL);
            continue;
        }
        used[j] = true;
        const Detection& d = detections[j];
        string detected = coordinateText(d.centre.x, d.centre.y);
        worksheet_write_string(sheet, row, 2, detected.c_str(), NULL);
        double distance = norm(d.centre - Point2f(truth[i].column, truth[i].row));
        worksheet_write_number(sheet, row, 3, distance, NULL);
        worksheet_write_string(sheet, row, 4, "TP", NULL);
        stats.sumError += distance;
        stats.errorCount++;
        stats.accurate++;
    }

    // Every unmatched detection is an FP, including distant or duplicate dots.
    for (int j = 0; j < (int)detections.size(); ++j) {
        if (used[j]) continue;
        int row = nextRow++;
        worksheet_write_string(sheet, row, 0, "Extra", NULL);
        string detected = coordinateText(detections[j].centre.x, detections[j].centre.y);
        worksheet_write_string(sheet, row, 2, detected.c_str(), NULL);
        worksheet_write_string(sheet, row, 4, "FP", NULL);
    }
    stats.missing = stats.groundTruth - stats.accurate;
    stats.falseDetections = stats.redDots - stats.accurate;
    return stats;
}

// Column labels in the two-sheet output.
void writeHeader(lxw_worksheet* ws, int row,
                 const vector<string>& labels, lxw_format* header) {
    for (int c = 0; c < (int)labels.size(); ++c)
        worksheet_write_string(ws, row, c, labels[c].c_str(), header);
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
    const string groundTruthPath = FOLDER + "ball_ground_truth_second_correction.csv";
    if (!ifstream(groundTruthPath)) {
        cerr << "Missing ground truth CSV: " << groundTruthPath << endl;
        return 1;
    }
    const string excelPath = FOLDER + "Ball_Comparison_Second_Correction.xlsx";
    lxw_workbook* workbook = workbook_new(excelPath.c_str());
    if (!workbook) {
        cerr << "Cannot create " << excelPath << endl;
        return 1;
    }

    lxw_worksheet* positions = workbook_add_worksheet(workbook, "Ball Positions");
    lxw_worksheet* summary = workbook_add_worksheet(workbook, "Performance Summary");
    lxw_format* header = workbook_add_format(workbook);
    format_set_bold(header);
    format_set_bg_color(header, 0xDCEAF7);
    format_set_border(header, LXW_BORDER_THIN);

    lxw_format* numberFmt = workbook_add_format(workbook);
    format_set_num_format(numberFmt, "0.0");
    lxw_format* percentFmt = workbook_add_format(workbook);
    format_set_num_format(percentFmt, "0.0%");

    worksheet_set_column(positions, 0, 0, 16, NULL);
    worksheet_set_column(positions, 1, 2, 32, NULL);
    worksheet_set_column(positions, 4, 4, 14, NULL);
    worksheet_set_column(positions, 3, 3, 18, numberFmt);
    worksheet_set_column(summary, 0, 0, 18, NULL);
    worksheet_set_column(summary, 1, 5, 17, NULL);
    worksheet_set_column(summary, 6, 11, 19, NULL);
    worksheet_freeze_panes(summary, 1, 1);

    writeHeader(summary, 0, {"Image", "Ground Truth", "Detections",
        "TP", "FN", "FP", "Detection Accuracy / Recall",
        "Precision", "F1 Score", "Mean TP Error (px)",
        "Tolerance (px)"}, header);

    int nextRow = 0;
    int summaryRow = 1;
    TableMetrics total;
    total.image = "Overall";

    for (int table = 1; table <= 6; ++table) {
        string filename = "Table" + to_string(table) + ".jpeg";
        Mat image = imread(FOLDER + filename);
        if (image.empty()) {
            cerr << "Skipping missing image: " << filename << endl;
            continue;
        }

        Mat transformed = perspectiveTransformTable(image);
        Mat segmented = applyMeanShift(transformed);
        Mat edges = detectEdges(segmented);
        Mat hsv;
        cvtColor(segmented, hsv, COLOR_BGR2HSV);

        vector<Detection> detectedBalls;
        for (const auto& range : getColourRanges()) {
            if (range.name == "Red2" || range.name == "Green2") continue;
            Mat mask = detectColour(hsv, range);
            // Captures the exact centre used to draw every red dot.
            detectBallCircles(transformed, edges, mask, &detectedBalls, range.name);
        }

        worksheet_write_string(positions, nextRow++, 0, filename.c_str(), header);
        writeHeader(positions, nextRow++, {"Ball", "Ground Truth (X,Y)",
                                      "Detected (X,Y)", "Error (px)", "Outcome"}, header);
        TableMetrics m = appendComparison(positions, nextRow, filename, detectedBalls);
        nextRow++; // Separator between tables.

        worksheet_write_string(summary, summaryRow, 0, filename.c_str(), NULL);
        worksheet_write_number(summary, summaryRow, 1, m.groundTruth, NULL);
        worksheet_write_number(summary, summaryRow, 2, m.redDots, NULL);
        worksheet_write_number(summary, summaryRow, 3, m.accurate, NULL);
        worksheet_write_number(summary, summaryRow, 4, m.missing, NULL);
        worksheet_write_number(summary, summaryRow, 5, m.falseDetections, NULL);
        worksheet_write_number(summary, summaryRow, 6, recall(m), percentFmt);
        worksheet_write_number(summary, summaryRow, 7, precision(m), percentFmt);
        worksheet_write_number(summary, summaryRow, 8, f1Score(m), percentFmt);
        if (m.errorCount)
            worksheet_write_number(summary, summaryRow, 9,
                                   m.sumError / m.errorCount, numberFmt);
        worksheet_write_number(summary, summaryRow, 10, ACCURACY_TOLERANCE_PX, NULL);
        ++summaryRow;

        total.groundTruth += m.groundTruth;
        total.redDots += m.redDots;
        total.accurate += m.accurate;
        total.missing += m.missing;
        total.falseDetections += m.falseDetections;
        total.sumError += m.sumError;
        total.errorCount += m.errorCount;
        cout << filename << ": " << m.redDots << " red dots exported" << endl;

        if (table == TABLE_NUMBER) {
            ColourRange selected = getSelectedColourRange(SELECTED_COLOUR);
            Mat mask = detectColour(hsv, selected);
            Mat colourOverlay = highlightColour(transformed, mask);
            Mat combined = detectBallCircles(transformed, edges, mask,
                                              nullptr, selected.name);
            imshow("1 - Perspective Transform", transformed);
            imshow("2 - Mean Shift", segmented);
            imshow("3 - Colour Overlay: " + selected.name, colourOverlay);
            imshow("4 - Canny on Mean Shift", edges);
            imshow("5 - Combined Detection: " + selected.name, combined);
        }
    }

    worksheet_write_string(summary, summaryRow, 0, "Overall", header);
    worksheet_write_number(summary, summaryRow, 1, total.groundTruth, header);
    worksheet_write_number(summary, summaryRow, 2, total.redDots, header);
    worksheet_write_number(summary, summaryRow, 3, total.accurate, header);
    worksheet_write_number(summary, summaryRow, 4, total.missing, header);
    worksheet_write_number(summary, summaryRow, 5, total.falseDetections, header);
    worksheet_write_number(summary, summaryRow, 6, recall(total), percentFmt);
    worksheet_write_number(summary, summaryRow, 7, precision(total), percentFmt);
    worksheet_write_number(summary, summaryRow, 8, f1Score(total), percentFmt);
    if (total.errorCount)
        worksheet_write_number(summary, summaryRow, 9,
                               total.sumError / total.errorCount, numberFmt);
    worksheet_write_number(summary, summaryRow, 10, ACCURACY_TOLERANCE_PX, NULL);

    worksheet_write_string(summary, summaryRow + 2, 0,
        "TP = unique detection within 50px; FN = GT - TP; FP = detections - TP.", NULL);
    worksheet_write_string(summary, summaryRow + 3, 0,
        "Detection accuracy (success rate) = recall = TP / GT.", NULL);
    worksheet_write_string(summary, summaryRow + 4, 0,
        "Precision = TP / detections; F1 = 2TP / (2TP + FP + FN).", NULL);
    worksheet_write_string(summary, summaryRow + 5, 0,
        "Mean TP Error = mean pixel distance for correctly matched pairs only.", NULL);
    worksheet_write_string(summary, summaryRow + 6, 0,
        "Matching is positional; it does NOT validate ball colour or identity.", NULL);
    worksheet_write_string(summary, summaryRow + 7, 0,
        "Classification accuracy requires true negatives and is not applicable here.", NULL);

    lxw_error error = workbook_close(workbook);
    if (error == LXW_NO_ERROR)
        cout << "Saved TWO worksheets: " << excelPath << endl;
    else
        cerr << "Excel error: " << lxw_strerror(error) << endl;

    waitKey(0);
    destroyAllWindows();
    return error == LXW_NO_ERROR ? 0 : 1;
}


