/* INCLUDES FOR THIS PROJECT */
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <limits>
#include <numeric>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include "dataStructures.h"
#include "matching2D.hpp"

using namespace std;

/* MAIN PROGRAM */
int main(int argc, const char *argv[])
{

    /* INIT VARIABLES AND DATA STRUCTURES */

    // data location
    string dataPath = "../";

    // camera
    string imgBasePath = dataPath + "images/";
    string imgPrefix = "KITTI/2011_09_26/image_00/data/000000"; // left camera, color
    string imgFileType = ".png";
    int imgStartIndex = 0; // first file index to load (assumes Lidar and camera names have identical naming convention)
    int imgEndIndex = 9;   // last file index to load
    int imgFillWidth = 4;  // no. of digits which make up the file index (e.g. img-0001.png)

    // misc
    int dataBufferSize = 2;       // no. of images which are held in memory (ring buffer) at the same time
    vector<DataFrame> dataBuffer; // list of data frames which are held in memory at the same time

    // Visualization settings
    bool matchingVis = true;            // visualize matching results
    bool detectorVis = false;    // visualize detector results

    //////////////////////////////////////////////
    // Detector, Descriptor and Matching settings
    // Detector settings:
    string detectorName = "AKAZE"; //SHITOMASI, HARRIS, FAST, BRISK, ORB, AKAZE, SIFT

    // Descriptor settings
    // Note: AKAZE Descriptor can be used only with AKAZE keypoints!
    string descriptorName = "ORB"; // BRISK (binary), BRIEF (binary), ORB (binary), FREAK (binary), AKAZE (binary), SIFT (HOG)

    // Matching settings
    string matcherType = "MAT_BF";        // MAT_BF, MAT_FLANN
    string descriptorType = (descriptorName.compare("SIFT") == 0) ? "DES_HOG" : "DES_BINARY"; // DES_BINARY, DES_HOG
    string selectorType = "SEL_KNN";       // SEL_NN, SEL_KNN
    //////////////////////////////////////////////

    // Focus on vehicle ahead
    bool bFocusOnVehicle = true;

    // Limit the number of keypoints for debugging
    bool bLimitKpts = false;

    // Keypoint statistics
    std::vector<unsigned int> statNumKeypoints;
    std::vector<unsigned int> statNumMatchedKeypoints;
    std::vector<float> statKeypointNeighborSize;
    std::vector<double> statCompTimeDetector;
    std::vector<double> statCompTimeDescriptor;

    /* MAIN LOOP OVER ALL IMAGES */

    for (size_t imgIndex = 0; imgIndex <= imgEndIndex - imgStartIndex; imgIndex++)
    {
        /* LOAD IMAGE INTO BUFFER */

        // assemble filenames for current index
        ostringstream imgNumber;
        imgNumber << setfill('0') << setw(imgFillWidth) << imgStartIndex + imgIndex;
        string imgFullFilename = imgBasePath + imgPrefix + imgNumber.str() + imgFileType;

        // load image from file and convert to grayscale
        cv::Mat img, imgGray;
        img = cv::imread(imgFullFilename);
        cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);

        // push image into data frame buffer
        DataFrame frame;
        frame.cameraImg = imgGray;
        dataBuffer.push_back(frame);

        // Remove the first (oldest) element from the vector
        if(dataBuffer.size() > dataBufferSize) {
            dataBuffer.erase(dataBuffer.begin());
        }

        cout << "#1 : LOAD IMAGE INTO BUFFER done" << endl;

        /* DETECT IMAGE KEYPOINTS */
        // extract 2D keypoints from current image
        vector<cv::KeyPoint> keypoints; // create empty feature list for current image

        double t = (double)cv::getTickCount();

        if (detectorName.compare("SHITOMASI") == 0)
        {
            detKeypointsShiTomasi(keypoints, imgGray, detectorVis);
        }
        else if (detectorName.compare("HARRIS") == 0)
        {
            detKeypointsHarris(keypoints,imgGray,detectorVis);
        }
        else if (detectorName.compare("FAST") == 0)
        {
            detKeypointsModern(keypoints, imgGray, "FAST", detectorVis);
        }
        else if (detectorName.compare("BRISK") == 0)
        {
            detKeypointsModern(keypoints, imgGray, "BRISK", detectorVis);
        }
        else if (detectorName.compare("ORB") == 0)
        {
            detKeypointsModern(keypoints, imgGray, "ORB", detectorVis);
        }
        else if (detectorName.compare("AKAZE") == 0)
        {
            detKeypointsModern(keypoints, imgGray, "AKAZE", detectorVis);
        }
        else if (detectorName.compare("SIFT") == 0)
        {
            detKeypointsModern(keypoints, imgGray, "SIFT", detectorVis);
        }

        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();

        // Only keep keypoints on the preceding vehicle
        cv::Rect vehicleRect(535, 180, 180, 150);
        if (bFocusOnVehicle)
        {
            for(auto it=keypoints.begin(); it!=keypoints.end();) {
                if(!vehicleRect.contains(it->pt))  {
                    it = keypoints.erase(it);
                }
                else {
                    ++it;
                }
            }
        }
        statNumKeypoints.push_back(keypoints.size());


        // optional : limit number of keypoints (helpful for debugging and learning)
        if (bLimitKpts)
        {
            int maxKeypoints = 50;

            if (detectorName.compare("SHITOMASI") == 0)
            { // there is no response info, so keep the first 50 as they are sorted in descending quality order
                keypoints.erase(keypoints.begin() + maxKeypoints, keypoints.end());
            }
            cv::KeyPointsFilter::retainBest(keypoints, maxKeypoints);
            cout << " NOTE: Keypoints have been limited!" << endl;
        }

        // push keypoints and descriptor for current frame to end of data buffer
        (dataBuffer.end() - 1)->keypoints = keypoints;
        cout << "#2 : DETECT KEYPOINTS (" << detectorName << ") DONE in " << 1000 * t / 1.0 << "ms"  << endl;
        statCompTimeDetector.push_back(1000 * t / 1.0);

        // Extract keypoint descriptors
        cv::Mat descriptors;
        t = (double)cv::getTickCount();
        descKeypoints((dataBuffer.end() - 1)->keypoints, (dataBuffer.end() - 1)->cameraImg, descriptors, descriptorName);
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        statCompTimeDescriptor.push_back(1000 * t / 1.0);


        // push descriptors for current frame to end of data buffer
        (dataBuffer.end() - 1)->descriptors = descriptors;

        cout << "#3 : EXTRACT DESCRIPTORS (" << descriptorName << ") DONE in " << 1000 * t / 1.0 << "ms" << endl;

        if (dataBuffer.size() > 1) // wait until at least two images have been processed
        {

            /* MATCH KEYPOINT DESCRIPTORS */

            vector<cv::DMatch> matches;

            matchDescriptors((dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints,
                             (dataBuffer.end() - 2)->descriptors, (dataBuffer.end() - 1)->descriptors,
                             matches, descriptorType, matcherType, selectorType);

            statNumMatchedKeypoints.push_back(matches.size());

            // store matches in current data frame
            (dataBuffer.end() - 1)->kptMatches = matches;

            cout << "#4 : MATCH KEYPOINT DESCRIPTORS done" << endl;

            // visualize matches between current and previous image
            if (matchingVis)
            {
                cv::Mat matchImg = ((dataBuffer.end() - 1)->cameraImg).clone();
                cv::drawMatches((dataBuffer.end() - 2)->cameraImg, (dataBuffer.end() - 2)->keypoints,
                                (dataBuffer.end() - 1)->cameraImg, (dataBuffer.end() - 1)->keypoints,
                                matches, matchImg,
                                cv::Scalar::all(-1), cv::Scalar::all(-1),
                                vector<char>(), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

                string windowName = "Matching keypoints between two camera images";
                cv::namedWindow(windowName, 7);
                cv::imshow(windowName, matchImg);
                cout << "Press key to continue to next image" << endl;
                cv::waitKey(0); // wait for key to be pressed
            }
        }

    } // eof loop over all images

    // Print the statistics
    unsigned int statNumKeypointsTotal = std::accumulate( statNumKeypoints.begin(), statNumKeypoints.end(), 0.0);
    unsigned int statNumMatchedKeypointsTotal = std::accumulate( statNumMatchedKeypoints.begin(), statNumMatchedKeypoints.end(), 0.0);
    double statCompTimeDetectorAvg = std::accumulate( statCompTimeDetector.begin(), statCompTimeDetector.end(), 0.0) / statCompTimeDetector.size();
    double statCompTimeDescriptorAvg = std::accumulate( statCompTimeDescriptor.begin(), statCompTimeDescriptor.end(), 0.0) / statCompTimeDescriptor.size();
    std::cout << "Detector: " << detectorName << " Descriptor: " << descriptorName << std::endl;
    std::cout << "Num keypoints in car area: " << statNumKeypointsTotal;
    std::cout << "\t\tNum matched keypoints: " << statNumMatchedKeypointsTotal;
    std::cout << "\t\tAvg time detector: " << statCompTimeDetectorAvg;
    std::cout << "\t\tAvg time descriptor: " << statCompTimeDescriptorAvg;

    return 0;
}
