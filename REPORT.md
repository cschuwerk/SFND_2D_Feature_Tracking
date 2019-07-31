# SFND 2D Feature Tracking

## Mid-Term Report

### MP.1 Data Buffer
Remove the first element of the vector, if the length of the vector is greater than the desired length, see MidTermProject_Camera_Student.cpp line 97-99

### MP.2 Keypoint Detection
Use the string "detectorName" MidTermProject_Camera_Student.cpp line 50 to select the detector

### MP.3 Keypoint Removal
use the cv::Rect::contains(...) method to check if the point lies inside, see lines 142-152 in MidTermProject_Camera_Student.cpp

### MP.4 Keypoint Descriptors
Use the string "descriptorName" MidTermProject_Camera_Student.cpp line 54 to select the detector

### MP.5 / MP.6 Keypoint Matching
Use the strings matcherType (line 57) and selectorType (line 59) to select the different types. Also see inline comments.

## MP.7 / MP.8 / MP.9
Spreadsheet: https://docs.google.com/spreadsheets/d/1TBnmddn5DbPttQTVZIm9n1Z5tD7-qPWA2pFfM1eqjDs/edit?usp=sharing

### Conclusions

- BRISK detector/descriptor is the slowest
- SIFT, AKAZE detectors/descriptor are too slow to run at the rate of the camera
- FAST (detector) + BRIEF (descriptor) and FAST+ORB are the fastest combinations
- HARRIS detector is also quite fast, but provides only a limited number of keypoints
- Regarding time, SHITOMASI+BRIEF/ORB, HARRIS+BRIEF/ORB, FAST+BRIEF/ORB, ORB+BRIEF/ORB are okay
- SHITOMASI and FAST detect keypoints quite well distributed over the backside of the car

### Recommendation:

1. FAST + BRIEF
2. FAST + ORB
3. SHITOMASI + BRIEF or ORB
