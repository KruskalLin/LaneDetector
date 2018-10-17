#include <iostream>
#include <cmath>
#include <cstring>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "Detector.h"
#define DIST_MAX 1000000
#define DIST_MIN -1000000
using namespace std;
using namespace cv;



int main() {
    int road_horizon = 120;
    Detector detector(road_horizon);
    KalmanFilter kf(16, 8, 0);
    Mat state (16, 1, CV_32FC1);
    Mat processNoise(16, 1, CV_32F);
    Mat measurement = Mat::zeros(8, 1, CV_32F);
    kf.transitionMatrix = (Mat_<float>(16, 16) <<
            1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,
            0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,
            0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,
            0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,
            0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,
            0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,
            0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,
            0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,
            0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,
            0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,
            0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,
            0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1
    );
    setIdentity(kf.measurementMatrix);
    setIdentity(kf.processNoiseCov, Scalar::all(1e-4));
    setIdentity(kf.measurementNoiseCov, Scalar::all(10));
    setIdentity(kf.errorCovPost, Scalar::all(1));

    String video_path = "./video.avi";
    VideoCapture capture(video_path);

    bool first = false;


    while(capture.isOpened()){
        Mat frame;
        capture.read(frame);


        Lanes l= detector.detect(frame);
        if(!first) {
            kf.statePost = (Mat_<float>(16, 1) <<
                    (float)l.getLeft()[0],
                    (float)l.getLeft()[1],
                    (float)l.getLeft()[2],
                    (float)l.getLeft()[3],
                    (float)l.getRight()[0],
                    (float)l.getRight()[1],
                    (float)l.getRight()[2],
                    (float)l.getRight()[3],
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0);

            first = true;
        }
        state = kf.predict();
        if(!l.isEmpty()) {
            line(frame, Point(state.at<float>(0), state.at<float>(1)), Point(state.at<float>(2), state.at<float>(3)), Scalar(0, 0, 255), 3);
            line(frame, Point(state.at<float>(4), state.at<float>(5)), Point(state.at<float>(6), state.at<float>(7)), Scalar(0, 0, 255), 3);
        }

        measurement.at<float>(0) = l.getLeft()[0];
        measurement.at<float>(1) = l.getLeft()[1];
        measurement.at<float>(2) = l.getLeft()[2];
        measurement.at<float>(3) = l.getLeft()[3];
        measurement.at<float>(4) = l.getRight()[0];
        measurement.at<float>(5) = l.getRight()[1];
        measurement.at<float>(6) = l.getRight()[2];
        measurement.at<float>(7) = l.getRight()[3];
        kf.correct(measurement);
        cout<<state<<endl;

        imshow("img", frame);
        if(!waitKey(1))
            break;
    }

    return 0;
}
