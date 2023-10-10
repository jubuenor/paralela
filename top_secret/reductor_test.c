#include <opencv2/opencv.hpp>
#include <omp.h>
#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, char *argv[])
{
    // Validate command-line arguments
    if (argc < 3) {
        cerr << "Usage: " << argv[0] << " <input_video> <output_video>" << endl;
        return 1;
    }

    string input = argv[1];
    string output = argv[2];

    VideoCapture cap(input);
    if (!cap.isOpened()) {
        cerr << "Error opening video stream or file" << endl;
        return 1;
    }

    int fps = static_cast<int>(cap.get(CAP_PROP_FPS));
    int frame_width = static_cast<int>(cap.get(CAP_PROP_FRAME_WIDTH));
    int frame_height = static_cast<int>(cap.get(CAP_PROP_FRAME_HEIGHT));
    int frameCount = static_cast<int>(cap.get(CAP_PROP_FRAME_COUNT));

    int fourcc = VideoWriter::fourcc('m', 'p', '4', 'v');
    VideoWriter video(output, fourcc, fps, Size(frame_width / 3, frame_height / 3));

    double init = omp_get_wtime();

    Mat frame;
    while (cap.read(frame)) {
        Mat newFrame = Mat::zeros(frame.rows / 3, frame.cols / 3, frame.type());

        for (int i = 0; i < frame.rows; i += 3) {
            for (int j = 0; j < frame.cols; j += 3) {
                Scalar color = mean(frame(Rect(j, i, 3, 3)));
                newFrame.at<Vec3b>(i / 3, j / 3) = Vec3b(color[0], color[1], color[2]);
            }
        }

        video.write(newFrame);
    }

    double _end = omp_get_wtime();
    cout << "Total time: " << _end - init << " seconds" << endl;

    cap.release();
    video.release();

    return 0;
}

