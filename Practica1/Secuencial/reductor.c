#include <opencv2/opencv.hpp>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

using namespace cv;
using namespace std;

long double init, _end;
long double total_time;

int main()
{

    VideoCapture cap("../japan.mp4");
    if (!cap.isOpened())
    {
        cout << "Error opening video stream or file" << endl;
        return 0;
    }

    int fps = cap.get(CAP_PROP_FPS);
    int fourcc = cap.get(CAP_PROP_FOURCC);
    int frameCount = cap.get(CAP_PROP_FRAME_COUNT);
    cout << frameCount << endl;

    VideoWriter video("outcpp.avi", VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, Size(640, 360));

    init = omp_get_wtime();

    for (int n = 0; n < frameCount; n++)
    {
        Mat frames;
        cap >> frames;

        // cout << frames.size() << endl;
        Mat newFrame = Mat::zeros(frames.size() / 3, frames.type());

        for (int i = 0; i < frames.rows; i += 3)
        {
            for (int j = 0; j < frames.cols; j += 3)
            {

                double blue = 0;
                double green = 0;
                double red = 0;
                for (int ik = 0; ik < 3; ik++)
                {
                    for (int jk = 0; jk < 3; jk++)
                    {
                        blue += frames.at<Vec3b>(i + ik, j + jk)[0];
                        green += frames.at<Vec3b>(i + ik, j + jk)[1];
                        red += frames.at<Vec3b>(i + ik, j + jk)[2];
                    }
                }

                red /= 9;
                green /= 9;
                blue /= 9;
                Vec3b color = Vec3b(blue, green, red);
                newFrame.at<Vec3b>(i / 3, j / 3) = color;
            }
        }
        // cout << newFrame.size() << endl;
        video.write(newFrame);
        // imshow("Frame", newFrame);
        // char c = (char)waitKey(1);
        // if (c == 27)
        // break;
    }
    _end = omp_get_wtime();
    total_time = _end - init;
    printf("Tiempo total: %Lf\n s", total_time);
    cap.release();
    video.release();
}