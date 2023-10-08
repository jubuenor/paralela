#include <opencv2/opencv.hpp>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#define N_THREADS 16

using namespace cv;
using namespace std;

long double init, _end;
long double total_time;

struct Pair
{
    Mat frame;
    int index;
};

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

    Mat finalVideoFrames[frameCount];

    init = omp_get_wtime();

    int n_frame = 0;

    while (true)
    {
        vector<pair<Mat, int>> videoFrames;

        for (int i = 0; i < N_THREADS; i++)
        {
            Mat frame;
            cap >> frame;

            if (frame.empty())
            {
                break;
            }
            videoFrames.push_back({frame, n_frame});
            n_frame++;
        }

        if (videoFrames.empty())
        {
            break;
        }
#pragma omp parallel num_threads(N_THREADS)
#pragma omp for
        for (int n = 0; n < videoFrames.size(); n++)
        {
            Mat newFrame = Mat::zeros(videoFrames[n].first.size() / 3, videoFrames[n].first.type());
            // cout << videoFrames[n].size() << endl;
            for (int i = 0; i < videoFrames[n].first.rows; i += 3)
            {
                for (int j = 0; j < videoFrames[n].first.cols; j += 3)
                {

                    double blue = 0;
                    double green = 0;
                    double red = 0;
                    for (int ik = 0; ik < 3; ik++)
                    {
                        for (int jk = 0; jk < 3; jk++)
                        {
                            blue += videoFrames[n].first.at<Vec3b>(i + ik, j + jk)[0];
                            green += videoFrames[n].first.at<Vec3b>(i + ik, j + jk)[1];
                            red += videoFrames[n].first.at<Vec3b>(i + ik, j + jk)[2];
                        }
                    }

                    red /= 9;
                    green /= 9;
                    blue /= 9;
                    Vec3b color = Vec3b(blue, green, red);
                    newFrame.at<Vec3b>(i / 3, j / 3) = color;
                }
            }
            //  cout << newFrame.size() << endl;
            finalVideoFrames[videoFrames[n].second] = newFrame;
            // cout << finalVideoFrames[n].size() << endl;
        }
    }

    _end = omp_get_wtime();
    total_time = _end - init;
    printf("Tiempo total: %Lf\n s", total_time);

    cout << n_frame << endl;

    for (int i = 0; i < frameCount; i++)
    {
        // cout << finalVideoFrames[i].size() << endl;
        video.write(finalVideoFrames[i]);
        //     imshow("Frame", finalVideoFrames[i]);
        char c = (char)waitKey(1);
        if (c == 27)
            break;
    }

    cap.release();
    video.release();
}