#include <stdio.h>
#include <stdlib.h>
#include "cuda_runtime.h"

#include <opencv2/opencv.hpp>

using namespace cv;  // Use OpenCV's namespace
using namespace std;  // Use the standard namespace

// Declare global variables for performance timing
long double init, _end;
long double total_time;

__global__ void reduce(Math *videoFrames, Math *finalVideoFrames, int n_frames, int n_threads){
    if(blockIdx.x<n_frames){
        Mat newFrame = Mat::zeros(videoFrames[blockIdx.x].first.size() / 3, videoFrames[blockIdx.x].first.type());
        for (int i = 0; i < videoFrames[blockIdx.x].first.rows; i += 3*n_threads)
        {
            for (int j = 0; j < videoFrames[blockIdx.x].first.cols; j += 3)
            {

                double blue = 0;
                double green = 0;
                double red = 0;

                for (int ik = 0; ik < 3; ik++)
                {
                    for (int jk = 0; jk < 3; jk++)
                    {
                        blue += videoFrames[blockIdx.x].first.at<Vec3b>(i+threadIdx.x + ik, j + jk)[0];
                        green += videoFrames[blockIdx.x].first.at<Vec3b>(i+threadIdx.x + ik, j + jk)[1];
                        red += videoFrames[blockIdx.x].first.at<Vec3b>(i+threadIdx.x + ik, j + jk)[2];
                    }
                }

                red /= 9;
                green /= 9;
                blue /= 9;
                Vec3b color = Vec3b(blue, green, red);
                newFrame.at<Vec3b>((i+threadIdx.x) / 3, j / 3) = color;
            }
        }
        finalVideoFrames[videoFrames[blockIdx.x].second] = newFrame;
    }
}

int main(int argc, char *argv[])
{
    // Initialize input and output video file names and number of threads from command-line arguments
    string input = argv[1];
    string output = argv[2];
    int n_blocks = atoi(argv[3]);  // Convert the third argument to an integer for number of blocks
    int n_threads = atoi(argv[4]);  // Convert the third argument to an integer for number of threads

    cout << "Iniciando programa con " << n_blocks << " bloques..." << endl;
    cout << "Iniciando programa con " << n_threads << " hilos..." << endl;

    // Open a file to write the results
    FILE *fp = fopen("results.txt", "a");
    fprintf(fp, "Bloques: %d \n", n_blocks);

    // Open input video file using OpenCV's VideoCapture
    VideoCapture cap(input);
    // Check if the video file is successfully opened
    if (!cap.isOpened())
    {
        cout << "Error opening video stream or file" << endl;
        return 0;
    }

    // Retrieve properties of the input video
    int fps = cap.get(CAP_PROP_FPS);
    int fourcc = cap.get(CAP_PROP_FOURCC);
    int frameCount = cap.get(CAP_PROP_FRAME_COUNT);

    // Initialize output video file using OpenCV's VideoWriter
    VideoWriter video(output, fourcc, fps, Size(640, 360));

    // Declare an array to hold the final processed frames
    Mat finalVideoFrames[frameCount];

    // Start performance timing
    init = omp_get_wtime();

    int n_frame = 0;

    // Main loop to read and process video frames
    while (true)
    {
        // Declare a vector to hold frames and their corresponding frame numbers
        vector<pair<Mat, int>> videoFrames;

        // Read 'n_threads' number of frames into the vector
        for (int i = 0; i < n_blocks; i++)
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

        // Exit the loop if no frames are read
        if (videoFrames.empty())
        {
            break;
        }

        // Enable OpenMP parallelization with the specified number of threads
        reduce<<<n_blocks,n_threads>>>(d_videoFrames, d_finalVideoFrames, videoFrames.size(), n_threads);
        // Loop through the vector to process each frame

    }

    // End performance timing and calculate total time
    _end = omp_get_wtime();
    total_time = _end - init;
    fprintf(fp, "- Tiempo total: %Lfs \n", total_time);
    cout << "Tiempo total: " << total_time << "s" << endl;
    cout << "Resultado guardado en results.txt" << endl;

    // Close the results file
    fclose(fp);

    // Write the final processed frames to the output video
    for (int i = 0; i < frameCount; i++)
    {
        video.write(finalVideoFrames[i]);
        char c = (char)waitKey(1);
        if (c == 27)
            break;
    }

    // Release video resources
    cap.release();
    video.release();
    return 0;
}
