#include <opencv2/opencv.hpp>  // Import OpenCV library for video processing
#include <omp.h>  // Import OpenMP library for parallel computing and performance timing
#include <stdio.h>
#include <stdlib.h>

using namespace cv;  // Use OpenCV's namespace
using namespace std;  // Use the standard namespace

// Declare global variables for performance timing
long double init, _end;
long double total_time;

int main(int argc, char *argv[])
{
    // Initialize input and output video file names and number of threads from command-line arguments
    string input = argv[1];
    string output = argv[2];
    int n_threads = atoi(argv[3]);  // Convert the third argument to an integer for number of threads

    // Display all command-line arguments
    for (int i = 0; i < argc; i++)
    {
        cout << argv[i] << endl;
    }

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

    // Start performance timing
    init = omp_get_wtime();

    // Loop through each frame of the input video
    for (int n = 0; n < frameCount; n++)
    {
        Mat frames;  // Declare a matrix to hold the current frame
        cap >> frames;  // Read the current frame into the matrix

        // Initialize a new matrix for the processed frame
        Mat newFrame = Mat::zeros(frames.size() / 3, frames.type());

        // Enable OpenMP parallelization with the specified number of threads
#pragma omp parallel num_threads(n_threads)
#pragma omp for
        // Loop through the rows and columns of the frame
        for (int i = 0; i < frames.rows; i += 3)
        {
            for (int j = 0; j < frames.cols; j += 3)
            {
                // Initialize RGB values
                double blue = 0;
                double green = 0;
                double red = 0;

                // Calculate the average RGB values for a 3x3 block
                for (int ik = 0; ik < 3; ik++)
                {
                    for (int jk = 0; jk < 3; jk++)
                    {
                        blue += frames.at<Vec3b>(i + ik, j + jk)[0];
                        green += frames.at<Vec3b>(i + ik, j + jk)[1];
                        red += frames.at<Vec3b>(i + ik, j + jk)[2];
                    }
                }

                // Finalize the average RGB values
                red /= 9;
                green /= 9;
                blue /= 9;

                // Create a new color from the average RGB values
                Vec3b color = Vec3b(blue, green, red);

                // Assign the new color to the corresponding pixel in the new frame
                newFrame.at<Vec3b>(i / 3, j / 3) = color;
            }
        }

        // Write the processed frame to the output video
        video.write(newFrame);
    }

    // End performance timing and calculate total time
    _end = omp_get_wtime();
    total_time = _end - init;
    printf("Tiempo total: %Lf\n s", total_time);

    // Release video resources
    cap.release();
    video.release();
}
