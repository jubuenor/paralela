#include <stdio.h>
#include <stdlib.h>
#include "cuda_runtime.h"
#include <omp.h>

#include <opencv2/opencv.hpp>

using namespace cv;  // Use OpenCV's namespace
using namespace std; // Use the standard namespace

// Declare global variables for performance timing
long double init, _end;
long double total_time;

struct Pixel
{
    int red;
    int green;
    int blue;
};

__global__ void reduce(struct Pixel *videoFrames, struct Pixel *finalVideoFrames, int n_threads, int width, int size)
{
    int n = blockIdx.x;
    int a = 1080 / n_threads;
    int b = 1920 / n_threads;
    int thread = threadIdx.x;

    for (int i = thread * a; i < thread * a + a; i += 3)
    {
        for (int j = thread * b; j < thread * b + b; j += 3)
        {
            double blue = 0;
            double green = 0;
            double red = 0;

            for (int ik = 0; ik < 3; ik++)
            {
                for (int jk = 0; jk < 3; jk++)
                {
                    blue += videoFrames[n * size + i * width + j + ik * 3 + jk].blue;
                    green += videoFrames[n * size + i * width + j + ik * 3 + jk].green;
                    red += videoFrames[n * size + i * width + j + ik * 3 + jk].red;
                }
            }

            red /= 9;
            green /= 9;
            blue /= 9;
            finalVideoFrames[n * size + i / 3 * width / 3 + j / 3].blue = blue;
            finalVideoFrames[n * size + i / 3 * width / 3 + j / 3].green = green;
            finalVideoFrames[n * size + i / 3 * width / 3 + j / 3].red = red;

            // printf("(%d, %d, %d) ", finalVideoFrames[n * size + i * width / 3 + j].blue, finalVideoFrames[n * size + i * width / 3 + j].green, finalVideoFrames[n * size + i * width / 3 + j].red);
        }
        // printf("\n");
    }
}

int main(int argc, char *argv[])
{
    // Initialize input and output video file names and number of threads from command-line arguments
    string input = argv[1];
    string output = argv[2];
    int n_blocks = atoi(argv[3]);  // Convert the third argument to an integer for number of blocks
    int n_threads = atoi(argv[4]); // Convert the third argument to an integer for number of threads

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

    // Start performance timing
    init = omp_get_wtime();

    cudaError_t err = cudaSuccess;
    int initialHeight = 1080;
    int initialWidth = 1920;
    int finalHeight = 360;
    int finalWidth = 640;

    printf("Iniciando procesamiento del video...\n");
    // Main loop to read and process video frames
    while (true)
    {
        // Declare a vector to hold frames and their corresponding frame numbers
        vector<Mat> videoFrames;
        int n_frame = 0;
        // Read 'n_threads' number of frames into the vector
        for (int i = 0; i < n_blocks; i++)
        {
            Mat frame;
            cap >> frame;

            if (frame.empty())
            {
                break;
            }
            videoFrames.push_back(frame);
            n_frame++;
        }

        // Exit the loop if no frames are read
        if (videoFrames.empty())
        {
            break;
        }
        // Memory allocation of the input and output arrays
        int arraySizeInitial = videoFrames.size() * initialWidth * initialHeight * sizeof(struct Pixel);
        int arraySizeFinal = videoFrames.size() * finalWidth * finalHeight * sizeof(struct Pixel);

        struct Pixel *finalVideoFrames = (struct Pixel *)malloc(arraySizeFinal);
        struct Pixel *videoFramesArray = (struct Pixel *)malloc(arraySizeInitial);
        // Verify that allocations succeeded
        if (finalVideoFrames == NULL)
        {
            fprintf(stderr, "Failed to allocate host vector finalVideoFrames!\n");
            exit(EXIT_FAILURE);
        }

        if (videoFramesArray == NULL)
        {
            fprintf(stderr, "Failed to allocate host vector videoFramesArray!\n");
            exit(EXIT_FAILURE);
        }

        // Copy the frame content to 1D Array
        for (int n = 0; n < videoFrames.size(); n++)
            for (int i = 0; i < initialHeight; i++)
            {
                for (int j = 0; j < initialWidth; j++)
                {
                    videoFramesArray[n * videoFrames.size() + i * initialWidth + j].blue = videoFrames[n].at<Vec3b>(i, j)[0];
                    videoFramesArray[n * videoFrames.size() + i * initialWidth + j].green = videoFrames[n].at<Vec3b>(i, j)[1];
                    videoFramesArray[n * videoFrames.size() + i * initialWidth + j].red = videoFrames[n].at<Vec3b>(i, j)[2];
                    // printf("(%d, %d, %d)", videoFrames[n].at<Vec3b>(i, j)[0], videoFrames[n].at<Vec3b>(i, j)[1], videoFrames[n].at<Vec3b>(i, j)[2]);
                    // printf("(%d, %d, %d) ", videoFramesArray[n * videoFrames.size() + i * initialWidth + j].blue, videoFramesArray[n * videoFrames.size() + i * initialWidth + j].green, videoFramesArray[n * videoFrames.size() + i * initialWidth + j].red);
                }
                // printf("\n");
            }

        // Memory allocation  of the device array
        struct Pixel *d_videoFrames = NULL;
        err = cudaMalloc((void **)&d_videoFrames, arraySizeInitial);
        // Verify that dev allocations succeeded
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to allocate device vector videoFrames (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        // Memory allocation  of the device array
        struct Pixel *d_finalVideoFrames = NULL;
        err = cudaMalloc((void **)&d_finalVideoFrames, arraySizeFinal);
        // Verify that dev allocations succeeded
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to allocate device vector videoFrames (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        // Copy the content of the frame from Host to Device
        err = cudaMemcpy(d_videoFrames, videoFramesArray, arraySizeInitial, cudaMemcpyHostToDevice);
        // Verify that copy succeeded
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to copy vector videoFramesArray from host to device (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        // run cuda function
        reduce<<<n_blocks, n_threads>>>(d_videoFrames, d_finalVideoFrames, n_threads, initialWidth, videoFrames.size());

        // copy output data from the cuda device to the host memory
        err = cudaMemcpy(finalVideoFrames, d_finalVideoFrames, arraySizeFinal, cudaMemcpyDeviceToHost);
        // verify that copy succeeded
        if (err != cudaSuccess)
        {
            fprintf(stderr, "failed to copy vector finalvideoframes from device to host (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        // write the video

        for (int n = 0; n < videoFrames.size(); n++)
        {
            Mat newFrame = Mat::zeros(videoFrames[n].size() / 3, videoFrames[n].type());
            for (int i = 0; i < finalHeight; i++)
            {
                for (int j = 0; j < finalWidth; j++)
                {
                    int blue = finalVideoFrames[n * videoFrames.size() + i * finalWidth + j].blue;
                    int green = finalVideoFrames[n * videoFrames.size() + i * finalWidth + j].green;
                    int red = finalVideoFrames[n * videoFrames.size() + i * finalWidth + j].red;
                    Vec3b color = Vec3b(blue, green, red);
                    newFrame.at<Vec3b>(i, j) = color;
                }
            }

            // for (int ik = 0; ik < newFrame.rows; ik++)
            //{
            // for (int jk = 0; jk < newFrame.cols; jk++)
            //{
            // printf("(%d, %d, %d) ", newFrame.at<Vec3b>(ik, jk)[0], newFrame.at<Vec3b>(ik, jk)[1], newFrame.at<Vec3b>(ik, jk)[2]);
            //}
            // printf("\n");
            //}
            video.write(newFrame);
        }

        char c = (char)waitKey(1);
        if (c == 27)
            break;
        cudaFree(d_videoFrames);
        cudaFree(d_finalVideoFrames);
        free(finalVideoFrames);
        free(videoFramesArray);
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

    // Release video resources
    cap.release();
    video.release();
    return 0;
}