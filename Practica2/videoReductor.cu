#include <stdio.h>
#include <stdlib.h>
#include "cuda_runtime.h"

#include <opencv2/opencv.hpp>

using namespace cv;

_global_ void add(int *a, int *b, int *c)
{
    *c = *a + *b;
}
int main()
{

    int a, b, c;
    // host copies of variables a, b & c
    int *d_a, *d_b, *d_c;
    // device copies of variables a, b & c
    int size = sizeof(int);
    // Allocate space for device copies of a, b, c
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);
    // Setup input values
    c = 0;
    a = 3;
    b = 5;
    // Copy inputs to device
    cudaMemcpy(d_a, &a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, &b, size, cudaMemcpyHostToDevice);
    // Launch add() kernel on GPU
    add<<<1, 1>>>(d_a, d_b, d_c);
    // Copy result back to host
    cudaError err = cudaMemcpy(&c, d_c, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        printf("CUDA error copying to Host: %s\n", cudaGetErrorString(err));
    }
    printf("result is %d\n", c);

    // initialize a 120X350 matrix of black pixels:
    Mat afdt;
    VideoCapture cap("japan.mp4");
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
    printf("fps: %d, fourcc: %d, frameCount: %d\n", fps, fourcc, frameCount);

    // Initialize output video file using OpenCV's VideoWriter
    VideoWriter video(output, fourcc, fps, Size(640, 360));

    // Declare an array to hold the final processed frames
    Mat finalVideoFrames[frameCount];

    // Cleanup
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return 0;
}