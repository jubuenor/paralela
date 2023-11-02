#include <stdio.h>
#include <stdlib.h>
#include "cuda_runtime.h"
#include <omp.h>

#include <opencv2/opencv.hpp>

using namespace cv;  // Use OpenCV's namespace
using namespace std;  // Use the standard namespace

// Declare global variables for performance timing
long double init, _end;
long double total_time;
unsigned char *d_videoFrames, *d_finalVideoFrames;


__global__ void reduce(unsigned char *videoFrames, unsigned char *finalVideoFrames, int n_blocks, int n_threads, int width, int height){
    // loop into the matrix rows, the rows will be managed by the threads
    for (int i = 0; i < height; i += n_threads) // step of 3*n_threads, a thread manages a 3X3 pixel grid
    {   // loop into the matrix columns, the columns will be managed by the blocks
        for (int j = 0; j < width; j += n_blocks) //step of 3*n_blocks a block manages n_trheadsx3 grid
        {
            //initialize the color variables
            if((i+threadIdx.x)<height && (j+blockIdx.x)<width){
                double blue = 0;
                double green = 0;
                double red = 0;
                //each pixel of the finalFrames will be a mean of a 3x3 grid of the original video frame
                for (int ik = 0; ik < 3; ik++)
                {
                    for (int jk = 0; jk < 3; jk++)
                    {
                        blue += videoFrames[(i+threadIdx.x)*27*width+(j+blockIdx.x)*9+9*jk+ik*9*width+0];  //sum over the blue value of the originals pixels
                        green += videoFrames[(i+threadIdx.x)*27*width+(j+blockIdx.x)*9+9*jk+ik*9*width+1]; //sum over the green value of the originals pixels
                        red += videoFrames[(i+threadIdx.x)*27*width+(j+blockIdx.x)*9+9*jk+ik*9*width+2];   //sum over the red value of the originals pixels
                    }
                }
                //mean of the colors
                red /= 9;
                green /= 9;
                blue /= 9;

                finalVideoFrames[((int) ((i))+threadIdx.x)*width*3+ ((int) ((j))+blockIdx.x)*3+0] = blue;   //asign the new color value to the final frame
                finalVideoFrames[((int) ((i))+threadIdx.x)*width*3+ ((int) ((j))+blockIdx.x)*3+1] = green;  //asign the new color value to the final frame
                finalVideoFrames[((int) ((i))+threadIdx.x)*width*3+ ((int) ((j))+blockIdx.x)*3+2] = red;    //asign the new color value to the final frame
            }
        }
    }
}

void init_cuda_kernel(cudaError_t err){
    // Verify that dev allocations succeeded
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device vector (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
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

    total_time = 0;

    // Open a file to write the results
    FILE *fp = fopen("results.txt", "a");
    fprintf(fp, "%d \t", n_blocks);
    fprintf(fp, "%d \t", n_threads);

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
    cudaError_t err;
    int height = 360;
    int width = 640;

    //Memory allocation of the host input and output arrays
    int *finalVideoFrames = (int *)malloc(height*width*3*sizeof(int));
    int *videoFramesArray = (int *)malloc(3*height*3*width*3*sizeof(int));
    // Verify that allocations succeeded
    if (finalVideoFrames == NULL) {
        fprintf(stderr, "Failed to allocate host vector finalVideoFrames!\n");
        exit(EXIT_FAILURE);
    }

    if (videoFramesArray == NULL) {
        fprintf(stderr, "Failed to allocate host vector videoFramesArray!\n");
        exit(EXIT_FAILURE);
    }

    Mat newFrame = Mat::zeros(height, width, CV_8UC3);

    init_cuda_kernel(cudaMalloc<unsigned char>(&d_finalVideoFrames, newFrame.step*newFrame.rows));
    init_cuda_kernel(cudaMalloc<unsigned char>(&d_videoFrames, 3*height*3*width*3*sizeof(unsigned char)));

    printf("Iniciando procesamiento del video...\n");
    init = omp_get_wtime();
    int n = 0;
    // Main loop to read and process video frames
    while (true)
    {
        // Declare a vector to hold frames and their corresponding frame numbers
        Mat videoFrames;
        cap >> videoFrames;

        if (videoFrames.empty())
        {
          break;
        }

        // Copy the content of the frame from Host to Device
        err = cudaMemcpy(d_videoFrames, videoFrames.ptr(), 3*height*3*width*3*sizeof(unsigned char), cudaMemcpyHostToDevice);
        // Verify that copy succeeded
        if (err != cudaSuccess) {
            fprintf(stderr, "Failed to copy vector videoFrames from host to device (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        // run cuda function
        reduce<<<n_blocks,n_threads>>>(d_videoFrames, d_finalVideoFrames, n_blocks, n_threads, width, height);

        //Write the video
        //Create a new frame to allocate the new pixel's values
        //loop over the pixel's values frame
        err = cudaMemcpy(newFrame.ptr(), d_finalVideoFrames, newFrame.step*newFrame.rows, cudaMemcpyDeviceToHost);
        // Verify that copy succeeded
        if (err != cudaSuccess) {
            fprintf(stderr, "Failed to copy vector finalVideoFrames from device to host (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        // write the video
        video.write(newFrame);
        char c = (char)waitKey(1);
        if (c == 27){
          break;
        }
    }

    _end = omp_get_wtime();
    total_time += _end - init;
    // End performance timing and calculate total time
    fprintf(fp, "%Lfs \n", total_time);
    cout << "Tiempo total: " << total_time << "s" << endl;
    cout << "Resultado guardado en results.txt" << endl;

    //clean the cuda memory
    err = cudaFree(d_videoFrames);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector d_videoFrames (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    //clean the cuda memory
    err = cudaFree(d_finalVideoFrames);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector d_finalVideoFrames (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    //clean the memory
    free(videoFramesArray);
    free(finalVideoFrames);
    // Close the results file
    fclose(fp);

    // Write the final processed frames to the output video

    // Release video resources
    cap.release();
    video.release();
    return 0;
}
