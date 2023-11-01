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

__global__ void reduce(int *videoFrames, int *finalVideoFrames, int n_frames,int n_blocks, int n_threads, int width, int height){
    // loop into the matrix rows, the rows will be managed by the threads
    for (int i = 0; i < height; i += n_threads) // step of 3*n_threads, a thread manages a 3X3 pixel grid
    {   // loop into the matrix columns, the columns will be managed by the blocks
        for (int j = 0; j < width; j += n_blocks) //step of 3*n_blocks a block manages n_trheadsx3 grid
        {
            //initialize the color variables
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

            finalVideoFrames[((int) ((i))+threadIdx.x)*width*3+ ((int) ((j))+blockIdx.x)*3+0] = blue;   //asign the neu color value to the final frame
            finalVideoFrames[((int) ((i))+threadIdx.x)*width*3+ ((int) ((j))+blockIdx.x)*3+1] = green;  //asign the neu color value to the final frame
            finalVideoFrames[((int) ((i))+threadIdx.x)*width*3+ ((int) ((j))+blockIdx.x)*3+2] = red;    //asign the neu color value to the final frame
        }
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

    cudaError_t err = cudaSuccess;
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

    //Memory allocation  of the device array
    int *d_finalVideoFrames = NULL;
    err = cudaMalloc((void **)&d_finalVideoFrames, height*width*3*sizeof(int));
    // Verify that dev allocations succeeded
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device vector videoFrames (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    //Memory allocation  of the device array
    int *d_videoFrames = NULL;
    err = cudaMalloc((void **)&d_videoFrames, 3*height*3*width*3*sizeof(int));
    // Verify that dev allocations succeeded
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device vector videoFrames (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }


    printf("Iniciando procesamiento del video...\n");
    // Main loop to read and process video frames
    while (true)
    {
        // Declare a vector to hold frames and their corresponding frame numbers
        vector<pair<Mat, int>> videoFrames;
        int n_frame = 0;
        // Read 'n_threads' number of frames into the vector
        for (int i = 0; i < n_threads; i++)
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

        //loop over the charged n_block frames of the original video
        for(int n =0; n<n_frame; n++){

            init = omp_get_wtime();
            //Copy the frame content to 1D Array
            for(int i = 0; i<height*3; i++){
                for(int j = 0; j<width*3; j++){
                    videoFramesArray[i*width*9+j*3+0] = videoFrames[n].first.at<Vec3b>(i, j)[0];
                    videoFramesArray[i*width*9+j*3+1] = videoFrames[n].first.at<Vec3b>(i, j)[1];
                    videoFramesArray[i*width*9+j*3+2] = videoFrames[n].first.at<Vec3b>(i, j)[2];
                }
            }
            _end = omp_get_wtime();
            total_time += _end - init;

            // Copy the content of the frame from Host to Device
            err = cudaMemcpy(d_videoFrames, videoFramesArray, 3*height*3*width*3*sizeof(int), cudaMemcpyHostToDevice);
            // Verify that copy succeeded
            if (err != cudaSuccess) {
                fprintf(stderr, "Failed to copy vector videoFramesArray from host to device (error code %s)!\n", cudaGetErrorString(err));
                exit(EXIT_FAILURE);
            }

            // run cuda function
            reduce<<<n_blocks,n_threads>>>(d_videoFrames, d_finalVideoFrames, n_frame, n_blocks, n_threads, width, height);

            //Copy output data from the CUDA device to the host memory
            err = cudaMemcpy(finalVideoFrames, d_finalVideoFrames, height*width*3*sizeof(int), cudaMemcpyDeviceToHost);
            // Verify that copy succeeded
            if (err != cudaSuccess) {
                fprintf(stderr, "Failed to copy vector finalVideoFrames from device to host (error code %s)!\n", cudaGetErrorString(err));
                exit(EXIT_FAILURE);
            }

            //Write the video

            //Create a new frame to allocate the new pixel's values
            Mat newFrame = Mat::zeros(videoFrames[n].first.size()/3, videoFrames[n].first.type());
            //loop over the pixel's values frame
            for(int i=0; i<height; i++){
                for(int j=0; j< width; j++){
                    int blue = finalVideoFrames[i*width*3+j*3+0];
                    int green = finalVideoFrames[i*width*3+j*3+1];
                    int red = finalVideoFrames[i*width*3+j*3+2];
                    Vec3b color = Vec3b(blue, green, red);
                    newFrame.at<Vec3b>(i, j) = color;
                }
            }

            // write the video
            video.write(newFrame);
            char c = (char)waitKey(1);
            if (c == 27)
                break;
        }
    }

    // End performance timing and calculate total time
    fprintf(fp, "- Tiempo total: %Lfs \n", total_time);
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
