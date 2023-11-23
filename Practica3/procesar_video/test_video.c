/*
#include <opencv2/opencv.hpp>  // Import OpenCV library for video processing
#include <stdio.h>
#include <stdlib.h>

int main() {
    CvCapture* capture = cvCaptureFromFile("Practica3/procesar_video/japan.mp4"); // Reemplaza con la ruta de tu video
    if (!capture) {
        fprintf(stderr, "Error al abrir el video\n");
        return -1;
    }

    IplImage* frame;
    cvNamedWindow("Video", CV_WINDOW_AUTOSIZE);

    while (1) {
        frame = cvQueryFrame(capture);
        if (!frame) {
            break;
        }
        cvShowImage("Video", frame);

        char c = cvWaitKey(33);
        if (c == 27) { // Esc key
            break;
        }
    }

    cvReleaseCapture(&capture);
    cvDestroyWindow("Video");
    return 0;
}
*/

#include <stdio.h>
#include <mpi.h>
#include <opencv2/opencv.hpp>
#include <stdlib.h>

using namespace cv;
using namespace std;

long double init, _end;
long double total_time;

unsigned char *videoFrames, *finalVideoFrames;


void reduce(unsigned char *videoFrames, unsigned char *finalVideoFrames, int rank, int width, int height){
    
    for (int i = 0; i < height; i += 1)
    {
        for (int j = 0; j < width; j += 1) 
        {
            
            if(i<height && j<width){
                double blue = 0;
                double green = 0;
                double red = 0;

                for (int ik = 0; ik < 3; ik++)
                {
                    for (int jk = 0; jk < 3; jk++)
                    {
                        blue += videoFrames[rank][i*27*width+j*9+9*jk+ik*9*width+0];  //sum over the blue value of the originals pixels
                        green += videoFrames[rank][i*27*width+j*9+9*jk+ik*9*width+1]; //sum over the green value of the originals pixels
                        red += videoFrames[rank][i*27*width+j*9+9*jk+ik*9*width+2];   //sum over the red value of the originals pixels
                    }
                }
                //mean of the colors
                red /= 9;
                green /= 9;
                blue /= 9;

                finalVideoFrames[rank][i*width*3+ j*3+0] = blue;   //asign the new color value to the final frame
                finalVideoFrames[rank][i*width*3+ j*3+1] = green;  //asign the new color value to the final frame
                finalVideoFrames[rank][i*width*3+ j*3+2] = red;    //asign the new color value to the final frame
            }
        }
    }
}


int main(int argc, char *argv[]){
    // Initialize input and output video file names and number of threads from command-line arguments
    string input = argv[1];
    string output = argv[2];
    int n_threads = atoi(argv[3]);  // Convert the third argument to an integer for number of threads

    cout << "Iniciando programa con " << n_threads << " hilos..." << endl;

    // Open a file to write the results
    FILE *fp = fopen("results.txt", "a");
    fprintf(fp, "Hilos: %d \n", n_threads);

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
    init = MPI_Wtime();


    vector<unsigned char*> videoFrames;
    Mat newFrame[frameCount];

    int n_frame = 0;
    // Main loop to read and process video frames
    while (true)
    {

        
        // Read 'n_threads' number of frames into the vector
        for (int i = 0; i < size; i++)
        {
            Mat frame;
            cap >> frame;

            if (frame.empty())
            {
                break;
            }
            videoFrames.push_back(frame.ptr());
            n_frame++;
        }

        // Exit the loop if no frames are read
        if (videoFrames.empty())
        {
            break;
        }
        MPI // INICIA EL MPI
        
        MPI_Reduce(&unsigned char test, &pi, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
       
        for(int i = 0; i<n_frame;i++){
            newFrame.ptr() = finalVideoFrames[i];
            video.write(newFrame);
            char c = (char)waitKey(1);
            if (c == 27)
                break;
        }
    }

    // End performance timing and calculate total time
    _end = MPI_Wtime();
    total_time = _end - init;
    fprintf(fp, "- Tiempo total: %Lfs \n", total_time);
    cout << "Tiempo total: " << total_time << "s" << endl;
    cout << "Resultado guardado en results.txt" << endl;

    // Close the results file
    fclose(fp);

    // Release video resources
    cap.release();
    video.release();
}