#include <stdio.h>
#include <mpi.h>
#include <opencv2/opencv.hpp>
#include <stdlib.h>

using namespace cv;
using namespace std;

long double init, _end; //Init and end time
long double total_time; //variable for the total time

unsigned char *videoFrames, *finalVideoFrames;  //Unsigned arrays for save the input and output frames


unsigned char * reduce(unsigned char *videoFrame, int width, int height){
    unsigned char finalVideoFrame[width*height];    //Array to save the reduced frame

    for (int i = 0; i < height; i += 3) //i goes over all the rows
    {
        for (int j = 0; j < width; j += 3)  //j goes over all the columns
        {
            double blue = 0;    //initialize the final blue
            double green = 0;   //initialize the final green
            double red = 0;     //initialize the final red

            for (int ik = 0; ik < 3; ik++)  //ik barriers over all the rows of a 3x3 pixel grid
            {
                for (int jk = 0; jk < 3; jk++)  //jk barriers over all the columns of a 3x3 pixel grid
                {
                    blue += videoFrame[i*27*width+j*9+9*jk+ik*9*width+0];  //sum over the blue value of the originals pixels
                    green += videoFrame[i*27*width+j*9+9*jk+ik*9*width+1]; //sum over the green value of the originals pixels
                    red += videoFrame[i*27*width+j*9+9*jk+ik*9*width+2];   //sum over the red value of the originals pixels
                }
            }
            //mean of each color
            red /= 9;
            green /= 9;
            blue /= 9;

            finalVideoFrame[i*width*3+ j*3+0] = blue;   //asign the new color value to the final frame
            finalVideoFrame[i*width*3+ j*3+1] = green;  //asign the new color value to the final frame
            finalVideoFrame[i*width*3+ j*3+2] = red;    //asign the new color value to the final frame
        }
    }
    return finalVideoFrame; //return the reduced frame
}


int main(int argc, char *argv[]){

    string input = argv[1];     //<-- Nombre del video de entrada // Hay que arreglar esto
    string output = argv[2];    //<--Nombre del video de salida //Hay que arreglar esto plis


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
    int width = 640;    //output video's width
    int height = 360;   //output video's height
    VideoWriter video(output, fourcc, fps, Size(width, height));

    // Start performance timing
    init = MPI_Wtime();

    Mat newFrame;   //Mat structure that saves all the reduced frames

    int n_frame;    //Num of frames readed in each iteration of while

    int world_size, world_rank; //Number of proces, current rank proces
    // Initialize MPI
    MPI_Init(&argc, &argv);
    // Obtain the number of process
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    // Obtain the process' rank number
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    printf("Inicializando la reduccion del video usando %d máquinas.", world_size);

    unsigned char videoFrame[3*width*3*height];     //Input Frame info inside each process
    unsigned char finalVideoFrame[width*height];    //Reduced Frame info inside each process

    // Main loop to read and process video frames
    while (true)
    {
        if(world_rank==0){  //Only the host Process reads and send the frames
            n_frame = 0;    //init number of Frames readed for iteration
            vector<unsigned char*> videoFrames; //unsigned char vector that saves all the frames that the host reads

            // Read 'world_size' number of frames into the vector
            for (int i = 0; i < world_size; i++)
            {
                Mat frame;
                cap >> frame;

                if (frame.empty())
                {
                    break;
                }
                videoFrames.push_back(frame.ptr()); //put the pointer to the uchar array of the Frame's info
                n_frame++;
            }

            // Exit the loop if no frames are read
            if (videoFrames.empty())
            {
                break;
            }

            for(int i = 1; i<world_size; i++){
                videoFrame = videoFrames[i];    //Sending each frame to each proces from the host process
                MPI_Sendrecv_replace(videoFrame, 1, MPI_UNSIGNED_CHAR, i, 0, 0, 0,MPI_COMM_WORLD, &status); //Recieving the frames in each process
                //(buf, count, MPI_Datatype, dest, sendtag, source, recvtag, MPI_Comm, status)
            }
            videoFrame = videoFrames[0];    //The host process takes the first frame
        }

        finalVideoFrame = reduce(videoFrame, width, height);    //Each process reduce the frame.

        if(world_rank!=0){
            MPI_Send(finalVideoFrame, 1, MPI_UNSIGNED_CHAR, 0, 0, MPI_COMM_WORLD);  //each non local host send the reduced Frames to the host
        }else{
            finalVideoFrames[world_rank] = finalVideoFrame; //The host process saves the reduced frame at pos 0 fo finalVideoFrames
            for(int i = 1; i<world_size; i++){
                MPI_Recv(finalVideoFrame, 1, MPI_UNSIGNED_CHAR, i, 0, &status); //The host process recieve the reduced frames from the others process
                finalVideoFrames[i] = finalVideoFrame;  //The host sort the reduced frames from the non host process in FinalVideoFrames array
            }

            for(int i = 0; i<n_frame;i++){
                newFrame.ptr() = finalVideoFrames[i];       //The host process write the video
                video.write(newFrame);
                char c = (char)waitKey(1);
                if (c == 27)
                    break;
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);    //The process wait for finish the video writing
    }
    //End MPI
    MPI_Finalize();
    // End performance timing and calculate total time
    _end = MPI_Wtime();
    total_time = _end - init;

    //Time writing un results.txt
    fprintf(fp, "- Tiempo total: %Lfs \n", total_time);
    cout << "Tiempo total: " << total_time << "s" << endl;
    cout << "Resultado guardado en results.txt" << endl;

    // Close the results file
    fclose(fp);

    // Release video resources
    cap.release();
    video.release();
}
