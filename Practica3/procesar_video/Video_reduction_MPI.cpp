#include <stdio.h>
#include <mpi.h>
#include <opencv2/opencv.hpp>
#include <stdlib.h>

using namespace cv;
using namespace std;

long double init, _end; //Init and end time
long double total_time; //variable for the total time

unsigned char *videoFrames, finalVideoFrames;  //Unsigned arrays for save the input and output frames


unsigned char* reduce(unsigned char *videoFrame, int width, int height){

    unsigned char* finalVideoFrame = new unsigned char[width*height*3]; // Dynamically allocated array
    //unsigned char finalVideoFrame[width*height*3];    //Array to save the reduced frame
    cout<<videoFrame[250*width*3+ 100*3+0]<<endl;
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
    //cout<<finalVideoFrame[250*width*3+ 100*3+0]<<endl;
    return  finalVideoFrame; //return the reduced frame
}


int main(int argc, char *argv[]){

    string input = "/home/user/Desktop/MateoCodes/nacional/Paralela/paralela/Practica3/japan.mp4"; 
    string output = "/home/user/Desktop/MateoCodes/nacional/Paralela/paralela/Practica3/roorr.mp4"; 
    
    // Open a file to write the results
    //FILE *fp = fopen("results.txt", "a");
    //fprintf(fp, "Hilos: %d \n", n_threads);

    // Open input video file using OpenCV's VideoCapture
    VideoCapture cap(input);
    // Check if the video file is successfully opened
    if (!cap.isOpened())
    {
        cout << "Error opening video stream or file" << endl;
        return 0;
    }

    // Retrieve properties of the input video
    int fps = cap.get(CAP_PROP_FPS) ;
    int fourcc = static_cast<int>(cap.get(CAP_PROP_FOURCC));  // Make sure to cast to int
    int frameCount = cap.get(CAP_PROP_FRAME_COUNT);

    // Initialize output video file using OpenCV's VideoWriter
    int width = 640;    //output video's width
    int height = 360;   //output video's height
    VideoWriter video(output, fourcc, fps, Size(width, height));

    // Start performance timing
    //init = MPI_Wtime();

    int n_frame;    //Num of frames readed in each iteration of while

    int world_size, world_rank; //Number of proces, current rank proces
    
    // Initialize MPI
    MPI_Init(&argc, &argv);

    MPI_Status status;
    MPI_Request request;
    
    // Obtain the number of process
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    // Obtain the process' rank number
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    printf("Inicializando la reduccion del video usando %d máquinas.", world_size);
    
    // unsigned char *videoFrame = new unsigned char[3*width*3*height]; // Dynamically allocated array
    //    unsigned char **finalVideoFrames = new unsigned char*[world_size]; // Pointer to a pointer for 2D array
    
    cout << "\n--->" << world_size<<endl;
    unsigned char *videoFrame = new unsigned char[3*width*3*height*3]; // Dynamically allocated array
    unsigned char **finalVideoFrames = new unsigned char*[world_size]; // Pointer to a pointer for 2D array
    unsigned char *finalVideoFrame = new unsigned char[width*height*3]; // Dynamically allocated array
    // Main loop to read a  nd process video frames

    int band = 0;
    int cont = 0 ;
    while (true)
    {
        if(world_rank==0){  
            //Only the host Process reads and send the frames
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
                cout<<"Entro"<<endl;
                band = 1;
                
            }
            for(int i = 1; i<world_size; i++){
                
                MPI_Isend(&band,1,MPI_INT,i,0,MPI_COMM_WORLD,&request);
                MPI_Wait(&request, &status); 
            }
            if (band == 1){
                break;
            }
            for(int i = 1; i<world_size; i++){
                
                videoFrame = videoFrames[i];    //Sending each frame to each proces from the host process

                
                MPI_Isend(videoFrame, 1, MPI_UNSIGNED_CHAR, i, 0, MPI_COMM_WORLD,&request); //Recieving the frames in each proces
                MPI_Wait(&request, &status); 
            }
            
            videoFrame = videoFrames[0];    //The host process takes the first frame
        }

        else{
            
            MPI_Irecv(&band,1,MPI_INT,0,0,MPI_COMM_WORLD,&request);
            MPI_Wait(&request, &status); 
            
            if (band == 1){
                break;
            }
            MPI_Irecv(videoFrame,1,MPI_UNSIGNED_CHAR,0,0,MPI_COMM_WORLD,&request);
            MPI_Wait(&request, &status); 
            
        }
        
        if(world_rank!=0){
            
            finalVideoFrame = reduce(videoFrame, width, height);    //Each process reduce the frame.
            MPI_Isend(finalVideoFrame, 1, MPI_UNSIGNED_CHAR, 0, 0, MPI_COMM_WORLD,&request);  //each non local host send the reduced Frames to the host
            MPI_Wait(&request, &status); 

        }else{ // rank 0

            
            //finalVideoFrames[world_rank] = finalVideoFrame; //The host process saves the reduced frame at pos 0 fo finalVideoFrames
            for(int i = 1; i<world_size; i++){
                
                MPI_Irecv(finalVideoFrame, 1, MPI_UNSIGNED_CHAR, i, 0,MPI_COMM_WORLD ,&request); //The host process recieve the reduced frames from the others process
                finalVideoFrames[i] = finalVideoFrame;  //The host sort the reduced frames from the non host process in FinalVideoFrames array
                //cout <<"recibio" <<world_rank<<endl;
                MPI_Wait(&request, &status); 
            }
            

            int cont = 0; // Initialize cont if not already done
            Mat newFrame = Mat::zeros(height, width, CV_8UC3);

            for(int i = 1; i < n_frame; i++) { // Start from 0 if you want to process all frames
                memcpy(newFrame.ptr(), finalVideoFrames[i], newFrame.step * newFrame.rows);
                video.write(newFrame);

                char c = (char)waitKey(1);
                if (c == 27) {
                    break;
                }

            }
        cout << cont <<endl;
        }
        
    }
    
    //End MPI
    MPI_Finalize();
    // End performance timing and calculate total time
    
    total_time = _end - init;

    //Time writing un results.txt
    //fprintf(fp, "- Tiempo total: %Lfs \n", total_time);
    //cout << "Tiempo total: " << total_time << "s" << endl;
    //cout << "Resultado guardado en results.txt" << endl;

    // Close the results file
    //fclose(fp);

    // Release video resources
    cap.release();
    video.release();
    cout << "Total frames" << fps <<endl;
    return 0;
}
