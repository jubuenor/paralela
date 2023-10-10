#include <stdio.h>
#include <stdlib.h>

int main()
{
    // Step 1: Update package list
    system("sudo apt-get update");

    // Step 2: Install required packages
    system("sudo apt-get install -y build-essential cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev");

    // Step 3: Clone OpenCV repository
    system("git clone https://github.com/opencv/opencv.git");

    // Step 4: Create build directory
    system("cd opencv && mkdir build && cd build");

    // Step 5: Run CMake
    system("cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=/usr/local ..");

    // Step 6: Compile OpenCV
    system("make -j4");

    // Step 7: Install OpenCV
    system("sudo make install");

    // Compile and run video_reduction
    system("g++ -o video_reduction video_reduction.c -fopenmp -std=c++11 `pkg-config --cflags --libs opencv4`");
    
    system("./video_reduction japan.mp4 output.mp4 1");
    system("./video_reduction japan.mp4 output.mp4 2");
    system("./video_reduction japan.mp4 output.mp4 4");
    system("./video_reduction japan.mp4 output.mp4 8");
    system("./video_reduction japan.mp4 output.mp4 16");
