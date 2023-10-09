#include <stdio.h>
#include <stdlib.h>

int main()
{
    system("g++ -o video_reduction video_reduction.c -fopenmp -std=c++11 `pkg-config --cflags --libs opencv4`");
    system("./video_reduction japan.mp4 output.mp4 1");
    system("./video_reduction japan.mp4 output.mp4 2");
    system("./video_reduction japan.mp4 output.mp4 4");
    system("./video_reduction japan.mp4 output.mp4 8");
    system("./video_reduction japan.mp4 output.mp4 16");
}