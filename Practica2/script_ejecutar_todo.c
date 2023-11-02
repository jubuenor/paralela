#include <stdio.h>
#include <stdlib.h>

int main()
{
    // Compile and run video_reduction
    system("make all");

    system("./videoReductor japan.mp4 output.mp4 1 1");
    system("./videoReductor japan.mp4 output.mp4 1 2");
    system("./videoReductor japan.mp4 output.mp4 1 4");
    system("./videoReductor japan.mp4 output.mp4 1 8");
    system("./videoReductor japan.mp4 output.mp4 1 16");
    system("./videoReductor japan.mp4 output.mp4 1 32");

    system("./videoReductor japan.mp4 output.mp4 2 1");
    system("./videoReductor japan.mp4 output.mp4 2 2");
    system("./videoReductor japan.mp4 output.mp4 2 4");
    system("./videoReductor japan.mp4 output.mp4 2 8");
    system("./videoReductor japan.mp4 output.mp4 2 16");
    system("./videoReductor japan.mp4 output.mp4 2 32");

    system("./videoReductor japan.mp4 output.mp4 4 1");
    system("./videoReductor japan.mp4 output.mp4 4 2");
    system("./videoReductor japan.mp4 output.mp4 4 4");
    system("./videoReductor japan.mp4 output.mp4 4 8");
    system("./videoReductor japan.mp4 output.mp4 4 16");
    system("./videoReductor japan.mp4 output.mp4 4 32");

    system("./videoReductor japan.mp4 output.mp4 8 1");
    system("./videoReductor japan.mp4 output.mp4 8 2");
    system("./videoReductor japan.mp4 output.mp4 8 4");
    system("./videoReductor japan.mp4 output.mp4 8 8");
    system("./videoReductor japan.mp4 output.mp4 8 16");
    system("./videoReductor japan.mp4 output.mp4 8 32");

    system("./videoReductor japan.mp4 output.mp4 16 1");
    system("./videoReductor japan.mp4 output.mp4 16 2");
    system("./videoReductor japan.mp4 output.mp4 16 4");
    system("./videoReductor japan.mp4 output.mp4 16 8");
    system("./videoReductor japan.mp4 output.mp4 16 16");
    system("./videoReductor japan.mp4 output.mp4 16 32");
}
