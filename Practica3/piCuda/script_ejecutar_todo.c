#include <stdio.h>
#include <stdlib.h>

int main()
{
    // Compile and run video_reduction
    system("make all");

    system("./piCuda 8");
    system("./piCuda 16");
    system("./piCuda 32");
    system("./piCuda 64");
    system("./piCuda 128");
    system("./piCuda 256");
}
