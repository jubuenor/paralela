#include <opencv2/highgui.hpp>
#include <stdio.h>

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
