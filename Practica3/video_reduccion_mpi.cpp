
#include <opencv2/opencv.hpp>
#include <iostream>
#include <mpi.h>

using namespace std;
using namespace cv;

int main(int argc, char *argv[]){
    // Inicializa el entorno MPI
    MPI_Init(&argc, &argv);
    cout<<endl;
    string input = "/home/user/Desktop/MateoCodes/nacional/Paralela/paralela/Practica3/test.mp4"; 
    string output = "/home/user/Desktop/MateoCodes/nacional/Paralela/paralela/Practica3/test_SALIDA.mp4"; 

    // Obtiene el rango (identificador) del proceso actual
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Obtiene el número total de procesos
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Nombre del video de entrada y salida
    string inputVideo = input;
    string outputVideo = output;
    if(rank == 0){
        VideoCapture cap(inputVideo);

        if (!cap.isOpened()) {
            cout << "Error al abrir el video." <<endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
            return -1;
        }
        // Obtiene las propiedades del video
        int frameWidth = cap.get(CAP_PROP_FRAME_WIDTH);
        int frameHeight = cap.get(CAP_PROP_FRAME_HEIGHT);
        int fps = cap.get(CAP_PROP_FPS);

        // Define el nuevo tamaño (por ejemplo, la mitad del original)
        int newWidth = frameWidth / 2;
        int newHeight = frameHeight / 2;

        // Crea un objeto VideoWriter para escribir el video de salida
        VideoWriter writer(outputVideo, VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, Size(newWidth, newHeight));

        if (!writer.isOpened()) {
            cerr << "Error al abrir el escritor de video" << endl;
            return -1;
        }


        Mat frame, resizedFrame;

        // Lee cada fotograma, reduce su tamaño y lo escribe en el video de salida
        while (true) {
            cap >> frame;
            if (frame.empty()) break;

            // Redimensiona el fotograma
            resize(frame, resizedFrame, Size(newWidth, newHeight));

            // Escribe el fotograma redimensionado en el video de salida
            writer.write(resizedFrame);
        }

        // Libera los recursos
        cap.release();
        writer.release();
    }else{
        // Si el proceso no es el proceso maestro, imprime "hola"
        cout << "Proceso " << rank << ": hola" << std::endl;
    }



    cout << "Video procesado con éxito" << endl;
      // Finaliza el entorno MPI
    MPI_Finalize();
    return 0;
}
