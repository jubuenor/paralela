
#include <iostream>
#include <mpi.h>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main(int argc, char** argv) {
    // Inicializa el entorno MPI
    MPI_Init(&argc, &argv);
    string inputVideo = "/home/user/Desktop/MateoCodes/nacional/Paralela/paralela/Practica3/test.mp4"; //argv[2];     
    // Obtiene el rango (identificador) del proceso actual
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Obtiene el número total de procesos
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        // Si el proceso es el proceso maestro (rango 0)

        // Abre un video usando OpenCV
        VideoCapture cap(inputVideo);

        if (!cap.isOpened()) {
            cout << "Error al abrir el video." <<endl
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        // Lee y muestra el video cuadro por cuadro
        cv::Mat frame;
        while (cap.read(frame)) {
            // Procesa el cuadro si es necesario

            // Envía el mensaje "hola" a los otros procesos
            for (int i = 1; i < size; ++i) {
                MPI_Send("hola", 5, MPI_CHAR, i, 0, MPI_COMM_WORLD);
            }

            // Muestra el cuadro en la ventana de OpenCV
            cv::imshow("Video", frame);
            cv::waitKey(10); // Espera 10 milisegundos
        }

        
        cap.release();
    } else {
        // Si el proceso no es el proceso maestro, imprime "hola"
        cout << "Proceso " << rank << ": hola" << std::endl;
    }

    // Finaliza el entorno MPI
    MPI_Finalize();

    return 0;
}
