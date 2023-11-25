#include <mpi.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <iostream>
#include <vector>

using namespace std;
using namespace cv;
using namespace std;
using namespace cv;

// Función para reducir la resolución de un frame
Mat reducirFrame(const Mat &frame, int factorReduccion) {
    int newRows = frame.rows / factorReduccion;
    int newCols = frame.cols / factorReduccion;
    Mat tempFrame(newRows, newCols, frame.type());

    for (int i = 0; i < newRows; ++i) {
        for (int j = 0; j < newCols; ++j) {
            int r = 0, g = 0, b = 0;
            int cnt = 0;
            for (int y = i * factorReduccion; y < (i + 1) * factorReduccion; ++y) {
                for (int x = j * factorReduccion; x < (j + 1) * factorReduccion; ++x) {
                    Vec3b pixel = frame.at<Vec3b>(y, x);
                    r += pixel[2];
                    g += pixel[1];
                    b += pixel[0];
                    cnt++;
                }
            }
            r /= cnt;
            g /= cnt;
            b /= cnt;
            tempFrame.at<Vec3b>(i, j) = Vec3b(b, g, r);
        }
    }
    return tempFrame;
}

int main(int argc, char **argv) {
    // Inicializar MPI
    
    MPI_Init(&argc, &argv);
    int rank, numProcesses;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcesses);

    int totalFrames;
    string inputPath = argv[1];;
    string outputPath = argv[2];;
     
    VideoCapture cap;

    VideoWriter writer;
    int factorReduccion = 3;
    // El proceso raíz calcula el número total de frames
    
    if (rank == 0) {
        cap.open(inputPath);
        if (!cap.isOpened()) {
            cerr << "Error al abrir el video de entrada." << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        totalFrames = (int)cap.get(CAP_PROP_FRAME_COUNT);

        // Asegúrate de que frameWidth y frameHeight sean enteros
        int frameWidth = static_cast<int>(cap.get(CAP_PROP_FRAME_WIDTH)) / factorReduccion;
        int frameHeight = static_cast<int>(cap.get(CAP_PROP_FRAME_HEIGHT)) / factorReduccion;

        // Luego, abre el VideoWriter
        writer.open(outputPath, VideoWriter::fourcc('X', '2', '6', '4'), cap.get(CAP_PROP_FPS), Size(frameWidth, frameHeight));

 
    } else {
        // Procesos No Raíz: Intentar abrir el video
        cap.open(inputPath);
        if (!cap.isOpened()) {
            cerr << "Error al abrir el video de entrada en el proceso " << rank << "." << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    // Distribuir el número total de frames a todos los procesos
    MPI_Bcast(&totalFrames, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    
    // Calcular startFrame y endFrame para cada proceso
    int framesPerProcess = totalFrames / numProcesses;
    int startFrame = rank * framesPerProcess;
    int endFrame = (rank == numProcesses - 1) ? totalFrames - 1 : startFrame + framesPerProcess - 1;

    
    if (rank != 0) {
        cap.open(inputPath);
        cap.set(CAP_PROP_POS_FRAMES, startFrame);
    }

    // Procesar cada frame
    for (int i = startFrame; i <= endFrame; i++) {
        Mat frame, reducedFrame;
        cap >> frame;
        if (frame.empty()) break;

        reducedFrame = reducirFrame(frame, 3);

        // Serializar y enviar al proceso raíz
        if (rank != 0) {
            vector<uchar> buf;
            imencode(".png", reducedFrame, buf);
            int size = buf.size();
            MPI_Send(&size, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
            MPI_Send(buf.data(), size, MPI_BYTE, 0, 0, MPI_COMM_WORLD);
        } else {
            writer.write(reducedFrame);
        }
    }

   // El proceso raíz recibe los frames de otros procesos
    if (rank == 0) {
        for (int i = 1; i < numProcesses; i++) {
            for (int j = i * framesPerProcess; j <= min((i + 1) * framesPerProcess - 1, totalFrames - 1); j++) {
                int size;
                MPI_Recv(&size, 1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                vector<uchar> buf(size);
                MPI_Recv(buf.data(), size, MPI_BYTE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                Mat frame = imdecode(buf, IMREAD_COLOR);
                writer.write(frame);
            }
        }
    }

    cap.release();
    if (rank == 0) writer.release();
    MPI_Finalize();
    return 0;
}
