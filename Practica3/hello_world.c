#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv) {
    // Inicializa el entorno MPI
    MPI_Init(&argc, &argv);

    // Obtiene el número de procesos
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Obtiene el rango del proceso
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // Imprime un mensaje desde cada proceso
    printf("Hello world from rank %d out of %d processors\n", world_rank, world_size);

    // Finaliza el entorno MPI
    MPI_Finalize();
    return 0;
}

