#include <stdio.h>
#include <mpi.h>

int main(int argc, char** argv) {
    
    // Inicializacion MPI
    MPI_Init(&argc, &argv);
    
    // Obtiene el número de procesos
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Obtiene el rank del proceso
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    
    // Impresion de mensajes desde cada proceso
    printf("Hola, saludos desde el rank %d del proceso %d \n",world_rank,world_size);


    // Impresion de tamaño total desde el proceso 0

    if (world_rank == 0){
        printf("Saludos desde el proceso 0, hay %d rank y %d procesos\n",world_rank,world_size);
    }
    
    MPI_Finalize();

    return 0;
}