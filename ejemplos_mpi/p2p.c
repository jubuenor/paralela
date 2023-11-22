/*

Examplo to create a Point to Point communication
The programm must do :
    * Create and run two process
    * Send using the commnad line two numbers
    
    Process #0

        Send your integer to Process #1
        Receive the integer of Process #1
        Write the sum of the two values on stdout

    Process #1

        Receive the integer of Process #0
        Send your integer to Process #0
        Write the product of the two values on stdout


*/

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char** argv) {
    // Inicialización MPI
    MPI_Init(&argc, &argv);

    // Obtiene el número de procesos
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Obtiene el rank del proceso
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // Verifica que haya exactamente dos procesos
    if (world_size != 2) {
        fprintf(stderr, "Este programa requiere exactamente dos procesos.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    
    
    int number;

    // Convierte los argumentos de la línea de comandos a enteros cuando se ejecuta el programa
    int num1 = atoi(argv[1]);
    int num2 = atoi(argv[2]);
    if (world_rank == 0) {

        printf("\nInicio del programa\n");
        // Recibir numero del proceso 1 
        MPI_Recv(&number, 1, MPI_INT, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        // Envia el número 1 al proceso # 1

        MPI_Send(&num1,1,MPI_INT,1,0,MPI_COMM_WORLD);
        
        printf("Proceso 0 recibió el número %d de Proceso 1\n", number);

        printf("La suma para los numeros recibidos del Proceso 1 es :\n");
        printf("%d\n",number+number);
    
    } else if (world_rank == 1) {
    
       // Proceso 1 solo envía su número al proceso 0
        MPI_Send(&num2, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);

        // Proceso #1 recibe el número del proceso 0 
        MPI_Recv(&number, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        printf("Proceso 1 recibió el número %d de Proceso 0\n", number);

        printf("La multiplicacion para los numeros recibidos del Proceso 0 es :\n");
        printf("%d\n",number+number);
    
    }

    MPI_Finalize();
    return 0;

}
