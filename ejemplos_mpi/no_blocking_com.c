
/*
Code created to implement no-blocking communication

if rank == 0 then

    Work for 3 seconds
    Initialise the send to process 1
    Work for 6 seconds
        Every milli-second, test if process 1 is ready to communicate
    Send the second batch of data to process 1
    Wait for process 1 to receive the data

else if rank == 1 then

    Work for 5 seconds
    Initialise receive from process 0
    Wait for a communication from process 0
    Work for 3 seconds
    Initialise receive from process 0
    Wait for a communication from process 0

*/ 


#include <stdio.h>
#include <mpi.h>

void do_something(int world_rank){
    
    printf("El proceso %d esta realizado una tarea ...\n",world_rank);

    for(int i = 0 ; i < 100000000000; i++){
        // No hacer nada
    }
    printf("El proceso %d ha terminado de realizar su tarea...\n",world_rank);
}

int main(int argc, char** argv) {
    
    // Inicializacion MPI
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

    MPI_Request request;
    MPI_Status status;
    int number;

    // Convierte los argumentos de la línea de comandos a enteros cuando se ejecuta el programa
    int num1 = atoi(argv[1]);
    

    if(world_rank == 0 ){
        printf("\nSaludos desde el proceso 0 \n");
        
        // Enviar número al proceso 1
        printf("Se esta enviando los numeros al proceso 1\n");
        MPI_Isend(&num1,1,MPI_INT,1,1,MPI_COMM_WORLD,&request);
        do_something(world_rank);

        MPI_Wait(&request, &status); // Espera a que se complete el envío
        printf("Proceso 0 completó el envío.\n");

    }else if (world_rank == 1){
        printf("Saludos desde el proceso 1 \n");
        // Recibir número del proceso 0
        
        MPI_Irecv(&number, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &request);
        do_something(world_rank);
        
        MPI_Wait(&request, &status); // Espera a que se complete la recepción
        printf("Proceso 1 recibió número %d del proceso 0.\n", number);
        
    
    }
    
    MPI_Finalize();

    return 0;
}