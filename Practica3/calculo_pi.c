#include <stdio.h>
#include <mpi.h>

// Funcion para calcular pi, desde un valor m hasta un valor n
double calculate_pi(int start, int end) {
    double sum = 0.0;
    for (int i = start; i < end; i++) {
        double term = 1.0 / (i * 2 + 1);
        if (i % 2 == 1) {
            term = -term;
        }
        sum += term;
    }
    return sum;
}

int main(int argc, char** argv) {
    int world_size, world_rank;

    // Inicializacion MPI
    MPI_Init(&argc, &argv);
    // Obtiene el número de procesos
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    // Obtiene el número rank del proceso
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    int num_terms = 2000000000; // 2 mil millones
    
    // Numero para saber los intervalos para hacer los calculos
    int terms_per_process = num_terms / world_size;

    // Inicia el cronómetro para obtener el tiempo 
    double start_time = MPI_Wtime();

    int start = world_rank * terms_per_process;
    int end = start + terms_per_process;
    
    // realizar calculo de pi
     
    // Imprimir la parte de la serie que cada proceso está calculando
    printf("Proceso %d calculando desde el término %d hasta %d\n", world_rank, start, end);

    double local_sum = calculate_pi(start, end);
    double pi;

    // Detiene el cronómetro
    double end_time = MPI_Wtime();

    // Tiempo de ejecución por proceso
    double process_time = end_time - start_time;

    // Sumatoria final de todos los resultados
    MPI_Reduce(&z, &pi, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (world_rank == 0) {
        printf("\nInicio de calculo usando %d hilos\n", world_size);
        pi *= 4.0;
        printf("Estimación de Pi: %f\n", pi);

        // Abre un archivo para escribir los tiempos en modo append
        FILE *file = fopen("tiempos.txt", "a");
        if (file != NULL) {
            fprintf(file, "Tiempo total de ejecución con %d procesos: %f segundos\n", world_size, end_time - start_time);
            fclose(file);
        } else {
            printf("Error al abrir el archivo para escribir los tiempos\n");
        }
    }

    MPI_Finalize();
    
    return 0;
}
