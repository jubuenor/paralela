#include <stdio.h>
#include <omp.h>

#define NUM_THREADS 12
#define IT 2e9

long double init,end;
long double total_time;

double sums[NUM_THREADS];
double pi = 0;
long int start[NUM_THREADS];
long int finish[NUM_THREADS];
long double time[NUM_THREADS];

void main(){

    for(int j=0; j<NUM_THREADS;j++){
        start[j]= (int) (IT/NUM_THREADS)*j;
        finish[j] = (int) (IT/NUM_THREADS)*(j+1)-1;
    }
    printf("%d\n",omp_get_max_threads());
    omp_set_num_threads(NUM_THREADS);
    init = omp_get_wtime();
    #pragma omp parallel
    {
        int ID = omp_get_thread_num();
        time[ID] = -omp_get_wtime();
        for(long int i = start[ID]; i<=finish[ID]; i++){
            sums[ID] += (i&1)? ((double) -4/(double) (2*i+1)): ((double) 4/(double) (2*i+1));
        }
        time[ID] += omp_get_wtime();
    }
    end = omp_get_wtime();
    total_time = end - init;

    for(int j=0; j<NUM_THREADS;j++){
        pi+=sums[j];
    }

    for(int j=0; j<NUM_THREADS;j++){
        printf("Hilo %d: %Lf \n", j, time[j]);
    }

    printf("%.15f\n", pi);
    printf("Número de Hilos: %d \nTiempo total: %Lf \n", NUM_THREADS,total_time);

}
