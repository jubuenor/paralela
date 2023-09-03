#include <stdio.h>
#include <omp.h>

#define NUM_THREADS 1
#define IT 2e9

long double init,end;
long double total_time;

long double sums[NUM_THREADS];
long double pi = 0;
long int start[NUM_THREADS];
long int finish[NUM_THREADS];

void main(){

    for(int j=0; j<NUM_THREADS;j++){
        start[j]= (int) (IT/NUM_THREADS)*j;
        finish[j] = (int) (IT/NUM_THREADS)*(j+1)-1;
    }

    omp_set_num_threads(NUM_THREADS);
    init = omp_get_wtime();
    #pragma omp parallel
    {
        int ID = omp_get_thread_num();

        for(long int i = start[ID]; i<=finish[ID]; i++){
            sums[ID] += (i&1)? ((long double) -4/(long double) (2*i+1)): ((long double) 4/(long double) (2*i+1));
        }
    }
    end = omp_get_wtime();
    total_time = end - init;

    for(int j=0; j<NUM_THREADS;j++){
        pi+=sums[j];
    }
    printf("%.15Lf\n", pi);
    printf("%d , %Lf \n", NUM_THREADS,total_time);
}
