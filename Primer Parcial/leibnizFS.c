#include <stdio.h>
#include <omp.h>
#include <time.h>

#define NUM_THREADS 5
#define IT 2e9

long double init,end;
long double total_time;

long double sums[NUM_THREADS][64];
long double pi = 0;
long int start[NUM_THREADS];
long int finish[NUM_THREADS];

void main(){
    for(int j=0; j<NUM_THREADS;j++){
        start[j]= (int) (IT/NUM_THREADS)*j;
        finish[j] = (int) (IT/NUM_THREADS)*(j+1)-1;
    }
    init = omp_get_wtime();
    #pragma omp parallel num_threads(NUM_THREADS)
    {
        int ID = omp_get_thread_num();

        for(long int i = start[ID]; i<=finish[ID]; i++){
            sums[ID][0] += (i&1)? ((long double) -4/(long double) (2*i+1)): ((long double) 4/(long double) (2*i+1));
        }
    }
    end = omp_get_wtime();
    total_time = end - init;

    for(int j=0; j<NUM_THREADS;j++){
        pi+=sums[j][0];
    }
    printf("%.15Lf\n", pi);
    printf("%d , %Lf \n", NUM_THREADS,total_time);
}
