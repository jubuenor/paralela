#include <stdio.h>
#include <omp.h>
#include <time.h>

#define NUM_THREADS 16
#define IT 2e9

clock_t init,end;
double total_time;

long double sums[NUM_THREADS];
long double pi = 0;
long int start[NUM_THREADS];
long int finish[NUM_THREADS];

void main(){
    for(int j=0; j<NUM_THREADS;j++){
        start[j]= (int) (IT/NUM_THREADS)*j;
        finish[j] = (int) (IT/NUM_THREADS)*(j+1)-1;
    }
    init = clock();
    #pragma omp parallel num_threads(NUM_THREADS)
    printf(".");
    {
        int ID = omp_get_thread_num();

        for(long int i = start[ID]; i<=finish[ID]; i++){
            sums[ID] += (i&1)? ((long double) -4/(long double) (2*i+1)): ((long double) 4/(long double) (2*i+1));
        }
    }
    end = clock();
    total_time = ((double)(end - init))/CLOCKS_PER_SEC;

    for(int j=0; j<NUM_THREADS;j++){
        pi+=sums[j];
        //printf("%d  %d\n", start[j], finish[j]);
    }
    printf("%.15Lf\n", pi);
    printf("%d , %lf \n", NUM_THREADS,total_time);
}
