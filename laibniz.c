#include <stdio.h>
#include <omp.h>
#include <time.h>

#define NUM_THREADS 16
#define IT 2e9

clock_t init,end;
double total_time;

long double sums[NUM_THREADS][64];
long double pi = 0;
int start[NUM_THREADS];
int finish[NUM_THREADS];

void main(){
    for(int j=0; j<NUM_THREADS;j++){
        start[j]= (int) (IT/NUM_THREADS)*j;
        finish[j] = (int) (IT/NUM_THREADS)*(j+1)-1;
    }
    init = clock();
    #pragma omp parallel num_threads(NUM_THREADS)
    //printf("%lf \n", ((double)(clock() - init))/CLOCKS_PER_SEC);
    {
        int ID = omp_get_thread_num();

        for(int i = start[ID]; i<=finish[ID]; i++){
            sums[ID][0] += (i&1)? ((long double) -4/(long double) (2*i+1)): ((long double) 4/(long double) (2*i+1));
        }
    }
    end = clock();
    total_time = ((double)(end - init))/CLOCKS_PER_SEC;

    for(int j=0; j<NUM_THREADS;j++){
        pi+=sums[j][0];
        //printf("%d  %d\n", start[j], finish[j]);
    }
    printf("%.15Lf\n", pi);
    printf("%d , %lf \n", NUM_THREADS,total_time);
}
