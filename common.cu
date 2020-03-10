#include<stdlib.h>
#include<stdio.h>
#include<time.h>
#include "common.h"
/***/
#include<cuda.h>
#include <cuda_runtime.h>
#include<device_launch_parameters.h>
/***/

/**************************/
/***Gaauss Jordan en cpu***/
/**************************/
void GaussJordan_cpu(float* AB, int n, int m, float* X, float time2) {
	cudaEvent_t timer1, timer2;
	cudaEventCreate(&timer1);
	cudaEventCreate(&timer2);
	cudaEventRecord(timer1, 0);
	cudaEventSynchronize(timer1);
	for (int poscol = 0; poscol < n; poscol++) {
		for (int idx = 0; idx < m; idx++) {
			int pospivot = (n + 2) * poscol;
			int posfinfila = (n + 1) * (poscol + 1);
			float piv = AB[pospivot];
			for (int j = pospivot; j < posfinfila; j++) {
				AB[j] = AB[j] / piv;
			}
			int posfactor = pospivot % (n + 1) + idx * (n + 1);
			if (posfactor != pospivot) {
				float factor = AB[posfactor];
				for (int j = pospivot; j < posfinfila; j++) {
					int posactualelim = j % (n + 1) + idx * (n + 1);
					AB[posactualelim] = -
						1 * factor * AB[j] + AB[posactualelim];
				}
			}
		}
		int posultimacol = (poscol + 1) * (n + 1) - 1;
		X[poscol] = AB[posultimacol];
	}
	cudaEventRecord(timer2, 0);
	cudaEventSynchronize(timer1);
	cudaEventSynchronize(timer2);
	cudaEventElapsedTime(&time2, timer1, timer2);
}
void Generador(int tamaño, float* hostPtrL) {
	time_t t;
	srand((unsigned)time(&t));
	for (int i = 0; i < tamaño; i++) {
		hostPtrL[i] = (int)(rand() & 0xFF);
	}
}
void enter() {
	char enter = 0;
	while (enter != '\r' && enter != '\n')
		enter = getchar();
}