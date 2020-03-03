#include<stdio.h>
#include<math.h>
#include<cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include<device_launch_parameters.h>

float* generar(int tamaño) {
	float* a = (float*)malloc(tamaño * sizeof(float));

	printf("tamaño total %d\n", tamaño);
	srand(123);

	// Populate arrays with values between about -10000 and 10000
	// Distribution is not uniform
	for (unsigned int i = 0; i < tamaño; i++) {
		do {
			a[i] = (rand() % 20000) - 10000;
			//printf("alemento: %d\n", i);
		} while ((int)a[i] == 0);
	}

	float* elements = (float*)malloc(tamaño * sizeof(float));

	for (unsigned int i = 0; i < tamaño; i++)
		*(elements + i) = (float)a[i];
	return elements;
}