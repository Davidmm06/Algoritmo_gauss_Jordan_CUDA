/*
Creado por David Mendieta Morales				 
				 Escuela Superior de Ingenieria Mecanica y Electrica
				 Instituto Politecnico Nacional - Unidad Culhuacan
Ciudad de Mexico 2020

OPEN SOURCE CODE - Libre modificación y distribución

*/
#include<stdio.h>
#include<cuda.h>
#include<cuda_runtime.h>
#include<device_launch_parameters.h>
#include<math.h>
#include "CUDA_Error_handling.cuh"

#define Nrows 3
#define Ncols 4

void impresion(float Z[Nrows][Ncols]) {									//imprime la matriz
	for (int i = 0; i < Nrows; i++) {
		for (int j = 0; j < Ncols; j++) {
			printf("%f ", Z[i][j]);
		}
		printf("\n\n");
	}
}

void imprime_resultado(float X[Nrows]) {
	for (int i = 0; i < Nrows; i++) {
		printf("%f ", X[i]);
	}
}

/**************/
/***Modelo 1***/
/**************/
__global__ void eliminacionAdelante(float* AB, int n, int poscol) {
	int id = blockDim.x * blockIdx.x + threadIdx.x;
	if (id >= n - 1 - poscol) {
		return;
	}
	int pospivot = (n + 2) * poscol;
	int posfinfila = (n + 1) * (poscol + 1);
	float piv = AB[pospivot];
	for (int j = pospivot; j < posfinfila; j++) {
		AB[j] = AB[j] / piv;
	}
	int posfactor = pospivot + (n + 1) * (id + 1);//posiciones bajo el pivot
		float factor = AB[posfactor];
	for (int j = pospivot; j < posfinfila; j++) {
		int posactualelim = j + (n + 1) * (id + 1);
		AB[posactualelim] = -1 * factor * AB[j] +
			AB[posactualelim];
	}
}
__global__ void eliminacionAtras(float* AB, int n, int poscol) {
	int id = blockDim.x * blockIdx.x + threadIdx.x;
	if (id >= n - 1 - poscol) {
		return;
	}
	int pospivot = (n + 2) * (n - 1 - poscol);
	if (poscol == 0) {
		float pivot = AB[pospivot];
		AB[pospivot] = AB[pospivot] / pivot;
		AB[pospivot + 1] = AB[pospivot + 1] / pivot;
	}
	float factor = AB[pospivot - (n + 1) * (id + 1)];
	int posactualelim1 = pospivot - (n + 1) * (id + 1);
	int posactualelim2 = pospivot - (n + 1) * (id + 1) + 1 + poscol;
	AB[posactualelim1] = -1 * factor * AB[pospivot] + AB[posactualelim1];
	AB[posactualelim2] = -1 * factor * AB[pospivot + 1 + poscol] + AB[posactualelim2];
}

/**************/
/***Modelo 2***/
/**************/
__global__ void gaussJordan(float* AB, int n, int poscol) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= n) {
		return;
	}
	int pospivot = (n + 2) * poscol;
	int posfinfila = (n + 1) * (poscol + 1);
	float piv = AB[pospivot];
	for (int j = pospivot; j < posfinfila; j++) {
		AB[j] = AB[j] / piv;
	}
	int posfactor = pospivot % (n + 1) + idx * (n + 1);
	if (posfactor != pospivot) {
		float factor =AB[posfactor];
		for (int j = pospivot; j < posfinfila; j++) {
			int posactualelim = j % (n + 1) + idx * (n + 1);
			AB[posactualelim] = -1 * factor * AB[j] + AB[posactualelim];
		}
	}
}

/****************/
/***Resultados***/
/****************/
__global__ void resultado(float* AB, int n, float* X) {
	int id = blockDim.x * blockIdx.x + threadIdx.x;
	if (id < n) {
		int posultimacol = (id + 1) * (n + 1) - 1;
		X[id] = AB[posultimacol];
	}
}

/********/
/* MAIN */
/********/
int main(int argc, char * argv[]) {
	float hostPtrAB[Nrows][Ncols], hostPtrX[Nrows], tempval=0;
	size_t pitch;												//no estoy seguro si vaya lo ocupo para matrices dinamicas
	int n = Nrows;
	int size = Nrows * Ncols * sizeof(float);

	for (int i = 0; i < Nrows; i++) {
		for (int j = 0; j < Ncols; j++) {
			fprintf(stdout, "ingrese el %d coeficiente de la %d ecuacion: ", j, i);
			scanf_s("%f", &tempval);
			hostPtrAB[i][j] = tempval;
		}
	}
	impresion(hostPtrAB);									//check
	/***********************************************/
	/***Carga de matriz a memoria del dispositivo***/
	/***********************************************/

	float* d_AB;
	gpuErrchk(cudaMalloc((void**)&d_AB, size));
	gpuErrchk(cudaMemcpy(d_AB, hostPtrAB, size, cudaMemcpyHostToDevice));

	float *d_X;
	gpuErrchk(cudaMalloc((void**)&d_X, Nrows * sizeof(float)));

	/****************************/
	/***Lanzamiento del kernel***/
	/****************************/
	int blocksize = 1024;
	int numBlocks = ceil((n) / ((float)blocksize));
	for (int i = 0; i < n; i++) {
		gaussJordan << <numBlocks, blocksize >> > (d_AB, n, i);
	}
	resultado << <numBlocks, blocksize >> > (d_AB, n, d_X);

	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
	//gpuErrchk(cudaMemcpy2D(hostPtr2, Ncols * sizeof(float), devPtr, pitch, Ncols * sizeof(float), Nrows, cudaMemcpyDeviceToHost));

	cudaMemcpy(hostPtrX, d_X, size, cudaMemcpyDeviceToHost);

	impresion(hostPtrAB);
	imprime_resultado(hostPtrX);

	/***********************/
	/***liberando memoria***/
	/***********************/
	cudaFree(d_AB);
	cudaFree(d_X);
	free(hostPtrAB);
	free(hostPtrX);
	return 0;
}