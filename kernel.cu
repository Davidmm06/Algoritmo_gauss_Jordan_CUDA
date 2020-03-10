#define _CRT_SECURE_NO_WARNINGS
#include<stdio.h>
#include<math.h>
#include<cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include<device_launch_parameters.h>
#include "CUDA_Error_handling.cuh"
#include "common.h"
#define Nrows 3
#define Ncols 4

void enter() {
	char enter = 0;
	while (enter != '\r' && enter != '\n')
		enter = getchar();
}

void impresion1(float Z[Nrows][Ncols]) {									//imprime la matriz
	for (int i = 0; i < Nrows; i++) {
		for (int j = 0; j < Ncols; j++) {
			printf("%8.4f ", Z[i][j]);
		}
		printf("\n\n");
	}
}

void conversion(float Y[Nrows * Ncols], float Z[Nrows][Ncols]) {
	for (int i = 0; i < Nrows; i++) {
		for (int j = 0; j < Ncols; j++) {
				Z[i][j] = Y[i *Ncols + j];
		}
	}
}

void impresion(float Y[Nrows*Ncols]) {									//imprime la matriz
	for (int i = 0; i < Nrows * Ncols; i++) {
		printf("%8.4f ", Y[i]);
		if (i % Ncols == 0) {
			printf("\n");
		}
	}
}

/**************************/
/***Lectura de la matriz***/
/**************************/
void matrix_read(float* L, int dimension) {
	FILE* fp;
	int row, col;
	fp = fopen("matriz500.txt", "r");										//open output file
	if (fp == NULL)															//open failed
		return;
	for (row = 0; row < dimension; row++) {									//!feof(fp)
		for (col = 0; col < dimension; col++)
			if (fscanf(fp, "%f,", &L[row * dimension + col]) == EOF) break; //read data

		if (feof(fp)) break;												//if the file is over
	}
	fclose(fp);																//close file
}

void imprime_resultado(float X[Nrows]) {
	for (int i = 0; i < Nrows; i++) {
		printf("%f ", X[i]);
	}
}

void imprime_resultadosvect(float X[Nrows]) {
	for (int i = 0; i < Nrows; i++) {
		printf("%f ", X[i]);
		printf("\n");
	}
}

void imprime_resultadosvect1(float X[Nrows]) {
	for (int i = 0; i < Nrows; i++) {
		printf("elemento: %d | %-16.4f\n", i, X[i]);
	}
}

/**************/
/***Modelo 1***/
/**************/
__global__ void gaussJordan1(float* AB, size_t pitch) {
	int r = blockDim.y * blockIdx.y + threadIdx.y;										//for (int f = 0; f <= fil - 1; f += 1)	equivalencia en for
	int c = blockDim.x * blockIdx.x + threadIdx.x;										//for (int c = 0; c <= col - 1; c += 1)	if (id >= n - 1 - poscol)
	if ((r < Nrows) && (c < Ncols)) {
		float* inicial = (float*)((char*)AB + r * pitch);
		float piv = inicial[r];															//Posicion pivote
		float* horizontal = (float*)((char*)AB + r * pitch);
		float pospivot = horizontal[c];													//recorre la matriz horizontalmemte
		//printf("la posicion es %f y el pivote es %f\n", pospivot, piv);				//comienza
		if ((r == 0) && (c == 0)) {
			for (int j = 0; j < Ncols; j++) {
				horizontal[j] = pospivot / piv;
				printf("check");
			}
		}

		/***GaussJordan***/
		for (int fil = 0; fil < Nrows; fil++) {											//vuelve 0s
			if ((fil > 0) && (c >= 0)) {												//Exeptua primer pivote
				if ((fil =!c) || (c =! (Ncols - 1))) {									//exepciones a eliminar pivote y resultados
					float* posfactor = (float*)((char*)AB + c * pitch);					//posicion bajo el pivote forma vertical
					float factor = posfactor[fil];										//factor son las pocisiones a eliminar de la matriz
					printf("Posicion del factor %f", factor);
					//printf("el factor es %f", factor);								//El factor no cambia hasta que cambia de fila o columna
					for (int col = 0; col < Ncols; col++) {								//Ya que la restriccion de los ceros es una matriz cuadrada se ocupa un nuevo contador
						float* horizontal2 = (float*)((char*)AB + fil * pitch);			//operaciones recorrido en forma horizontal
						float pospivot2 = horizontal2[col];								//eliminacion
						horizontal2[col] = (pospivot2 / factor) - pospivot2;			//operacion
					}
				}
				else if (fil == c) {													//vuelve 1s del segundo pivote en adelante
					float* inicial = (float*)((char*)AB + c * pitch);
					float pivote = inicial[c];											//posicion pivote
					float* horizontal2 = (float*)((char*)AB + fil * pitch);				//operaciones recorrido en forma horizontal
					float pospivot2 = horizontal2[c];									//cambio a 1s
					horizontal2[c] = pospivot2 / pivote;								//operacion
				}
			}
		}
		///***prueba***/
		////if ((r = !0) && (c = !0)){		
		//for (int fil = 0; fil < Nrows; fil++) {										//??????? esta se movera mas rapido
		//	for (int col = 0; col < Ncols-1; col++){
		//		//if ((fil > 0) && (c >= 0)) {										//agregar exepciones de factor
		//		if ((fil = !col)||(col==Ncols-1)) {
		//			float* posfactor = (float*)((char*)AB + col * pitch);		//posicion bajo el pivote forma vertical
		//			float factor = posfactor[fil];								// posfactor vuelve 0s					
		//			printf("\n\nel factor es = %f\n", factor);
		//			float* horizontal2 = (float*)((char*)AB + fil * pitch);			//operaciones recorrido en forma horizontal
		//			float pospivot2 = horizontal2[c];
		//			horizontal2[c] = (pospivot2 / factor) - pospivot;
		//		}
		//	}
		//}
	}
}

/**************/
/***Modelo 2***/
/**************/
__global__ void eliminacionAdelante(float* AB, int n, int poscol) {
	int id = blockDim.x * blockIdx.x + threadIdx.x;
	if (id >= n - 1 - poscol) {					//realiza el procedimiento debajo del pivote
		return;
	}
	int pospivot = (n + 2) * poscol	;			//posicion del pivoten en el vector
	int posfinfila = (n + 1) * (poscol + 1);	//posicion final de la fila en el vector
	float piv = AB[pospivot];					
	for (int j = pospivot; j < posfinfila; j++) {
		AB[j] = AB[j] / piv;
		//printf("pospivote = %f\n\n", AB[j]);
	}
	int posfactor = pospivot + (n + 1) * (id + 1);//posiciones bajo el pivote
		float factor = AB[posfactor];
	for (int j = pospivot; j < posfinfila; j++) {
		int posactualelim = j + (n + 1) * (id + 1);
		AB[posactualelim] = -1 * factor * AB[j] + AB[posactualelim];
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
		//printf("posicion pivote = %f\n\n", AB[pospivot]);
	}
	float factor = AB[pospivot - (n + 1) * (id + 1)];
	int posactualelim1 = pospivot - (n + 1) * (id + 1);
	int posactualelim2 = pospivot - (n + 1) * (id + 1) + 1 + poscol;
	AB[posactualelim1] = -1 * factor * AB[pospivot] + AB[posactualelim1];
	AB[posactualelim2] = -1 * factor * AB[pospivot + 1 + poscol] + AB[posactualelim2];
}

/**************/
/***Modelo 3***/
/**************/
__global__ void GaussJordan(float* AB, int n, int poscol) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;	
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
		float factor = AB[posfactor];
		for (int j = pospivot; j < posfinfila; j++) {
			int posactualelim = j % (n + 1) + idx * (n + 1);
			AB[posactualelim] = -
				1 * factor * AB[j] + AB[posactualelim];
		}
	}
}

/****************/
/***Resultados***/
/****************/
__global__ void resultado1(float* AB, float* X, size_t pitch) {
	int r = blockDim.y * blockIdx.y + threadIdx.y;		//for (int f = 0; f <= fil - 1; f += 1)	equivalencia en for
	if (r < Nrows) {
		float* posfinfila = (float*)((char*)AB + (Ncols-1) * pitch);
		float finfila = posfinfila[r];
		X[r] = finfila;
	}
}
__global__ void resultado(float* AB, int n, float* X) {
	int id = blockDim.x * blockIdx.x + threadIdx.x;
	if (id < n) {
		int posultimacol = (id + 1) * (n + 1) - 1;
		X[id] = AB[posultimacol];
	}
}

int iDivUp(int hostPtr, int b) { return ((hostPtr % b) != 0) ? (hostPtr / b + 1) : (hostPtr / b); }

/********/
/* MAIN */
/********/
int main(int argc, char * argv[]) {
	float hostPtrAB2[Nrows][Ncols], hostPtrZ2[Nrows][Ncols], hostPtrZ[Ncols * Nrows], hostPtrX[Nrows], time, time2, hostPtrXcpu[Nrows];
	float* hostPtrL = new float[Nrows * Ncols], *hostPtrcpu = new float[Nrows * Ncols];	//vector de lectura
	int n = Nrows;														//filas
	int m = Ncols;														//columnas
	int tamaño = Nrows * Ncols;											//tanaño de la matriz
	int block_size = 1024;												//hilos por bloque
	int size1=Nrows * sizeof(float);									//tamaño del vector de resultados
	int size = Nrows * Ncols * sizeof(float);							//tamaño de la matriz en bits
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	//Generador(tamaño, hostPtrL);
	//printf("tamaño: %d", tamaño);
	printf("leyendo matriz");
	matrix_read(hostPtrL, m);											//hostPtr matriz leida de un archivo .txt
	printf("matriz origen\n\n");
	//conversion(hostPtrL, hostPtrAB2);
	//impresion1(hostPtrAB2);
	enter();
	float hostPtrAB[Ncols * Nrows] = { 2,-1, 1, 2,						//Matriz de prueba predefinida 4x3
									   3, 1,-2, 9,
									  -1, 2, 5,-5 };	

	/***********************************************/
	/***Carga de matriz a memoria del dispositivo***/
	/***********************************************/
	//Matriz AB		Matriz original
	float* d_AB;
	cudaMalloc((void**)&d_AB, size);
	cudaMemcpy(d_AB, hostPtrL, size, cudaMemcpyHostToDevice);


	//Matriz Z		Matriz unitaria
	float* d_Z;					
	cudaMalloc((void**)&d_Z, size);
	cudaMemcpy(d_Z, hostPtrZ, size, cudaMemcpyHostToDevice);


	//Matriz X		vector de resultados
	float *d_X;
	cudaMalloc((void**)&d_X, size1);

	cudaEventCreate(&start);
	cudaEventRecord(start, 0);

	/************************************/
	/***Lanzamiento del kernel Modelo1***/
	/************************************/
	//int blocksize = 1024;
	//for (int i = 0; i < n - 1; i++) {
	//	int numBlocks = ceil((n - 1 - i) / ((float)blocksize));
	//	eliminacionAdelante << <numBlocks, blocksize >> >(d_AB, n, i);
	//	//gpuErrchk(cudaDeviceSynchronize());
	//}
	//for (int i = 0; i < n - 1; i++) {
	//	int numBlocks = ceil((n - 1 - i) / ((float)blocksize));
	//	eliminacionAtras << <numBlocks, blocksize >> > (d_AB, n, i);
	//	//gpuErrchk(cudaDeviceSynchronize());
	//}
	//int nbloques = ceil(n / ((float)blocksize));
	//resultado << <nbloques, blocksize >> > (d_AB, n, d_X);
	/************************************/
	/***Lanzamiento del kernel Modelo3***/
	/************************************/
	//int	tamaño_de_bloque = ((tamaño / 1024) + 1);
	//printf("tamaño: %d", tamaño_de_bloque);
	dim3 block (32,32);													//hilos porv bloque		1024
	dim3 grid ((tamaño / block.x)+1, (tamaño / block.y) + 1);			//numero de bloques		suponiendo500 => (250 500) + 1 = 245 => bloques en total	40*1024 =40960 (32*32=1024) 245
	for (int i = 0; i < n; i++) {										//											1024
		GaussJordan << <grid, block >> > (d_AB, n, i);
	}
	resultado << <grid, block >> > (d_AB, n, d_X);	

	gpuErrchk(cudaDeviceSynchronize());
	//cudaEventRecord(stop, 0);
	//cudaEventSynchronize(stop);
	//cudaEventElapsedTime(&time, start, stop);
	//cudaEventDestroy(start);
	//cudaEventDestroy(stop);
	cudaEventCreate(&stop);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);

	cudaMemcpy(hostPtrX, d_X, size1, cudaMemcpyDeviceToHost);
	cudaMemcpy(hostPtrZ, d_AB, size, cudaMemcpyDeviceToHost);
	cudaDeviceReset();

	/***********************************/
	/***Algoritmo gauss jordan en cpu***/
	/***********************************/
	GaussJordan_cpu(hostPtrL, n, m, hostPtrXcpu, time2);

	/*************************/
	/***Tiempo de respuesta***/
	/*************************/
	printf("\n\nCuda Time: %f ms\n", time);
	printf("\n\n");

	printf("             CPU | GPU             \n");
	printf("-----------------+-----------------\n");
	imprime_resultadosvect1(hostPtrX);
	printf("Pulse enter para ver resultados completos...\n");
	//printf("numero de bloques: %8.4f", numBlocks);
	enter();

	/*****************************/
	/***impresion de resultados***/
	/*****************************/
	printf("matriz original\n\n");
	conversion(hostPtrL, hostPtrAB2);
	impresion1(hostPtrAB2);

	printf("matriz clon\n\n");
	conversion(hostPtrZ, hostPtrZ2);
	impresion1(hostPtrZ2);

	/***********************/
	/***liberando memoria***/
	/***********************/
	cudaFree(d_AB);
	cudaFree(d_X);
	cudaFree(d_Z);

	return 0;
}

