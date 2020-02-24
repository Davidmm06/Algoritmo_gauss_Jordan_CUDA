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
#define BLOCKSIZE_x 16
#define BLOCKSIZE_y 16
#define Nrows 3
#define Ncols 4

void impresion1(float Z[Nrows][Ncols]) {									//imprime la matriz
	for (int i = 0; i < Nrows; i++) {
		for (int j = 0; j < Ncols; j++) {
			printf("%f ", Z[i][j]);
		}
		printf("\n\n");
	}
}

void impresion(float Y[Nrows*Ncols]) {									//imprime la matriz
	for (int i = 0; i < Nrows * Ncols; i++) {
		printf("%f ", Y[i]);
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
__global__ void gaussJordan(float* AB, size_t pitch) {
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
	printf("Posoivot %d\n\n", pospivot);
	int posfinfila = (n + 1) * (poscol + 1);	//posicion final de la fila en el vector
	float piv = AB[pospivot];					
	printf("piv %f\n\n", piv);					
	for (int j = pospivot; j < posfinfila; j++) {
		AB[j] = AB[j] / piv;
		printf("pospivote = %f\n\n", AB[j]);
	}
	int posfactor = pospivot + (n + 1) * (id + 1);//posiciones bajo el pivote
		float factor = AB[posfactor];
		printf("factor %f\n\n", factor);
	for (int j = pospivot; j < posfinfila; j++) {
		int posactualelim = j + (n + 1) * (id + 1);
		AB[posactualelim] = -1 * factor * AB[j] + AB[posactualelim];
		printf("posactualelim %f\n\n", AB[posactualelim]);
	}
}
__global__ void eliminacionAtras(float* AB, int n, int poscol) {
	int id = blockDim.x * blockIdx.x + threadIdx.x;
	if (id >= n - 1 - poscol) {
		return;
	}
	printf("matrix2\n\n");
	int pospivot = (n + 2) * (n - 1 - poscol);
	printf("Posoivot %d\n\n", pospivot);
	if (poscol == 0) {
		float pivot = AB[pospivot];
		AB[pospivot] = AB[pospivot] / pivot;
		AB[pospivot + 1] = AB[pospivot + 1] / pivot;
		printf("posicion pivote = %f\n\n", AB[pospivot]);
	}
	float factor = AB[pospivot - (n + 1) * (id + 1)];
	printf("factor %f\n\n", factor);
	int posactualelim1 = pospivot - (n + 1) * (id + 1);
	int posactualelim2 = pospivot - (n + 1) * (id + 1) + 1 + poscol;
	AB[posactualelim1] = -1 * factor * AB[pospivot] + AB[posactualelim1];
	printf("posactualelim1 %f\n\n", AB[posactualelim1]);
	AB[posactualelim2] = -1 * factor * AB[pospivot + 1 + poscol] + AB[posactualelim2];
	printf("posactualelim2 %f\n\n", AB[posactualelim2]);

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
	//float hostPtrAB[Nrows][Ncols], hostPtrZ[Nrows][Ncols], hostPtrX[Nrows], tempval=0;
	float hostPtrZ[Ncols * Nrows], hostPtrX[Ncols];
	size_t pitch;	
	int size1=Nrows * sizeof(float);
	int size = Nrows * Ncols * sizeof(float);
	int n = Nrows;

	float hostPtrAB[Ncols * Nrows] = { 2,-1, 1, 2,
									   3, 1,-2, 9,
									  -1, 2, 5,-5 };			//{2, 1, 0, 6, 5, -2}; //
	/*for (int i = 0; i < Nrows; i++) {
		for (int j = 0; j < Ncols; j++) {
			fprintf(stdout, "ingrese el %d coeficiente de la %d ecuacion: ", j, i);
			scanf_s("%f", &tempval);
			hostPtrAB[i][j] = tempval;
		}
	}
	impresion(hostPtrAB);*/									//check
	/***********************************************/
	/***Carga de matriz a memoria del dispositivo***/
	/***********************************************/
	//Matriz AB
	float* d_AB;
	//gpuErrchk(cudaMallocPitch(&d_AB, &pitch, Ncols * sizeof(float), Nrows));
	//cudaMemcpy2D(d_AB, pitch, hostPtrAB, Ncols * sizeof(float), Ncols * sizeof(float), Nrows, cudaMemcpyHostToDevice);
	cudaMalloc((void**)&d_AB, size);
	cudaMemcpy(d_AB, hostPtrAB, size, cudaMemcpyHostToDevice);


	//Matriz Z
	float* d_Z;
	cudaMalloc((void**)&d_Z, size);
	cudaMemcpy(d_Z, hostPtrZ, size, cudaMemcpyHostToDevice);


	//Matriz X
	float *d_X;
	cudaMalloc((void**)&d_X, size1);

	/****************************/
	/***Lanzamiento del kernel Modelo1***/
	/****************************/
	//dim3 gridSize(iDivUp(Ncols, BLOCKSIZE_x), iDivUp(Nrows, BLOCKSIZE_y));
	//dim3 blockSize(BLOCKSIZE_y, BLOCKSIZE_x);
	//gaussJordan << <gridSize, blockSize >> > (d_AB, pitch);
	//cudaDeviceSynchronize();
	//resultado << <gridSize, blockSize >> > (d_AB, d_X, pitch);
	//gpuErrchk(cudaPeekAtLastError());
	//gpuErrchk(cudaDeviceSynchronize());

	//cudaMemcpy2D(hostPtrZ, Ncols * sizeof(float), d_AB, pitch, Ncols * sizeof(float), Nrows, cudaMemcpyDeviceToHost);
	//cudaMemcpy(hostPtrX, d_X, size1, cudaMemcpyDeviceToHost);

	//printf("matriz original\n\n");
	//impresion(hostPtrAB);
	//printf("Matriz clonada\n\n");
	//impresion(hostPtrZ);
	//printf("resultados\n\n");
	//imprime_resultado(hostPtrX);
	//cudaDeviceReset();

	/************************************/
	/***Lanzamiento del kernel Modelo1***/
	/************************************/
	int blocksize = 1024;
	for (int i = 0; i < n - 1; i++) {
		int numBlocks = ceil((n - 1 - i) / ((float)blocksize));
		eliminacionAdelante << <numBlocks, blocksize >> >(d_AB, n, i);
		//gpuErrchk(cudaDeviceSynchronize());
	}
	for (int i = 0; i < n - 1; i++) {
		int numBlocks = ceil((n - 1 - i) / ((float)blocksize));
		eliminacionAtras << <numBlocks, blocksize >> > (d_AB, n, i);
		//gpuErrchk(cudaDeviceSynchronize());
	}
	int nbloques = ceil(n / ((float)blocksize));
	resultado << <nbloques, blocksize >> > (d_AB, n, d_X);
	gpuErrchk(cudaDeviceSynchronize());

	cudaMemcpy(hostPtrX, d_X, size1, cudaMemcpyDeviceToHost);
	cudaMemcpy(hostPtrZ, d_AB, size, cudaMemcpyDeviceToHost);
	cudaDeviceReset();

	printf("matriz original\n\n");
	impresion(hostPtrAB);
	printf("\n\n");
	
	printf("matriz clon\n\n");
	impresion(hostPtrZ);
	printf("\n\n");
	
	printf("resultados\n\n");
	imprime_resultado(hostPtrX);
	printf("\n\n");


	/***********************/
	/***liberando memoria***/
	/***********************/
	cudaFree(d_AB);
	cudaFree(d_X);
	return 0;
}