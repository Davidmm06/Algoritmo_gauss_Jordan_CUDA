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
__global__ void gaussJordan2(float * AB, int n, size_t pitch) {
	int r = blockIdx.x*blockDim.x + threadIdx.x;				
	int c = blockIdx.y*blockDim.y + threadIdx.y;				
	int r2 = c, c2 = r;
	float piv, unos;

	if (r2 < Nrows) {												//Localiza los pivotes solo re realiza el numero de filas que tiene la matriz
		float *pospiv = (float *)((char*)AB + r2 * pitch);
		piv = pospiv[r2];
		float *vuelve_1 = (float *)((char*)AB + r2 * pitch);		//Operacion convierte a 1 los pivotes
		unos = vuelve_1[c2] / piv;
		vuelve_1[c2] = unos;
	}

	for (int r3 = Nrows-1; r3>=0; r3--) {							//Posiciones bajo el pivote solo re realiza el numero de filas que tiene la matriz
		if (c != r3) {												//Si la fila y columna son iguales la brinca
			float *remplazo = (float *)((char*)AB + c * pitch);
			float remp = remplazo[r3];		
			float *vuelve_0 = (float *)((char*)AB + r3* pitch);		//Operacion convierte a 0 los elementos alrededor de los pivotes
			float ceros = -1 * vuelve_0[c] / remp + unos;
			vuelve_0[c] = ceros;
		}
	}
}
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

__global__ void resultado(float *AB, int n, float *X, size_t pitch) {
	int ulcol = Ncols - 1;
	int r = blockDim.x * blockIdx.x + threadIdx.x;
	if (r < n) {
		float *ultimacol = (float *)((char*)AB + r * pitch);
		float element = ultimacol[ulcol];		
		X[r] = element;		
	}
}

void d_mem(float hostPtrAB[Nrows][Ncols],float hostPtrX[Nrows], int n, size_t pitch) {
	int blocksize = 1024;
	int numBlocks = ceil((n) / ((float)blocksize));
	float *devPtrAB;
	cudaMallocPitch(&devPtrAB, &pitch, Ncols * sizeof(float), Nrows);
	cudaMemcpy2D(devPtrAB, pitch, hostPtrAB, Ncols * sizeof(float), Ncols * sizeof(float), Nrows, cudaMemcpyHostToDevice);
	
	float *devPtrX;
	cudaMalloc(&devPtrX, Nrows * sizeof(float));
	
	gaussJordan << <numBlocks, blocksize >> > (devPtrAB, n, pitch);
	cudaDeviceSynchronize();
	cudaMemcpy2D(hostPtrAB, Ncols * sizeof(float), devPtrAB, pitch, Ncols * sizeof(float), Nrows, cudaMemcpyDeviceToHost);

	resultado << <numBlocks, blocksize >> >(devPtrAB, n, devPtrX, pitch);
	cudaDeviceSynchronize();
	cudaMemcpy(hostPtrX, devPtrX, Nrows * sizeof(int), cudaMemcpyDeviceToHost);

	cudaFree(devPtrAB);
	cudaFree(devPtrX);

}

/********/
/* MAIN */
/********/
int main(int argc, char * argv[]) {
	float hostPtrAB[Nrows][Ncols], hostPtrX[Nrows], tempval=0;
	size_t pitch;
	int n = Nrows;
	for (int i = 0; i < Nrows; i++) {
		for (int j = 0; j < Ncols; j++) {
			printf("ingrese el %d coeficiente de la %d ecuación: ", j, i);
			scanf("%f \n\n", tempval);
			hostPtrAB[i][j] = tempval;
		}
	}
	impresion(hostPtrAB);
	//float hostPtr[Nrows][Ncols], hostPtr2[Nrows][Ncols];
	//float *devPtr;
	//size_t pitch;
	//int size = Ncols*Nrows;
	//srand(time(NULL));
	//for (int i = 0; i < Nrows; i++)
	//	for (int j = 0; j < Ncols; j++) {
	//		hostPtr[i][j] = 1 + rand() % (11 - 1);
	//	}
	//impresion (hostPtr);
	/***********************************************/
	/***Carga de matriz a memoria del dispositivo***/			//cambiar a matriz lineal
	/***********************************************/
	
	/******************/
	/******************/
	float* d_AB;
	cudaMalloc((void**)&d_AB,)
	cudaMallocPitch(&d_AB, &pitch, Ncols * sizeof(float), Nrows);
	cudaMemcpy2D(d_AB, pitch, hostPtrAB, Ncols * sizeof(float), Ncols * sizeof(float), Nrows, cudaMemcpyHostToDevice);

	float d_X;
	cudaMalloc(&d_X, Nrows * sizeof(float));

	/****************************/
	/***Lanzamiento del kernel***/
	/****************************/
	int blocksize = 1024;
	int nunBlocks = ceil((n) / ((float)blocksize));
	for (int i = 0; i < n; i++) {
		gaussJordan< << nunBlocks, blocksize >> > (d_AB, n, d_X);
	}
	resultado << <nunBlocks, blocksize >> > (d_AB, n, d_X);

	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
	//gpuErrchk(cudaMemcpy2D(hostPtr2, Ncols * sizeof(float), devPtr, pitch, Ncols * sizeof(float), Nrows, cudaMemcpyDeviceToHost));

	/*******************/

	d_mem(hostPtrAB, hostPtrX, n, pitch);
	impresion(hostPtrAB);
	imprime_resultado(hostPtrX);
	return 0;
}