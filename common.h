#ifndef COMMON_H
#define COMMON_H

float GaussJordan_cpu(float* AB, int n, int m, float* X);
void Generador(int tamaño, float* hostPtrL);
//void resultado_cpu(float* AB, int n, int m, float* X);
//void Gauss_cpu(int n, int m, float* hostPtrL, float* X);

#endif //!COMMON_H