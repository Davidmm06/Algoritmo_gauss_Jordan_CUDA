#ifndef COMMON_H
#define COMMON_H

void GaussJordan_cpu(float* AB, int n, int m, float* X, float time2);
void Generador(int tamaño, float* hostPtrL);
void enter();
//void resultado_cpu(float* AB, int n, int m, float* X);
//void Gauss_cpu(int n, int m, float* hostPtrL, float* X);

#endif //!COMMON_H