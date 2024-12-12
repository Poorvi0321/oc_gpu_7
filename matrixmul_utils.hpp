#ifndef MATRIX_MUL_UTILS
#define MATRIX_MUL_UTILS

#include<iostream>
#include<cassert>
#include<ctime>

void init_matrix(int* matrix, int r, int c, int identity=false);

void display_matrix(int* matrix, int r, int c);

void cpu_matmul(int* a, int* b, int* c, int m, int n, int p);
#endif
