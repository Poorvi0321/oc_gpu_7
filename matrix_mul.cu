#include"matmul_utils.hpp"

#define M 3
#define N 3
#define P 3
#define Q 3

__global__ void matmulKernel(int* a, int*b, int* c)
{
    int i = blockIdx.x;
    int j = blockIdx.y;
    c[i*Q + j] = 0;

    for(int k=0; k<N; ++k)
    {
        c[i*Q + j] += (a[i*N + k] * b[k*Q + j]);
    }
}

int main()
{
    int* a = new int[M * N * sizeof(int)];
    int* b = new int[P * Q * sizeof(int)];
    int* c = new int[M * Q * sizeof(int)];
    init_matrix(a, M, N);
    init_matrix(b, P, Q);

    std::cout<<"A =\n";
    display_matrix(a, M, N);
    std::cout<<"B =\n";
    display_matrix(b, P, Q);

    assert(N==P); 

    int *d_a, *d_b, *d_c;
    cudaMalloc((void**)&d_a, M * N * sizeof(int));
    cudaMalloc((void**)&d_b, P * Q * sizeof(int));
    cudaMalloc((void**)&d_c, M * Q * sizeof(int));

    cudaMemcpy(d_a, a, M * N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, P * Q * sizeof(int), cudaMemcpyHostToDevice);
    
    dim3 grid(M, Q);

    matmulKernel<<<grid, 1>>>(d_a, d_b, d_c);

    cudaMemcpy(c, d_c, M * Q * sizeof(int), cudaMemcpyDeviceToHost);
    std::cout<<"A*B =\n";
    display_matrix(c, M, Q);

    delete a;
    delete b;
    delete c;

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return 0;
}
