
/*******************************************************************************
**                                                                            **
**  Language Learning - Reservoir Computing - GPU                             **
**  An interface for language learning with neuron computing using GPU        **
**  acceleration.                                                             **
**                                                                            **
**  This program is free software: you can redistribute it and/or modify      **
**  it under the terms of the GNU Lesser General Public License as published  **
**  by the Free Software Foundation, either version 3 of the License, or      **
**  (at your option) any later version.                                       **
**                                                                            **
**  This program is distributed in the hope that it will be useful,           **
**  but WITHOUT ANY WARRANTY; without even the implied warranty of            **
**  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the             **
**  GNU Lesser General Public License for more details.                       **
**                                                                            **
**  You should have received a copy of the GNU Lesser General Public License  **
**  along with Foobar.  If not, see <http://www.gnu.org/licenses/>.           **
**                                                                            **
********************************************************************************/

/**
 * \file multiplications.cu
 * \brief defines cuda matrix multiplications functions/
 * \author Florian Lance
 * \date 01/10/14
 */

#include "gpuMat/configCuda.h"

#include <stdio.h>
#include <cublas_v2.h>

// slower than opencv, useless
void vectorSquareMatrixMult(const MatrixD &matA, const MatrixD &vecB, MatrixD &res)
{
    // device pointers
    double *l_dataMatA, *l_dataVecB, *l_dataTemp;

    int l_N = matA.height;
    cudaMalloc((void**)&l_dataMatA, l_N * l_N * sizeof(double));
    cudaMalloc((void**)&l_dataTemp, l_N *     sizeof(double));
    cudaMalloc((void**)&l_dataVecB, l_N *     sizeof(double));

    cublasSetVector(l_N,    sizeof(double), vecB.elements, 1, l_dataVecB, 1);
    cublasSetMatrix(l_N, l_N, sizeof(double), matA.elements, l_N, l_dataMatA, l_N);

    cublasHandle_t l_handle;
    cublasCreate(&l_handle);

    double l_alpha = 1.0f;
    double l_beta  = 0.0f;
    cublasDgemv(l_handle, CUBLAS_OP_T, l_N, l_N, &l_alpha, l_dataMatA, l_N, l_dataVecB, 1, &l_beta, l_dataTemp, 1);
    cublasGetVector(l_N, sizeof(double), l_dataTemp, 1, res.elements, 1);

    cudaFree(l_dataMatA);
    cudaFree(l_dataVecB);
    cudaFree(l_dataTemp);
}




//////////////////////////////////////////////////////
//! Matrix multiplication on the device: C = A * B
//! wA is A's width and wB is B's width
//////////////////////////////////////////////////////
__global__ void
matrixMul( float* C, float* A, float* B, int wA, int wB)
{
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Index of the first sub-matrix of A processed
    // by the block
    int aBegin = wA * BLOCKSIZE * by;

    // Index of the last sub-matrix of A processed
    // by the block
    int aEnd   = aBegin + wA - 1;

    // Step size used to iterate through the
    // sub-matrices of A
    int aStep  = BLOCKSIZE;

    // Index of the first sub-matrix of B processed
    // by the block
    int bBegin = BLOCKSIZE * bx;

    // Step size used to iterate through the
    // sub-matrices of B
    int bStep  = BLOCKSIZE * wB;

    float Csub = 0.f;

    // Loop over all the sub-matrices of A and B
    // required to compute the block sub-matrix
    for (int a = aBegin, b = bBegin;
             a <= aEnd;
             a += aStep, b += bStep)
    {

        // Declaration of the shared memory array As
        // used to store the sub-matrix of A
        __shared__ float As[BLOCKSIZE][BLOCKSIZE];

        // Declaration of the shared memory array Bs
        // used to store the sub-matrix of B
        __shared__ float Bs[BLOCKSIZE][BLOCKSIZE];

        // Load the matrices from global memory
        // to shared memory; each thread loads
        // one element of each matrix
        As[ty][tx] = A[a + wA * ty + tx];
        Bs[ty][tx] = B[b + wB * ty + tx];

        // Synchronize to make sure the matrices
        // are loaded
        __syncthreads();

        // Multiply the two matrices together;
        // each thread computes one element
        // of the block sub-matrix
        for (int k = 0; k < BLOCKSIZE; ++k)
            Csub += As[ty][k] * Bs[k][tx];

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();

    }

    // Write the block sub-matrix to device memory;
    // each thread writes one element
    int c = wB * BLOCKSIZE * by + BLOCKSIZE * bx;
    C[c + wB * ty + tx] = Csub;

}


 void matMult3(const Matrix &A, const Matrix &B, Matrix &C)
 {

   // 1. allocate host memory for matrices A and B
   unsigned int size_A = A.width * A.height;
   unsigned int mem_size_A = sizeof(float) * size_A;
//    float* h_A = (float*) malloc(mem_size_A);

   unsigned int size_B = B.width * B.height;
   unsigned int mem_size_B = sizeof(float) * size_B;
//    float* h_B = (float*) malloc(mem_size_B);

   // 2. initialize host memory
//    randomInit(h_A, size_A);
//    randomInit(h_B, size_B);

   // 8. allocate device memory
   float* d_A;
   float* d_B;
   cudaMalloc((void**) &d_A, mem_size_A);
   cudaMalloc((void**) &d_B, mem_size_B);

   // 9. copy host memory to device
   cudaMemcpy(d_A, A.elements, mem_size_A,
   cudaMemcpyHostToDevice);
   cudaMemcpy(d_B, B.elements, mem_size_B,
   cudaMemcpyHostToDevice);

   // 4. allocate host memory for the result C
   unsigned int size_C = C.width * C.height;
   unsigned int mem_size_C = sizeof(float) * size_C;
//    float* h_C = (float*) malloc(mem_size_C);

   // 10. allocate device memory for the result
   float* d_C;
   cudaMalloc((void**) &d_C, mem_size_C);

   // 5. perform the calculation
   // setup execution parameters
   dim3 threads(BLOCKSIZE, BLOCKSIZE);
   dim3 grid(C.width / threads.x, C.height / threads.y);

   // execute the kernel
   matrixMul<<< grid, threads >>>(d_C, d_A,
                                  d_B, A.width, B.width);

   cudaFree(d_A);
   cudaFree(d_B);

   // 11. copy result from device to host
   cudaMemcpy(C.elements, d_C, mem_size_C,
   cudaMemcpyDeviceToHost);

   // 7. clean up memory
   cudaFree(d_C);
}







 //////////////////////////////////////////////////////
 //! Matrix multiplication on the device: C = A * B
 //! wA is A's width and wB is B's width
 //////////////////////////////////////////////////////
 __global__ void
 matrixMul( double* C, double* A, double* B, int wA, int wB)
 {
     // Block index
     int bx = blockIdx.x;
     int by = blockIdx.y;

     // Thread index
     int tx = threadIdx.x;
     int ty = threadIdx.y;

     // Index of the first sub-matrix of A processed
     // by the block
     int aBegin = wA * BLOCKSIZE * by;

     // Index of the last sub-matrix of A processed
     // by the block
     int aEnd   = aBegin + wA - 1;

     // Step size used to iterate through the
     // sub-matrices of A
     int aStep  = BLOCKSIZE;

     // Index of the first sub-matrix of B processed
     // by the block
     int bBegin = BLOCKSIZE * bx;

     // Step size used to iterate through the
     // sub-matrices of B
     int bStep  = BLOCKSIZE * wB;

     double Csub = 0.f;

     // Loop over all the sub-matrices of A and B
     // required to compute the block sub-matrix
     for (int a = aBegin, b = bBegin;
              a <= aEnd;
              a += aStep, b += bStep)
     {

         // Declaration of the shared memory array As
         // used to store the sub-matrix of A
         __shared__ double As[BLOCKSIZE][BLOCKSIZE];

         // Declaration of the shared memory array Bs
         // used to store the sub-matrix of B
         __shared__ double Bs[BLOCKSIZE][BLOCKSIZE];

         // Load the matrices from global memory
         // to shared memory; each thread loads
         // one element of each matrix
         As[ty][tx] = A[a + wA * ty + tx];
         Bs[ty][tx] = B[b + wB * ty + tx];

         // Synchronize to make sure the matrices
         // are loaded
         __syncthreads();

         // Multiply the two matrices together;
         // each thread computes one element
         // of the block sub-matrix
         for (int k = 0; k < BLOCKSIZE; ++k)
             Csub += As[ty][k] * Bs[k][tx];

         // Synchronize to make sure that the preceding
         // computation is done before loading two new
         // sub-matrices of A and B in the next iteration
         __syncthreads();

     }

     // Write the block sub-matrix to device memory;
     // each thread writes one element
     int c = wB * BLOCKSIZE * by + BLOCKSIZE * bx;
     C[c + wB * ty + tx] = Csub;

 }


 void matMult3D(const MatrixD &A, const MatrixD &B, MatrixD &C)
 {

   // 1. allocate host memory for matrices A and B
   unsigned int size_A = A.width * A.height;
   unsigned int mem_size_A = sizeof(double) * size_A;

   unsigned int size_B = B.width * B.height;
   unsigned int mem_size_B = sizeof(double) * size_B;

   // 8. allocate device memory
   double* d_A;
   double* d_B;
   cudaMalloc((void**) &d_A, mem_size_A);
   cudaMalloc((void**) &d_B, mem_size_B);

   // 9. copy host memory to device
   cudaMemcpy(d_A, A.elements, mem_size_A,
   cudaMemcpyHostToDevice);
   cudaMemcpy(d_B, B.elements, mem_size_B,
   cudaMemcpyHostToDevice);

   // 4. allocate host memory for the result C
   unsigned int size_C = C.width * C.height;
   unsigned int mem_size_C = sizeof(double) * size_C;

   // 10. allocate device memory for the result
   double* d_C;
   cudaMalloc((void**) &d_C, mem_size_C);

   // 5. perform the calculation
   // setup execution parameters
   dim3 threads(BLOCKSIZE, BLOCKSIZE);
   dim3 grid(C.width / threads.x, C.height / threads.y);

   // execute the kernel
   matrixMul<<< grid, threads >>>(d_C, d_A,
                                  d_B, A.width, B.width);

   cudaFree(d_A);
   cudaFree(d_B);

   // 11. copy result from device to host
   cudaMemcpy(C.elements, d_C, mem_size_C,
   cudaMemcpyDeviceToHost);

   // 7. clean up memory
   cudaFree(d_C);
}


