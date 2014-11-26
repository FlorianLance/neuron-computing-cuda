
/**
 * \file inversions.cu
 * \brief defines cuda matrix inversion functions/
 * \author Florian Lance
 * \date 01/10/14
 */

#include "cula.h"

#include "gpuMat/configCuda.h"


#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/copy.h>


int culaWarmup(int gpuDevice)
{
    int cudaMinimumVersion   = culaGetCudaMinimumVersion();
    int cudaRuntimeVersion   = culaGetCudaRuntimeVersion();
    int cudaDriverVersion    = culaGetCudaDriverVersion();
    int cublasMinimumVersion = culaGetCublasMinimumVersion();
    int cublasRuntimeVersion = culaGetCublasRuntimeVersion();

    if(cudaRuntimeVersion < cudaMinimumVersion)
    {
        printf("-1\n %d", cudaMinimumVersion);
        return -1;
    }

    if(cudaDriverVersion < cudaMinimumVersion)
    {
        printf("-2\n %d", cudaMinimumVersion);
        return -1;
    }

    if(cublasRuntimeVersion < cublasMinimumVersion)
    {
        printf("-3\n %d", cublasMinimumVersion);
        return -1;
    }

    culaStatus l_oStatus;
    char l_buf[256];
    int l_info;

    culaSelectDevice(gpuDevice);
    l_oStatus = culaInitialize();

    if(l_oStatus != culaNoError)
    {
        l_info = culaGetErrorInfo();

        culaGetErrorInfoString(l_oStatus, l_info, l_buf, sizeof(l_buf));

        printf("%s\n", culaGetStatusString(l_oStatus));
        printf("%s\n", l_buf);
        return -1;
    }

    return 0;
}

void culaStop()
{
    culaShutdown();
}

// ############################################################################################# SVD DECOMPOSITION

bool svdDecomposition(float* data, int m, int n, float* S, float* VT, float* U)
{
    const int l_count = m * n;
    const int l_minDim = std::min(m,n);
    const char l_jobu  = 'A';
    const char l_jobvt = 'A';

    culaStatus l_status;

    thrust::device_vector<float> l_data(data, data + l_count);
    thrust::device_vector<float> l_U(m * m);
    thrust::device_vector<float> l_sigma(l_minDim);
    thrust::device_vector<float> l_Vt(n * n);

    if ((l_status = culaDeviceSgesvd(l_jobu, l_jobvt, m, n, l_data.data().get(), m, l_sigma.data().get(),l_U.data().get(), m, l_Vt.data().get(), n)) != culaNoError)
    {
        printf("%s\n", culaGetStatusString(l_status));
        delete[] data;
        data = NULL;
        return false;
    }

    delete[] data;
    data = NULL;

    thrust::copy(l_U.begin(), l_U.end(), U);
    thrust::copy(l_Vt.begin(), l_Vt.end(), VT);
    thrust::copy(l_sigma.begin(), l_sigma.end(), S);

    return true;
}

bool svdDecomposition_all(float* dataOverwrittenVt, int m, int n, float* S, float* U)
{
    const int l_count = m * n;
    const int l_minDim = std::min(m,n);

    culaStatus l_status;
    thrust::device_vector<float> l_dataOvt(dataOverwrittenVt, dataOverwrittenVt + l_count);
    thrust::device_vector<float> l_U(m * m);
    thrust::device_vector<float> l_sigma(l_minDim);

    if ((l_status = culaDeviceSgesvd('A', 'O', m, n, l_dataOvt.data().get(), m, l_sigma.data().get(),l_U.data().get(), m, NULL, n)) != culaNoError)
    {
        printf("%s\n", culaGetStatusString(l_status));
        return false;
    }

    thrust::copy(l_dataOvt.begin(), l_dataOvt.end(), dataOverwrittenVt);
    thrust::copy(l_U.begin(), l_U.end(), U);
    thrust::copy(l_sigma.begin(), l_sigma.end(), S);

    return true;
}

bool svdDecomposition_Vt_S(float* dataOverwrittenVt, float *S, int m, int n)
{
    const int l_count = m * n;
    const int l_minDim = std::min(m,n);

    culaStatus l_status;
    thrust::device_vector<float> l_dataOvt(dataOverwrittenVt, dataOverwrittenVt + l_count);
    thrust::device_vector<float> l_sigma(l_minDim);

    if ((l_status = culaDeviceSgesvd('O', 'N', m, n, l_dataOvt.data().get(), m, l_sigma.data().get(),NULL, m, NULL, n)) != culaNoError)
    {
        printf("%s\n", culaGetStatusString(l_status));
        return false;
    }

    thrust::copy(l_dataOvt.begin(), l_dataOvt.end(), dataOverwrittenVt);
    thrust::copy(l_sigma.begin(), l_sigma.end(), S);

    return true;
}

bool svdDecomposition_U_S(float* dataOverwrittedU, float *S, int m, int n)
{
    const int l_count = m * n;
    const int l_minDim = std::min(m,n);

    culaStatus l_status;
    thrust::device_vector<float> l_dataOU(dataOverwrittedU, dataOverwrittedU + l_count);
    thrust::device_vector<float> l_sigma(l_minDim);

    if ((l_status = culaDeviceSgesvd('N', 'O', m, n, l_dataOU.data().get(), m, l_sigma.data().get(),NULL, m, NULL, n)) != culaNoError)
    {
        printf("%s\n", culaGetStatusString(l_status));
        return false;
    }

    thrust::copy(l_dataOU.begin(), l_dataOU.end(), dataOverwrittedU);
    thrust::copy(l_sigma.begin(), l_sigma.end(), S);

    return true;
}

bool svdDecomposition_S(float* data, float *S, int m, int n)
{
    const int l_count = m * n;
    const int l_minDim = std::min(m,n);

    culaStatus l_status;
    thrust::device_vector<float> l_data(data, data + l_count);
    thrust::device_vector<float> l_sigma(l_minDim);

    if ((l_status = culaDeviceSgesvd('N', 'N', m, n, l_data.data().get(), m, l_sigma.data().get(),NULL, m, NULL, n)) != culaNoError)
    {
        printf("%s\n", culaGetStatusString(l_status));
        return false;
    }

    thrust::copy(l_sigma.begin(), l_sigma.end(), S);

    // JOBU != ‘O’ and JOBVT != ‘O’, the contents of A are destroyed.
    data = NULL;

    return true;
}


// ############################################################################################# LU INVERSION

//// Get a matrix element
//__device__ float GetElement(const Matrix &A, int row, int col)
//{
//    return A.elements[row * A.stride + col];
//}

//// Get a matrix element
//__device__ float GetElement(const MatrixD &A, int row, int col)
//{
//    return A.elements[row * A.stride + col];
//}

//// Set a matrix element
//__device__ void SetElement(Matrix &A, int row, int col,
//                           float value)
//{
//    A.elements[row * A.stride + col] = value;
//}

//// Set a matrix element
//__device__ void SetElement(MatrixD &A, int row, int col,
//                           float value)
//{
//    A.elements[row * A.stride + col] = value;
//}


//// Get the BLOCK_SIZExBLOCK_SIZE sub-matrix Asub of A that is
//// located col sub-matrices to the right and row sub-matrices down
//// from the upper-left corner of A
// __device__ Matrix GetSubMatrix(Matrix &A, int row, int col)
//{
//    Matrix Asub;
//    Asub.width    = BLOCKSIZE;
//    Asub.height   = BLOCKSIZE;
//    Asub.stride   = A.stride;
//    Asub.elements = &A.elements[A.stride * BLOCKSIZE * row
//                                         + BLOCKSIZE * col];
//    return Asub;
//}

// // Get the BLOCK_SIZExBLOCK_SIZE sub-matrix Asub of A that is
// // located col sub-matrices to the right and row sub-matrices down
// // from the upper-left corner of A
//  __device__ MatrixD GetSubMatrix(MatrixD &A, int row, int col)
// {
//     MatrixD Asub;
//     Asub.width    = BLOCKSIZE;
//     Asub.height   = BLOCKSIZE;
//     Asub.stride   = A.stride;
//     Asub.elements = &A.elements[A.stride * BLOCKSIZE * row
//                                          + BLOCKSIZE * col];
//     return Asub;
// }

//// Matrix multiplication kernel called by MatMul()
// __global__ void MatMulKernel(Matrix &A, Matrix &B, Matrix &C)
//{
//    // Block row and column
//    int blockRow = blockIdx.y;
//    int blockCol = blockIdx.x;

//    // Each thread block computes one sub-matrix Csub of C
//    Matrix Csub = GetSubMatrix(C, blockRow, blockCol);

//    // Each thread computes one element of Csub
//    // by accumulating results into Cvalue
//    float Cvalue = 0;

//    // Thread row and column within Csub
//    int row = threadIdx.y;
//    int col = threadIdx.x;

//    // Loop over all the sub-matrices of A and B that are
//    // required to compute Csub
//    // Multiply each pair of sub-matrices together
//    // and accumulate the results
//    for (int m = 0; m < (A.width / BLOCKSIZE); ++m) {

//        // Get sub-matrix Asub of A
//        Matrix Asub = GetSubMatrix(A, blockRow, m);

//        // Get sub-matrix Bsub of B
//        Matrix Bsub = GetSubMatrix(B, m, blockCol);

//        // Shared memory used to store Asub and Bsub respectively
//        __shared__ float As[BLOCKSIZE][BLOCKSIZE];
//        __shared__ float Bs[BLOCKSIZE][BLOCKSIZE];

//        // Load Asub and Bsub from device memory to shared memory
//        // Each thread loads one element of each sub-matrix
//        As[row][col] = GetElement(Asub, row, col);
//        Bs[row][col] = GetElement(Bsub, row, col);

//        // Synchronize to make sure the sub-matrices are loaded
//        // before starting the computation
//        __syncthreads();

//        // Multiply Asub and Bsub together
//        for (int e = 0; e < BLOCKSIZE; ++e)
//            Cvalue += As[row][e] * Bs[e][col];

//        // Synchronize to make sure that the preceding
//        // computation is done before loading two new
//        // sub-matrices of A and B in the next iteration
//        __syncthreads();
//    }

//    // Write Csub to device memory
//    // Each thread writes one element
//    SetElement(Csub, row, col, Cvalue);
//}

// // Matrix multiplication kernel called by MatMul()
//  __global__ void MatMulKernel(MatrixD &A, MatrixD &B, MatrixD &C)
// {
//     // Block row and column
//     int blockRow = blockIdx.y;
//     int blockCol = blockIdx.x;

//     // Each thread block computes one sub-matrix Csub of C
//     MatrixD Csub = GetSubMatrix(C, blockRow, blockCol);

//     // Each thread computes one element of Csub
//     // by accumulating results into Cvalue
//     float Cvalue = 0;

//     // Thread row and column within Csub
//     int row = threadIdx.y;
//     int col = threadIdx.x;

//     // Loop over all the sub-matrices of A and B that are
//     // required to compute Csub
//     // Multiply each pair of sub-matrices together
//     // and accumulate the results
//     for (int m = 0; m < (A.width / BLOCKSIZE); ++m) {

//         // Get sub-matrix Asub of A
//         MatrixD Asub = GetSubMatrix(A, blockRow, m);

//         // Get sub-matrix Bsub of B
//         MatrixD Bsub = GetSubMatrix(B, m, blockCol);

//         // Shared memory used to store Asub and Bsub respectively
//         __shared__ double As[BLOCKSIZE][BLOCKSIZE];
//         __shared__ double Bs[BLOCKSIZE][BLOCKSIZE];

//         // Load Asub and Bsub from device memory to shared memory
//         // Each thread loads one element of each sub-matrix
//         As[row][col] = GetElement(Asub, row, col);
//         Bs[row][col] = GetElement(Bsub, row, col);

//         // Synchronize to make sure the sub-matrices are loaded
//         // before starting the computation
//         __syncthreads();

//         // Multiply Asub and Bsub together
//         for (int e = 0; e < BLOCKSIZE; ++e)
//             Cvalue += As[row][e] * Bs[e][col];

//         // Synchronize to make sure that the preceding
//         // computation is done before loading two new
//         // sub-matrices of A and B in the next iteration
//         __syncthreads();
//     }

//     // Write Csub to device memory
//     // Each thread writes one element
//     SetElement(Csub, row, col, Cvalue);
// }

// // Matrix multiplication - Host code
// // Matrix dimensions are assumed to be multiples of BLOCK_SIZE
// void matMult(const Matrix &A, const Matrix &B, Matrix &C, const int blockSize)
// {
//     // Load A and B to device memory
//     Matrix d_A;
//     d_A.width = d_A.stride = A.width; d_A.height = A.height;
//     size_t size = A.width * A.height * sizeof(float);
//     cudaMalloc(&d_A.elements, size);
//     cudaMemcpy(d_A.elements, A.elements, size,
//                cudaMemcpyHostToDevice);
//     Matrix d_B;
//     d_B.width = d_B.stride = B.width; d_B.height = B.height;
//     size = B.width * B.height * sizeof(float);
//     cudaMalloc(&d_B.elements, size);
//     cudaMemcpy(d_B.elements, B.elements, size,
//     cudaMemcpyHostToDevice);

//     // Allocate C in device memory
//     Matrix d_C;
//     d_C.width = d_C.stride = C.width; d_C.height = C.height;
//     size = C.width * C.height * sizeof(float);
//     cudaMalloc(&d_C.elements, size);

//     // Invoke kernel
//     dim3 dimBlock(blockSize, blockSize);
//     dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y);
//     MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);

//     // Read C from device memory
//     cudaMemcpy(C.elements, d_C.elements, size,
//                cudaMemcpyDeviceToHost);

//     // Free device memory
//     cudaFree(d_A.elements);
//     cudaFree(d_B.elements);
//     cudaFree(d_C.elements);
// }

// // Matrix multiplication - Host code
// // Matrix dimensions are assumed to be multiples of BLOCK_SIZE
// void matMultD(const MatrixD &A, const MatrixD &B, MatrixD &C, const int blockSize)
// {
//     // Load A and B to device memory
//     MatrixD d_A;
//     d_A.width = d_A.stride = A.width; d_A.height = A.height;
//     size_t size = A.width * A.height * sizeof(double);
//     cudaMalloc(&d_A.elements, size);
//     cudaMemcpy(d_A.elements, A.elements, size,
//                cudaMemcpyHostToDevice);
//     MatrixD d_B;
//     d_B.width = d_B.stride = B.width; d_B.height = B.height;
//     size = B.width * B.height * sizeof(double);
//     cudaMalloc(&d_B.elements, size);
//     cudaMemcpy(d_B.elements, B.elements, size,
//     cudaMemcpyHostToDevice);

//     // Allocate C in device memory
//     MatrixD d_C;
//     d_C.width = d_C.stride = C.width; d_C.height = C.height;
//     size = C.width * C.height * sizeof(double);
//     cudaMalloc(&d_C.elements, size);

//     // Invoke kernel
//     dim3 dimBlock(blockSize, blockSize);
//     dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y);
//     MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);

//     // Read C from device memory
//     cudaMemcpy(C.elements, d_C.elements, size,
//                cudaMemcpyDeviceToHost);

//     // Free device memory
//     cudaFree(d_A.elements);
//     cudaFree(d_B.elements);
//     cudaFree(d_C.elements);
// }
