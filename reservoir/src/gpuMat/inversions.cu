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
 * \file inversions.cu
 * \brief defines cuda matrix inversion functions/
 * \author Florian Lance
 * \date 01/10/14
 */

#include "cula.h"

#include "gpuMat/configCuda.h"


#include <stdio.h>
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

        printf("culaWarmup : %s\n", culaGetStatusString(l_oStatus));
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
        printf("svdDecomposition : %s\n", culaGetStatusString(l_status));
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
        printf("svdDecomposition_all : %s\n", culaGetStatusString(l_status));
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
        printf("svdDecomposition_Vt_S : %s\n", culaGetStatusString(l_status));
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
        printf("svdDecomposition_U_S : %s\n", culaGetStatusString(l_status));
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
        printf("svdDecomposition_S : %s\n", culaGetStatusString(l_status));
        return false;
    }

    thrust::copy(l_sigma.begin(), l_sigma.end(), S);

    // JOBU != ‘O’ and JOBVT != ‘O’, the contents of A are destroyed.
    data = NULL;

    return true;
}
