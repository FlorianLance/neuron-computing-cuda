
/**
 * \file cudaInversions.h
 * \brief defines cuda matrix inversion interface functions
 * \author Florian Lance
 * \date 1/10/14
 */

#ifndef _GPUMATUTILITY_
#define _GPUMATUTILITY_

// CUDA
#include "gpuMat/configCuda.h"

// OPENCV
#include "opencv2/imgproc/imgproc.hpp"

/**
 * @brief culaWarmup
 * @param [in] gpuDevice : index of the GPU device
 * @return -1 if error during the initialization else return 0
 */
int culaWarmup(int gpuDevice);

/**
 * @brief culaStop
 */
void culaStop();

/**
 * @brief svdDecomposition
 * @param [in,out] data :
 * @param [in] m        : ...
 * @param [in] n        : ...
 * @param [in,out] S    : ...
 * @param [in,out] VT   : ...
 * @param [in,out] U    : ...
 * @return true if success else return false
 */
bool svdDecomposition(float* data, int m, int n, float* S, float* VT, float* U);

/**
 * @brief svdDecomposition_all
 * @param [in,out] dataOverwrittenVt    :
 * @param [in] m                        :
 * @param [in] n                        :
 * @param [in,out] S                    : ...
 * @param [in,out] U                    : ...
 * @return true if success else return false
 */
bool svdDecomposition_all(float* dataOverwrittenVt, int m, int n, float* S, float* U);

/**
 * @brief svdDecomposition_Vt_S
 * @param dataOverwrittenVt
 * @param S
 * @param m
 * @param n
 * @return
 */
bool svdDecomposition_Vt_S(float* dataOverwrittenVt, float *S, int m, int n);

/**
 * @brief svdDecomposition_U_S
 * @param dataOverwrittedU
 * @param S
 * @param m
 * @param n
 * @return
 */
bool svdDecomposition_U_S(float* dataOverwrittedU, float *S, int m, int n);

/**
 * @brief svdDecomposition_S
 * @param data
 * @param S
 * @param m
 * @param n
 * @return
 */
bool svdDecomposition_S(float* data, float *S, int m, int n);

namespace swCuda
{
    /**
     * @brief squareMatrixSingularValueDecomposition
     * @param M
     * @param S
     * @param U
     * @param VT
     * @return
     */
    static bool squareMatrixSingularValueDecomposition(const cv::Mat &M, cv::Mat &S, cv::Mat &U, cv::Mat &VT)
    {
        // check depth input data
            bool l_32b = false;
            if(M.depth() == CV_32FC1)
            {
                l_32b = true;
            }

        // input array data
            float *l_M  = new float[M.rows * M.cols];

            if(l_32b)
            {
                for(int ii = 0; ii < M.rows; ++ii)
                {
                    for(int jj = 0; jj < M.cols; ++jj)
                    {
                        l_M[jj*M.rows + ii] = M.at<float>(ii,jj);
                    }
                }
            }
            else
            {
                for(int ii = 0; ii < M.rows; ++ii)
                {
                    for(int jj = 0; jj < M.cols; ++jj)
                    {
                        l_M[jj*M.rows + ii] = static_cast<float>(M.at<double>(ii,jj));
                    }
                }
            }

            int l_minDim = std::min(M.cols, M.rows);

        // output arrays data
            float *l_S = NULL,*l_U = NULL,*l_VT = NULL;
            l_S  = new float[l_minDim * sizeof(float)];
            l_U  = new float[M.rows * M.rows * sizeof(float)];
            l_VT = new float[M.cols * M.cols * sizeof(float)];

        // singular value decompostion with cuda
            if(!svdDecomposition(l_M, M.rows, M.cols, l_S, l_VT, l_U))
            {
                delete[] l_S; l_S = NULL;
                delete[] l_U; l_U = NULL;
                delete[] l_VT; l_VT = NULL;
                delete[] l_M; l_M = NULL;
                return false;
            }

        // fill result mat
            if(l_32b)
            {
                S   = cv::Mat(M.rows,M.cols, CV_32FC1, cv::Scalar(0.f));
                    for(int ii = 0; ii < l_minDim; ++ii)
                    {
                        S.at<float>(ii,ii) =l_S[ii];
                    }
                delete[] l_S;

                U   = cv::Mat(M.rows,M.rows, CV_32FC1);
                    for(int ii = 0; ii < M.rows; ++ii)
                    {
                        for(int jj = 0; jj < M.rows; ++jj)
                        {
                            U.at<float>(ii,jj)  = l_U[jj*M.rows + ii];
                        }
                    }
                delete[] l_U;

                VT  = cv::Mat(M.cols,M.cols, CV_32FC1);
                    for(int ii = 0; ii < M.cols; ++ii)
                    {
                        for(int jj = 0; jj < M.cols; ++jj)
                        {
                            VT.at<float>(ii,jj) = l_VT[jj*M.cols + ii];
                        }
                    }
                delete[] l_VT;
            }
            else
            {
                S   = cv::Mat(M.rows,M.cols, CV_64FC1, cv::Scalar(0.0));
                    for(int ii = 0; ii < l_minDim; ++ii)
                    {
                        S.at<double>(ii,ii) = static_cast<double>(l_S[ii]);
                    }
                delete[] l_S;

                U   = cv::Mat(M.rows,M.rows, CV_64FC1);
                    for(int ii = 0; ii < M.rows; ++ii)
                    {
                        for(int jj = 0; jj < M.rows; ++jj)
                        {
                            U.at<double>(ii,jj)  = static_cast<double>(l_U[jj*M.rows + ii]);
                        }
                    }
                delete[] l_U;

                VT  = cv::Mat(M.cols,M.cols, CV_64FC1);
                    for(int ii = 0; ii < M.cols; ++ii)
                    {
                        for(int jj = 0; jj < M.cols; ++jj)
                        {
                            VT.at<double>(ii,jj) = static_cast<double>(l_VT[jj*M.cols + ii]);
                        }
                    }
                delete[] l_VT;
            }


        return true;
    }

    /**
     * @brief squareMatrixSingularValueDecomposition
     * @param M
     * @param S
     * @param U
     * @param VT
     * @return
     */
    static bool new_squareMatrixSingularValueDecomposition(const cv::Mat &M, cv::Mat &SUt, cv::Mat &VT)
    {
        // check depth input data
            bool l_32b = false;
            if(M.depth() == CV_32FC1)
            {
                l_32b = true;
            }

        // input array data
            float *l_dataM_oVt  = new float[M.rows * M.cols];

            if(l_32b)
            {
                for(int ii = 0; ii < M.rows; ++ii)
                {
                    for(int jj = 0; jj < M.cols; ++jj)
                    {
                        l_dataM_oVt[jj*M.rows + ii] = M.at<float>(ii,jj);
                    }
                }
            }
            else
            {
                for(int ii = 0; ii < M.rows; ++ii)
                {
                    for(int jj = 0; jj < M.cols; ++jj)
                    {
                        l_dataM_oVt[jj*M.rows + ii] = static_cast<float>(M.at<double>(ii,jj));
                    }
                }
            }

            int l_minDim = std::min(M.cols, M.rows);

        // output arrays data
            float *l_S = NULL,*l_U = NULL;
            l_S  = new float[l_minDim * sizeof(float)];
            l_U  = new float[M.rows * M.rows * sizeof(float)];

        // singular value decompostion with cuda
            svdDecomposition_all(l_dataM_oVt, M.rows, M.cols, l_S, l_U);

        // fill result mat
            if(l_32b)
            {
                SUt   = cv::Mat(M.rows,M.rows, CV_32FC1);
                    for(int ii = 0; ii < M.rows; ++ii)
                    {
                        for(int jj = 0; jj < M.rows; ++jj)
                        {
                            SUt.at<float>(jj,ii)  = l_U[jj*M.rows + ii];
                        }
                    }
                delete[] l_U;

                VT  = cv::Mat(M.cols,M.cols, CV_32FC1);
                    for(int ii = 0; ii < M.cols; ++ii)
                    {
                        for(int jj = 0; jj < M.cols; ++jj)
                        {
                            VT.at<float>(ii,jj) = l_dataM_oVt[jj*M.cols + ii];
                        }
                    }
                delete[] l_dataM_oVt;
            }
            else
            {
                SUt   = cv::Mat(M.rows,M.rows, CV_64FC1);
                    for(int ii = 0; ii < M.rows; ++ii)
                    {
                        for(int jj = 0; jj < M.rows; ++jj)
                        {
                            SUt.at<double>(jj,ii)  = static_cast<double>(l_U[jj*M.rows + ii]);
                        }
                    }
                delete[] l_U;

                VT  = cv::Mat(M.cols,M.cols, CV_64FC1);
                    for(int ii = 0; ii < M.cols; ++ii)
                    {
                        for(int jj = 0; jj < M.cols; ++jj)
                        {
                            VT.at<double>(ii,jj) = static_cast<double>(l_dataM_oVt[jj*M.cols + ii]);
                        }
                    }
                delete[] l_dataM_oVt;
            }


            for(int ii= 0; ii < l_minDim ;++ii)
            {
                SUt.col(ii) *= l_S[ii];
            }


        return true;
    }


    /**
     * @brief squareMatrixSingularValueDecomposition
     * @param M
     * @param S
     * @param U
     * @param VT
     * @return
     */
    template<typename T>
    static bool low_memory_squareMatrixSingularValueDecomposition(cv::Mat &M, cv::Mat &SUt, cv::Mat &VT)
    {
        // check depth input data
            bool l_32b = false;
            if(M.depth() == CV_32FC1)
            {
                l_32b = true;
            }

        // input array data
            float *l_data  = new float[M.rows * M.cols];
            for(int ii = 0; ii < M.rows; ++ii)
            {
                for(int jj = 0; jj < M.cols; ++jj)
                {
                    l_data[jj*M.rows + ii] = static_cast<float>(M.at<T>(ii,jj));
                }
            }

            int l_minDim = std::min(M.cols, M.rows);

        // output arrays data
            float *l_S = NULL;
            l_S  = new float[l_minDim * sizeof(float)];

        // compute U and S
            if(!svdDecomposition_U_S(l_data, l_S, M.rows, M.cols))
            {
                delete[] l_data;
                delete[] l_S;
                return false;
            }

            if(l_32b)
            {
                SUt   = cv::Mat(M.rows,M.rows, CV_32FC1);
            }
            else
            {
                SUt   = cv::Mat(M.rows,M.rows, CV_64FC1);
            }

            for(int ii = 0; ii < M.rows; ++ii)
            {
                for(int jj = 0; jj < M.rows; ++jj)
                {
                    // invert jj and ii for retrieving the U^t
                    SUt.at<T>(jj,ii)  = static_cast<T>(l_data[jj*M.rows + ii]);
                }
            }

            // comput U^tt * Sigma
            for(int ii= 0; ii < l_minDim ;++ii)
            {
                SUt.col(ii) *= static_cast<float>(l_S[ii]);
            }

            // reinit data
            for(int ii = 0; ii < M.rows; ++ii)
            {
                for(int jj = 0; jj < M.cols; ++jj)
                {
                    l_data[jj*M.rows + ii] = static_cast<float>(M.at<T>(ii,jj));
                }
            }

        // compute VT and S
            if(!svdDecomposition_Vt_S(l_data, l_S, M.rows, M.cols))
            {
                delete[] l_data;
                delete[] l_S;
                return false;
            }

            delete[] l_S; // Sigma doesn't needed anymore

            if(l_32b)
            {
                VT   = cv::Mat(M.rows,M.rows, CV_32FC1);
            }
            else
            {
                VT   = cv::Mat(M.rows,M.rows, CV_64FC1);
            }

            for(int ii = 0; ii < M.cols; ++ii)
            {
                for(int jj = 0; jj < M.cols; ++jj)
                {
                    VT.at<T>(ii,jj) = static_cast<float>(l_data[jj*M.cols + ii]);
                }
            }

            delete[] l_data;


        return true;
    }
}


#endif


