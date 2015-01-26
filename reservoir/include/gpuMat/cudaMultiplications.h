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
 * \file cudaMultiplications.h
 * \brief defines cuda matrix multiplication interface functions
 * \author Florian Lance
 * \date 1/10/14
 */

#ifndef CUDAMULTIPLICATIONS_H
#define CUDAMULTIPLICATIONS_H


// CUDA
#include "gpuMat/configCuda.h"

// OPENCV
#include "opencv2/imgproc/imgproc.hpp"

/**
 * @brief vectorSquareMatrixMult
 * @param matA
 * @param vecB
 * @param res
 */
void vectorSquareMatrixMult(const MatrixD &matA, const MatrixD &vecB, MatrixD &res);

/**
 * @brief matMult
 * @param A
 * @param B
 * @param C
 * @param blockSize
 */
void matMult(const Matrix &A, const Matrix &B, Matrix &C, const int blockSize = 16);

/**
 * @brief matMultD
 * @param A
 * @param B
 * @param C
 * @param blockSize
 */
void matMultD(const MatrixD &A, const MatrixD &B, MatrixD &C, const int blockSize = 16);

/**
 * @brief matMult3
 * @param A
 * @param B
 * @param C
 */
void matMult3(const Matrix &A, const Matrix &B, Matrix &C);

/**
 * @brief matMult3D
 * @param A
 * @param B
 * @param C
 */
void matMult3D(const MatrixD &A, const MatrixD &B, MatrixD &C);

namespace swCuda
{
        /**
         * @brief vectorSquareMatrixMultiplication
         * @param matA
         * @param vecB
         * @param resC
         */
        static void vectorSquareMatrixMultiplicationD(const cv::Mat &matA, const cv::Mat &vecB, cv::Mat &resC)
        {
            if(matA.depth() != CV_64F ||  vecB.depth() != CV_64F)
            {
                std::cerr << "-ERROR : vectorSquareMatrixMultiplication -> input depth data must be 64 bits. " << std::endl;
                return;
            }

            if(matA.rows != matA.cols)
            {
                std::cerr << "-ERROR : vectorSquareMatrixMultiplication -> matA is not square. " << std::endl;
                return;
            }

            if(vecB.rows != 1 && vecB.cols != 1)
            {
                std::cerr << "-ERROR : vectorSquareMatrixMultiplication -> vecB is not a 1D vector. " << std::endl;
                return;
            }

            if(vecB.rows != matA.rows && vecB.cols != matA.rows)
            {
                std::cerr << "-ERROR : vectorSquareMatrixMultiplication -> matA and vecB cannot be multiplied. " << std::endl;
                return;
            }

            MatrixD A, B, C;
            A.height = matA.rows;
            A.width  = matA.cols;
            A.elements = new double[A.width * A.height];

            B.height = vecB.rows;
            B.width  = vecB.cols;
            B.elements = new double[B.width * B.height];

            C.height  = A.height;
            C.width   = B.width;
            C.elements= new double[C.width * C.height];

            for(int ii = 0; ii < A.height; ++ii)
            {
                for(int jj = 0; jj < A.width; ++jj)
                {
                    A.elements[ii*A.width + jj] = matA.at<double>(ii,jj);
                }

                B.elements[ii] = vecB.at<double>(ii);
            }

            // call cuda function
            vectorSquareMatrixMult(A,B,C);
            delete[] A.elements;
            delete[] B.elements;

            resC = cv::Mat(matA.rows, 1, CV_64FC1);

            for(int ii = 0; ii < matA.rows; ++ii)
            {
                resC.at<double>(ii) = C.elements[ii];
            }

            delete[] C.elements;
        }

        /**
         * \brief GPU matrix multiplication. Res(l,n) = A(l,m)*B(m,n)
         * \param [in]  oMatA    : input A matrix
         * \param [in]  oMatB    : input B matrix
         * \param [out] oMatRes  : res C matrix
         */
        static void matrixMultiplicationF(const cv::Mat &oMatA, const cv::Mat &oMatB, cv::Mat &oMatRes)
        {
            int l_i32BlockSize = 16;

            // Padd matrix offset
            int l_i32PaddOffsetRowsA = (l_i32BlockSize - (oMatA.rows % l_i32BlockSize))% l_i32BlockSize;
            int l_i32PaddOffsetColsA = (l_i32BlockSize - (oMatA.cols % l_i32BlockSize))% l_i32BlockSize;
            int l_i32PaddOffsetRowsB = (l_i32BlockSize - (oMatB.rows % l_i32BlockSize))% l_i32BlockSize;
            int l_i32PaddOffsetColsB = (l_i32BlockSize - (oMatB.cols % l_i32BlockSize))% l_i32BlockSize;

            Matrix A, B, C;
            A.height = oMatA.rows + l_i32PaddOffsetRowsA;
            A.width  = oMatA.cols + l_i32PaddOffsetColsA;
            A.elements = new float[A.width * A.height];

            B.height = oMatB.rows + l_i32PaddOffsetRowsB;
            B.width  = oMatB.cols + l_i32PaddOffsetColsB;
            B.elements = new float[B.width * B.height];;
            C.height = A.height;
            C.width  = B.width;
            C.elements = new float[C.width * C.height];

            for(int ii = 0; ii < A.height; ++ii)
            {
                for(int jj = 0; jj < A.width; ++jj)
                {
                    if(ii < oMatA.rows && jj < oMatA.cols)
                    {
                        A.elements[ii*A.width + jj] = oMatA.at<float>(ii,jj);
                    }
                    else
                    {
                        A.elements[ii*A.width + jj] = 0.f;
                    }
                }
            }

            for(int ii = 0; ii < B.height; ++ii)
            {
                for(int jj = 0; jj < B.width; ++jj)
                {
                    if(ii < oMatB.rows && jj < oMatB.cols)
                    {
                        B.elements[ii*B.width + jj] = oMatB.at<float>(ii,jj);
                    }
                    else
                    {
                        B.elements[ii*B.width + jj] = 0.f;
                    }
                }
            }

            matMult3(A, B, C);

            delete[] A.elements;
            delete[] B.elements;

            int l_i32ResHeight = C.height- l_i32PaddOffsetRowsA;
            int l_i32ResWidth  = C.width - l_i32PaddOffsetColsB;

            oMatRes = cv::Mat(l_i32ResHeight, l_i32ResWidth, CV_32FC1);

            for(int ii = 0; ii < oMatRes.rows; ++ii)
            {
                for(int jj = 0; jj < oMatRes.cols; ++jj)
                {
                    oMatRes.at<float>(ii,jj) = C.elements[ii*C.width + jj];
                }
            }

            delete[] C.elements;
        }

        /**
         * \brief GPU matrix multiplication. Res(l,n) = A(l,m)*B(m,n)
         * \param [in]  oMatA    : input A matrix
         * \param [in]  oMatB    : input B matrix
         * \param [out] oMatRes  : res C matrix
         */
        static void matrixMultiplicationD(const cv::Mat &oMatA, const cv::Mat &oMatB, cv::Mat &oMatRes)
        {
            int l_i32BlockSize = 16;

            // Padd matrix offset
            int l_i32PaddOffsetRowsA = (l_i32BlockSize - (oMatA.rows % l_i32BlockSize))% l_i32BlockSize;
            int l_i32PaddOffsetColsA = (l_i32BlockSize - (oMatA.cols % l_i32BlockSize))% l_i32BlockSize;
            int l_i32PaddOffsetRowsB = (l_i32BlockSize - (oMatB.rows % l_i32BlockSize))% l_i32BlockSize;
            int l_i32PaddOffsetColsB = (l_i32BlockSize - (oMatB.cols % l_i32BlockSize))% l_i32BlockSize;

            MatrixD A, B, C;
            A.height = oMatA.rows + l_i32PaddOffsetRowsA;
            A.width  = oMatA.cols + l_i32PaddOffsetColsA;
            A.elements = new double[A.width * A.height];

            B.height = oMatB.rows + l_i32PaddOffsetRowsB;
            B.width  = oMatB.cols + l_i32PaddOffsetColsB;
            B.elements = new double[B.width * B.height];

            C.height = A.height;
            C.width  = B.width;
            C.elements = new double[C.width * C.height];

            for(int ii = 0; ii < A.height; ++ii)
            {
                for(int jj = 0; jj < A.width; ++jj)
                {
                    if(ii < oMatA.rows && jj < oMatA.cols)
                    {
                        A.elements[ii*A.width + jj] = oMatA.at<double>(ii,jj);
                    }
                    else
                    {
                        A.elements[ii*A.width + jj] = 0.0;
                    }
                }
            }

            for(int ii = 0; ii < B.height; ++ii)
            {
                for(int jj = 0; jj < B.width; ++jj)
                {
                    if(ii < oMatB.rows && jj < oMatB.cols)
                    {
                        B.elements[ii*B.width + jj] = oMatB.at<double>(ii,jj);
                    }
                    else
                    {
                        B.elements[ii*B.width + jj] = 0.0;
                    }
                }
            }

            matMult3D(A, B, C);

            delete[] A.elements;
            delete[] B.elements;

            int l_i32ResHeight = C.height- l_i32PaddOffsetRowsA;
            int l_i32ResWidth  = C.width - l_i32PaddOffsetColsB;

            bool l_bC32 = false, l_bC64 = false;

            oMatRes = cv::Mat(l_i32ResHeight, l_i32ResWidth, CV_64FC1);

            for(int ii = 0; ii < oMatRes.rows; ++ii)
            {
                for(int jj = 0; jj < oMatRes.cols; ++jj)
                {
                    oMatRes.at<double>(ii,jj) = C.elements[ii*C.width + jj];
                }
            }

            delete[] C.elements;
        }


        template<typename T>
        /**
         * @brief block
         * @param oMat
         * @param aFBlock
         * @param ui32X
         * @param ui32Y
         * @param ui32HeightBlock
         * @param ui32WidthBlock
         */
        static void block(const cv::Mat &oMat, T *aFBlock, cuint ui32X, cuint ui32Y, cuint ui32HeightBlock, cuint ui32WidthBlock)
        {
            for(uint ii = 0; ii < ui32HeightBlock; ++ii)
            {
                for(uint jj = 0; jj < ui32WidthBlock; ++jj)
                {
                    int l_i32MatII = ui32X * ui32HeightBlock + ii;
                    int l_i32MatJJ = ui32Y * ui32WidthBlock  + jj;

                    if(l_i32MatII < oMat.rows && l_i32MatJJ < oMat.cols)
                    {
                        aFBlock[ii * ui32WidthBlock + jj] = oMat.at<T>(l_i32MatII, l_i32MatJJ);
                    }
                    else // case where the block contains a padded part of the matrix
                    {
                        aFBlock[ii * ui32WidthBlock + jj] = static_cast<T>(0);
                    }
                }
            }
        }

        template<typename T>
        /**
         * @brief updateMatWithBlock
         * @param oMat
         * @param aFBlock
         * @param ui32X
         * @param ui32Y
         * @param ui32HeightBlock
         * @param ui32WidthBlock
         */
        static void updateMatWithBlock(cv::Mat &oMat, T *aFBlock, cuint ui32X, cuint ui32Y, cuint ui32HeightBlock, cuint ui32WidthBlock)
        {
            for(uint ii = 0; ii < ui32HeightBlock; ++ii)
             {
                 for(uint jj = 0; jj < ui32WidthBlock; ++jj)
                 {
                     int l_i32MatII = ui32X * ui32HeightBlock + ii;
                     int l_i32MatJJ = ui32Y * ui32WidthBlock  + jj;

                     if(l_i32MatII < oMat.rows && l_i32MatJJ < oMat.cols)
                     {
                         if(aFBlock[ii * ui32WidthBlock + jj] != static_cast<T>(0))
                         {
                            oMat.at<T>(l_i32MatII, l_i32MatJJ) = aFBlock[ii * ui32WidthBlock + jj];
                         }
                     }
                 }
             }
        }

        /**
        * @brief blockMatrixMultiplicationD
        * @param oMatA
        * @param oMatB
        * @param oMatRes
        * @param i32SizeMatBlock
        */
        static void blockMatrixMultiplicationD(const cv::Mat &oMatA, const  cv::Mat &oMatB, cv::Mat &oMatRes, cint i32SizeMatBlock = 2)
        {
            int l_i32SizeMatBlock    = i32SizeMatBlock;
            int l_i32SizeMatDivBlock = l_i32SizeMatBlock * BLOCKSIZE;

            // Padd matrix offset
                int l_i32PaddOffsetRowsA = (l_i32SizeMatDivBlock - (oMatA.rows % l_i32SizeMatDivBlock)) % l_i32SizeMatDivBlock;
                int l_i32PaddOffsetColsA = (l_i32SizeMatDivBlock - (oMatA.cols % l_i32SizeMatDivBlock)) % l_i32SizeMatDivBlock;
                int l_i32PaddOffsetRowsB = (l_i32SizeMatDivBlock - (oMatB.rows % l_i32SizeMatDivBlock)) % l_i32SizeMatDivBlock;
                int l_i32PaddOffsetColsB = (l_i32SizeMatDivBlock - (oMatB.cols % l_i32SizeMatDivBlock)) % l_i32SizeMatDivBlock;

            // Init A,B,C sizes
                MatrixD A, B, C;
                A.height = oMatA.rows + l_i32PaddOffsetRowsA;
                A.width  = oMatA.cols + l_i32PaddOffsetColsA;
                B.height = oMatB.rows + l_i32PaddOffsetRowsB;
                B.width  = oMatB.cols + l_i32PaddOffsetColsB;
                C.height = A.height;
                C.width  = B.width;

                int l_i32ResHeight = C.height- l_i32PaddOffsetRowsA;
                int l_i32ResWidth  = C.width - l_i32PaddOffsetColsB;

                oMatRes = cv::Mat(l_i32ResHeight,l_i32ResWidth, CV_64FC1);

                #pragma omp parallel for
                    for(int ii = 0; ii < l_i32SizeMatBlock; ++ii) // C ii ..
                    {
                        // Init subA, suB, subC
                        MatrixD subA, subB, subC;
                        subA.height   = A.height / l_i32SizeMatBlock;
                        subA.width    = A.width  / l_i32SizeMatBlock;
                        subA.elements = new double[subA.height*subA.width];
                        subB.height   = B.height / l_i32SizeMatBlock;
                        subB.width    = B.width  / l_i32SizeMatBlock;
                        subB.elements = new double[subB.height*subB.width];
                        subC.height   = subA.height;
                        subC.width    = subB.width;
                        subC.elements = new double[subC.height*subC.width];
                        double *l_subCCopy = new double[subC.height*subC.width];

                        for(int jj = 0; jj < l_i32SizeMatBlock; ++jj) // C .. jj
                        {
                            for(int kk = 0; kk < subC.height * subC.width; ++kk)
                            {
                                l_subCCopy[kk] = 0.0;
                            }

                            // compute Cij
                            for(int kk = 0; kk < l_i32SizeMatBlock; ++kk)
                            {
                                block<double>(oMatA, subA.elements, ii, kk, subA.height, subA.width);

                                block<double>(oMatB, subB.elements, kk, jj, subB.height, subB.width);

                                #pragma omp critical(cuda)
                                {
                                    matMult3D(subA, subB, subC);
                                }

                                for(int ll = 0; ll < subC.height* subC.width; ++ll)
                                {
                                    l_subCCopy[ll] += subC.elements[ll];
                                }
                            }

                            updateMatWithBlock<double>(oMatRes, l_subCCopy, ii, jj, subC.height, subC.width);
                        }

                        delete[] subA.elements;
                        delete[] subB.elements;
                        delete[] subC.elements;
                        delete[] l_subCCopy;
                    }
                // end ompparallel
        }

        /**
         * @brief blockMatrixMultiplicationF
         * @param oMatA
         * @param oMatB
         * @param oMatRes
         * @param i32SizeMatBlock
         */
        static void blockMatrixMultiplicationF(const cv::Mat &oMatA, const  cv::Mat &oMatB, cv::Mat &oMatRes, cint i32SizeMatBlock = 2)
        {
            int l_i32SizeMatBlock    = i32SizeMatBlock;
            int l_i32SizeMatDivBlock = l_i32SizeMatBlock * BLOCKSIZE;

            // Padd matrix offset
                int l_i32PaddOffsetRowsA = (l_i32SizeMatDivBlock - (oMatA.rows % l_i32SizeMatDivBlock)) % l_i32SizeMatDivBlock;
                int l_i32PaddOffsetColsA = (l_i32SizeMatDivBlock - (oMatA.cols % l_i32SizeMatDivBlock)) % l_i32SizeMatDivBlock;
                int l_i32PaddOffsetRowsB = (l_i32SizeMatDivBlock - (oMatB.rows % l_i32SizeMatDivBlock)) % l_i32SizeMatDivBlock;
                int l_i32PaddOffsetColsB = (l_i32SizeMatDivBlock - (oMatB.cols % l_i32SizeMatDivBlock)) % l_i32SizeMatDivBlock;

            // Init A,B,C sizes
                Matrix A, B, C;
                A.height = oMatA.rows + l_i32PaddOffsetRowsA;
                A.width  = oMatA.cols + l_i32PaddOffsetColsA;
                B.height = oMatB.rows + l_i32PaddOffsetRowsB;
                B.width  = oMatB.cols + l_i32PaddOffsetColsB;
                C.height = A.height;
                C.width  = B.width;

                int l_i32ResHeight = C.height- l_i32PaddOffsetRowsA;
                int l_i32ResWidth  = C.width - l_i32PaddOffsetColsB;


                oMatRes = cv::Mat(l_i32ResHeight,l_i32ResWidth, CV_32FC1);

                #pragma omp parallel for
                    for(int ii = 0; ii < l_i32SizeMatBlock; ++ii) // C ii ..
                    {
                        Matrix subA, subB, subC;
                        subA.height   = A.height / l_i32SizeMatBlock;
                        subA.width    = A.width  / l_i32SizeMatBlock;
                        subA.elements = new float[subA.height*subA.width];
                        subB.height   = B.height / l_i32SizeMatBlock;
                        subB.width    = B.width  / l_i32SizeMatBlock;
                        subB.elements = new float[subB.height*subB.width];
                        subC.height   = subA.height;
                        subC.width    = subB.width;
                        subC.elements = new float[subC.height*subC.width];
                        float *l_subCCopy = new float[subC.height*subC.width];

                        for(int jj = 0; jj < l_i32SizeMatBlock; ++jj) // C .. jj
                        {
                            for(int kk = 0; kk < subC.height * subC.width; ++kk)
                            {
                                l_subCCopy[kk] = 0.f;
                            }

                            // compute Cij
                            for(int kk = 0; kk < l_i32SizeMatBlock; ++kk)
                            {
                                block<float>(oMatA, subA.elements, ii, kk, subA.height, subA.width);

                                block<float>(oMatB, subB.elements, kk, jj, subB.height, subB.width);

                                #pragma omp critical(cuda)
                                {
                                    matMult3(subA, subB, subC);
                                }

                                for(int ll = 0; ll < subC.height* subC.width; ++ll)
                                {
                                    l_subCCopy[ll] += subC.elements[ll];
                                }
                            }

                            updateMatWithBlock<float>(oMatRes, l_subCCopy, ii, jj, subC.height, subC.width);
                        }

                        delete[] subA.elements;
                        delete[] subB.elements;
                        delete[] subC.elements;
                        delete[] l_subCCopy;
                    }
                // end omp parallel
        }

        template<typename T>
        /**
         * @brief blockMatrixMultiplication
         * @param oMatA
         * @param oMatB
         * @param oMatRes
         * @param i32SizeMatBlock
         */
        static void blockMatrixMultiplication(const cv::Mat &oMatA, const  cv::Mat &oMatB, cv::Mat &oMatRes, cint i32SizeMatBlock = 2)
        {
            if(typeid(float) == typeid(T))
            {
                blockMatrixMultiplicationF(oMatA,oMatB,oMatRes,i32SizeMatBlock);
            }
            else if(typeid(double) == typeid(T))
            {
                blockMatrixMultiplicationD(oMatA,oMatB,oMatRes,i32SizeMatBlock);
            }
            else
            {
                std::cerr << "-ERROR : blockMatrixMultiplication -> type not managed. " << std::endl;
            }
        }
}


#endif // CUDAMULTIPLICATIONS_H
