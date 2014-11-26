
/**
 * \file Reservoir.h
 * \brief defines Reservoir
 * \author Florian Lance
 * \date 01/10/14
 */

/*! \mainpage
 *
 * \section Reservoir-cuda
 *
 * \subsection intro_sec Introduction
 *
 * A cuda/openMP reservoir computing software for sentences learning.
 *
 */

#ifndef _RESERVOIR_
#define _RESERVOIR_

#include <iostream>
#include <time.h>

// Opencv
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "Utility.h"
#include "gpuMat/cudaInversions.h"
#include "gpuMat/cudaMultiplications.h"

/**
 * @brief The Reservoir class
 */
class Reservoir
{
    public :

        /**
         * @brief Reservoir
         */
        Reservoir();

        /**
         * @brief Reservoir
         * @param nbNeurons
         * @param spectralRadius
         * @param inputScaling
         * @param leakRate
         * @param sparcity
         * @param ridge
         * @param verbose
         */
        Reservoir(cuint nbNeurons, cdouble spectralRadius, cdouble inputScaling, cdouble leakRate, cdouble sparcity = -1.0, cdouble ridge = 1e-5, cbool verbose = true);
        /**
         * @brief Reservoir
         * @param nbNeurons
         * @param spectralRadius
         * @param inputScaling
         * @param leakRate
         * @param sparcity
         * @param ridge
         * @param verbose
         */
        Reservoir(cuint nbNeurons, cfloat spectralRadius, cfloat inputScaling, cfloat leakRate, cfloat sparcity = -1.f, cfloat ridge = 1e-5f, cbool verbose = true);

        /**
         * @brief setCudaProperties
         * @param cudaInv
         * @param cudaMult
         */
        void setCudaProperties(cbool cudaInv, cbool cudaMult);

        /**
         * @brief generateMatrixW : generate the main matrix of the reservoir
         */
        void generateMatrixW();
        /**
         * @brief generateMatrixWF
         */
        void generateMatrixWF();

        /**
         * @brief generateWIn : generate the input matrix of the reservoir
         * @param dimInput
         */
        void generateWIn(cuint dimInput);
        /**
         * @brief generateWInF
         * @param dimInput
         */
        void generateWInF(cuint dimInput);

        /**
         * @brief tikhonovRegularization : computation of the weight of w_out
         * @param xTot
         * @param yTeacher
         * @param dimInput
         */
        void tikhonovRegularization(const cv::Mat &xTot, const cv::Mat &yTeacher, cuint dimInput);
        /**
         * @brief tikhonovRegularizationF
         * @param xTot
         * @param yTeacher
         * @param dimInput
         */
        void tikhonovRegularizationF(const cv::Mat &xTot, const cv::Mat &yTeacher, cuint dimInput);

        /**
         * @brief train : training of the reseroir
         * @param meaningInputTrain
         * @param teacher
         * @param sentencesOutputTrain
         * @param xTot
         */
        void train(const cv::Mat &meaningInputTrain, const cv::Mat &teacher, cv::Mat &sentencesOutputTrain, cv::Mat &xTot);
        /**
         * @brief trainF
         * @param meaningInputTrain
         * @param teacher
         * @param sentencesOutputTrain
         * @param xTot
         */
        void trainF(const cv::Mat &meaningInputTrain, const cv::Mat &teacher, cv::Mat &sentencesOutputTrain, cv::Mat &xTot);

        /**
         * @brief testing of the reservoir, this function works exactly like the training function except w_out is already known
         * so there is no Thikhonov regularization
         * @param meaningInputTest
         * @param sentencesOutputTest
         * @param xTot
         */
        void test(const cv::Mat &meaningInputTest, cv::Mat &sentencesOutputTest, cv::Mat &xTot);
        /**
         * @brief testF
         * @param meaningInputTest
         * @param sentencesOutputTest
         * @param xTot
         */
        void testF(const cv::Mat &meaningInputTest, cv::Mat &sentencesOutputTest, cv::Mat &xTot);

    private :

        bool m_useCudaInversion;        /**< ... */
        bool m_useCudaMultiplication;   /**< ... */

        bool m_initialized;             /**< ... */
        bool m_verbose;                 /**< ... */
        int m_nbNeurons;                /**< ... */

        double m_sparcity;              /**< ... */
        double m_spectralRadius;        /**< ... */
        double m_inputScaling;          /**< ... */
        double m_leakRate;              /**< ... */
        double m_ridge;                 /**< ... */

        cv::Mat m_w;                    /**< ... */
        cv::Mat m_wIn;                  /**< ... */
        cv::Mat m_wOut;                 /**< ... */

        clock_t m_oTime;                /**< ... */

        // FLOAT TEST
        float m_sparcityF;              /**< ... */
        float m_spectralRadiusF;        /**< ... */
        float m_inputScalingF;          /**< ... */
        float m_leakRateF;              /**< ... */
        float m_ridgeF;                 /**< ... */

        cv::Mat m_wF;                   /**< ... */
        cv::Mat m_wInF;                 /**< ... */
        cv::Mat m_wOutF;                /**< ... */

};


template<class T>
/**
 * @brief The Reservoir2 class
 */
class Reservoir2
{
    public :


        /**
         * @brief Reservoir
         */
        Reservoir2();

        /**
         * @brief Reservoir
         * @param nbNeurons
         * @param spectralRadius
         * @param inputScaling
         * @param leakRate
         * @param sparcity
         * @param ridge
         * @param verbose
         */
        Reservoir2(cuint nbNeurons, const T spectralRadius, const T inputScaling, const T leakRate,
                   const T sparcity = static_cast<T>(-1), const T ridge = static_cast<T>(1e-5), cbool verbose = true);

        /**
         * @brief setCudaProperties
         * @param cudaInv
         * @param cudaMult
         */
        void setCudaProperties(cbool cudaInv, cbool cudaMult);

        /**
         * @brief generateMatrixW : generate the main matrix of the reservoir
         */
        void generateMatrixW();

        /**
         * @brief generateWIn : generate the input matrix of the reservoir
         * @param dimInput
         */
        void generateWIn(cuint dimInput);

        /**
         * @brief tikhonovRegularization : computation of the weight of w_out
         * @param xTot
         * @param yTeacher
         * @param dimInput
         */
        void tikhonovRegularization(const cv::Mat &xTot, const cv::Mat &yTeacher, cuint dimInput);

        /**
         * @brief train : training of the reseroir
         * @param meaningInputTrain
         * @param teacher
         * @param sentencesOutputTrain
         * @param xTot
         */
        void train(const cv::Mat &meaningInputTrain, const cv::Mat &teacher, cv::Mat &sentencesOutputTrain, cv::Mat &xTot);

        /**
         * @brief testing of the reservoir, this function works exactly like the training function except w_out is already known
         * so there is no Thikhonov regularization
         * @param meaningInputTest
         * @param sentencesOutputTest
         * @param xTot
         */
        void test(const cv::Mat &meaningInputTest, cv::Mat &sentencesOutputTest, cv::Mat &xTot);

    private :

        bool m_useCudaInversion;        /**< ... */
        bool m_useCudaMultiplication;   /**< ... */

        bool m_initialized;             /**< ... */
        bool m_verbose;                 /**< ... */
        int m_nbNeurons;                /**< ... */

        T m_sparcity;                   /**< ... */
        T m_spectralRadius;             /**< ... */
        T m_inputScaling;               /**< ... */
        T m_leakRate;                   /**< ... */
        T m_ridge;                      /**< ... */

        cv::Mat m_w;                    /**< ... */
        cv::Mat m_wIn;                  /**< ... */
        cv::Mat m_wOut;                 /**< ... */

        clock_t m_oTime;                /**< ... */
};

template<class T>
void Reservoir2<T>::generateMatrixW()
{
    // debug
    displayTime("START : generate W ", m_oTime, false, m_verbose);

    // init w matrix [N x N]
    initMatrix<T>(m_w, m_nbNeurons, m_nbNeurons, true);

    for(int ii = 0; ii < m_w.rows*m_w.cols;++ii)
    {
        if(static_cast <T> (rand()) / static_cast <T> (RAND_MAX) < m_sparcity)
        {
            double r = static_cast <T> (rand()) / static_cast <T> (RAND_MAX);
            m_w.at<T>(ii) = (r - static_cast<T>(0.5)) * m_spectralRadius;
        }
    }
        // debug
    displayTime("END : generate W ", m_oTime, false, m_verbose);
}

template<class T>
void Reservoir2<T>::generateWIn(cuint dimInput)
{
    // debug
    displayTime("START : generate WIn ", m_oTime, false, m_verbose);

    // init wIn
    initMatrix<T>(m_w, m_nbNeurons, dimInput + 1, false);

    // fill wIn matrix with random values [0, 1]
        cv::MatIterator_<T> it = m_wIn.begin<T>(), it_end = m_wIn.end<T>();
        T l_randMax = static_cast <T> (RAND_MAX);
        for(;it != it_end; ++it)
        {
            (*it) = (static_cast <T> (rand()) / l_randMax) * m_inputScaling;
        }

    // debug
    displayTime("END : generate WIn ", m_oTime, false, m_verbose);
}

template<class T>
void Reservoir2<T>::tikhonovRegularization(const cv::Mat &xTot, const cv::Mat &yTeacher, cuint dimInput)
{
    int l_subdivisionBlocks = 2;
    if(m_nbNeurons > 3000)
    {
        l_subdivisionBlocks = 4;
    }
    if(m_nbNeurons > 6000)
    {
        l_subdivisionBlocks = 6;
    }
    if(m_nbNeurons > 8000)
    {
        l_subdivisionBlocks = 8;
    }

    displayTime("START : tikhonovRegularization ", m_oTime, false, m_verbose);

    cv::Mat l_xTotReshaped;
    initMatrix<T>(l_xTotReshaped, xTot.size[1], xTot.size[0] * xTot.size[2], false);

    #pragma omp parallel for
        for(int ii = 0; ii < xTot.size[0]; ++ii)
        {
            for(int jj = 0; jj < xTot.size[1]; ++jj)
            {
                for(int kk = 0; kk < xTot.size[2]; ++kk)
                {
                    l_xTotReshaped.at<T>(jj, ii*xTot.size[2] + kk) = xTot.at<T>(ii,jj,kk);
                }
            }
        }
    // end pragma

    displayTime("1 : tikhonovRegularization ", m_oTime, false, m_verbose);
    cv::Mat l_mat2inv;

    if(m_useCudaInversion)
    {
        swCuda::blockMatrixMultiplication<T>(l_xTotReshaped,l_xTotReshaped.t(), l_mat2inv, l_subdivisionBlocks);
    }
    else
    {
        l_mat2inv = (l_xTotReshaped * l_xTotReshaped.t());
    }

    if(typeid(T) == typeid(float))
    {
        l_mat2inv += (cv::Mat::eye(1 + dimInput + m_nbNeurons,1 + dimInput + m_nbNeurons,CV_32FC1) * m_ridge);
    }
    else
    {
        l_mat2inv += (cv::Mat::eye(1 + dimInput + m_nbNeurons,1 + dimInput + m_nbNeurons,CV_64FC1) * m_ridge);
    }

    cv::Mat invCuda, invCV;
    cv::Mat matCudaS,matCudaU,matCudaVT;

    if(m_useCudaInversion)
    {
        swCuda::squareMatrixSingularValueDecomposition(l_mat2inv,matCudaS,matCudaU,matCudaVT);
        l_mat2inv.release();

        displayTime("2 : tikhonovRegularization ", m_oTime, false, m_verbose);

        for(int ii = 0; ii < matCudaS.rows;++ii)
        {
            if(matCudaS.at<T>(ii,ii) > static_cast<T>(1e-6))
            {
                matCudaS.at<T>(ii,ii) = static_cast<T>(1)/matCudaS.at<T>(ii,ii);
            }
            else
            {
                matCudaS.at<T>(ii,ii) = static_cast<T>(0);
            }
        }

        if(m_useCudaMultiplication)
        {
            cv::Mat l_tempCudaMult;
            swCuda::blockMatrixMultiplication<T>(matCudaS, matCudaU.t(), l_tempCudaMult, l_subdivisionBlocks);
            matCudaS.release();
            matCudaU.release();

            swCuda::blockMatrixMultiplication<T>(matCudaVT.t(), l_tempCudaMult, invCuda, l_subdivisionBlocks);
            matCudaVT.release();
        }
        else
        {
            invCuda = (matCudaVT.t() * matCudaS * matCudaU.t());
        }

        displayTime("3 : tikhonovRegularization ", m_oTime, false, m_verbose);

        if(m_useCudaMultiplication)
        {
            cv::Mat l_tempCudaMult;
            l_xTotReshaped = l_xTotReshaped.t();

            swCuda::blockMatrixMultiplication<T>(l_xTotReshaped, invCuda, l_tempCudaMult, l_subdivisionBlocks);
            invCuda.release();
            l_xTotReshaped.release();

            cv::Mat l_yTeacherReshaped;
            initMatrix<T>(l_yTeacherReshaped, yTeacher.size[0] *yTeacher.size[1], yTeacher.size[2], false);

            #pragma omp parallel for
                for(int ii = 0; ii < yTeacher.size[0]; ++ii)
                {
                    for(int jj = 0; jj < yTeacher.size[1]; ++jj)
                    {
                        for(int kk = 0; kk < yTeacher.size[2]; ++kk)
                        {
                            l_yTeacherReshaped.at<T>(ii*yTeacher.size[1] + jj,kk) = yTeacher.at<T>(ii,jj,kk);
                        }
                    }
                }
            // end pragma

            m_wOut = l_yTeacherReshaped.t() * l_tempCudaMult;
        }
        else
        {
            cv::Mat l_yTeacherReshaped;
            initMatrix<T>(l_yTeacherReshaped, yTeacher.size[0] *yTeacher.size[1], yTeacher.size[2], false);

            #pragma omp parallel for
                for(int ii = 0; ii < yTeacher.size[0]; ++ii)
                {
                    for(int jj = 0; jj < yTeacher.size[1]; ++jj)
                    {
                        for(int kk = 0; kk < yTeacher.size[2]; ++kk)
                        {
                            l_yTeacherReshaped.at<T>(ii*yTeacher.size[1] + jj,kk) = yTeacher.at<T>(ii,jj,kk);
                        }
                    }
                }
            // end pragma

            m_wOut = l_yTeacherReshaped.t() * l_xTotReshaped.t() * invCuda;
        }
    }
    else
    {
        cv::invert(l_mat2inv, invCV, cv::DECOMP_SVD);
        l_mat2inv.release();

        displayTime("2-3 : tikhonovRegularization ", m_oTime, false, m_verbose);

        cv::Mat l_yTeacherReshaped;
        initMatrix<T>(l_yTeacherReshaped, yTeacher.size[0] *yTeacher.size[1], yTeacher.size[2], false);

        #pragma omp parallel for
            for(int ii = 0; ii < yTeacher.size[0]; ++ii)
            {
                for(int jj = 0; jj < yTeacher.size[1]; ++jj)
                {
                    for(int kk = 0; kk < yTeacher.size[2]; ++kk)
                    {
                        l_yTeacherReshaped.at<T>(ii*yTeacher.size[1] + jj,kk) = yTeacher.at<T>(ii,jj,kk);
                    }
                }
            }
        // end pragma

        m_wOut = (l_yTeacherReshaped.t() * l_xTotReshaped.t()) * invCV;
    }


    displayTime("END : tikhonovRegularization ", m_oTime, false, m_verbose);
}


template<class T>
void Reservoir2<T>::train(const cv::Mat &meaningInputTrain, const cv::Mat &teacher, cv::Mat &sentencesOutputTrain, cv::Mat &xTot)
{
    m_oTime = clock();

    displayTime("START : train ", m_oTime, false, m_verbose);

    // generate matrices
        generateMatrixW();
        generateWIn(meaningInputTrain.size[2]);

    displayTime("START : sub train ", m_oTime, false, m_verbose);

        int l_sizeTot[3] = {meaningInputTrain.size[0], 1 + meaningInputTrain.size[2] + m_nbNeurons,  meaningInputTrain.size[1]};
        xTot = cv::Mat (3,l_sizeTot, CV_64FC1, cv::Scalar(0.0)); //  will contain the internal states of the reservoir for all sentences and all timesteps

        cv::Mat l_X2Copy = cv::Mat::zeros(1 + meaningInputTrain.size[2] + m_nbNeurons, meaningInputTrain.size[1], CV_64FC1); // OPTI

        int l_size[1] = {m_w.rows}; // OPTI
        cv::Mat l_xPrev2Copy(1,l_size, CV_64FC1, cv::Scalar(0.0)); // OPTI

        double l_invLeakRate = 1.0 - m_leakRate;

//        fillRandomMat_(xTot);

//        #pragma omp parallel for num_threads(7)
        #pragma omp parallel for
            for(int ii = 0; ii < meaningInputTrain.size[0]; ++ii)
            {
                if(m_verbose)
                {
                    printf("input train : %d / %d\n", ii,meaningInputTrain.size[0]);
                }

                cv::Mat l_X = l_X2Copy.clone();
                cv::Mat l_xPrev, l_x;
                cv::Mat l_subMean(meaningInputTrain.size[1], meaningInputTrain.size[2], CV_64FC1, meaningInputTrain.data + meaningInputTrain.step[0] *ii);

                for(int jj = 0; jj < meaningInputTrain.size[1]; ++jj)
                {
                    // X will contain all the internal states of the reservoir for all timesteps
                    if(jj == 0)
                    {
                        l_xPrev = l_xPrev2Copy;
                    }
                    else
                    {
                        l_xPrev = l_x;
                    }

                    cv::Mat l_u = l_subMean.row(jj);
                    cv::Mat l_temp(l_subMean.cols+1, 1, CV_64FC1);
                    l_temp.at<double>(0) = 1.0;

                    for(int kk = 0; kk < l_subMean.cols; ++kk)
                    {
                        l_temp.at<double>(kk+1) = l_u.at<double>(kk);
                    }

                    cv::Mat l_xTemp = (m_wIn * l_temp)+ (m_w * l_xPrev);

                    cv::MatIterator_<double> it = l_xTemp.begin<double>(), it_end = l_xTemp.end<double>();
                    for(;it != it_end; ++it)
                    {
                        (*it) = tanh(*it);
                    }

                    l_x = (l_xPrev * l_invLeakRate) + (l_xTemp * m_leakRate);

                    cv::Mat l_temp2(l_temp.rows + l_x.rows, 1, CV_64FC1);

                    for(int kk = 0; kk < l_temp.rows + l_x.rows; ++kk)
                    {
                        if(kk < l_temp.rows)
                        {
                            l_temp2.at<double>(kk) = l_temp.at<double>(kk);
                        }
                        else
                        {
                            l_temp2.at<double>(kk) = l_x.at<double>(kk - l_temp.rows);
                        }
                    }

                    l_temp2.copyTo( l_X.col(jj));
                }

                for(int jj = 0; jj < l_X.rows; ++jj)
                {
                    for(int kk = 0; kk < l_X.cols; ++kk)
                    {
                        xTot.at<double>(ii,jj,kk) = l_X.at<double>(jj,kk);
                    }
                }
            }
        // end pragma

        l_X2Copy.release();
        l_xPrev2Copy.release();

        displayTime("END : sub train ", m_oTime, false, m_verbose);

        tikhonovRegularization(xTot, teacher, meaningInputTrain.size[2]);

        sentencesOutputTrain = teacher.clone();
        sentencesOutputTrain.setTo(0.0);

        #pragma omp parallel for
            for(int ii = 0; ii < xTot.size[0]; ++ii)
            {
                cv::Mat l_X = cv::Mat::zeros(xTot.size[1], xTot.size[2], CV_64FC1);

                for(int jj = 0; jj < l_X.rows; ++jj)
                {
                    for(int kk = 0; kk < l_X.cols; ++kk)
                    {
                        l_X.at<double>(jj,kk) = xTot.at<double>(ii,jj,kk);
                    }
                }

                cv::Mat res;

                res = (m_wOut * l_X).t();

                for(int jj = 0; jj < sentencesOutputTrain.size[1]; ++jj)
                {
                    for(int kk = 0; kk < sentencesOutputTrain.size[2]; ++kk)
                    {
                        sentencesOutputTrain.at<double>(ii,jj,kk) = res.at<double>(jj,kk);
                    }
                }
            }
        // end pragma

    displayTime("END : train ", m_oTime, false, m_verbose);

}

#endif
