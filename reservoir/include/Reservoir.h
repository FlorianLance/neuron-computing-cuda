

/**
 * \file Reservoir.h
 * \brief defines Reservoir
 * \author Florian Lance
 * \date 02/12/14
 */

#ifndef RESERVOIR_H
#define RESERVOIR_H

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
class Reservoir  : public QObject
{

    Q_OBJECT

    public :

        /**
         * @brief ReservoirQt
         */
        Reservoir();

        /**
         * @brief ReservoirQt
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
         * @brief setParameters
         * @param nbNeurons
         * @param spectralRadius
         * @param inputScaling
         * @param leakRate
         * @param sparcity
         * @param ridge
         * @param verbose
         */
        void setParameters(cuint nbNeurons, cfloat spectralRadius, cfloat inputScaling, cfloat leakRate, cfloat sparcity, cfloat ridge, cbool verbose);

        /**
         * @brief setCudaProperties
         * @param cudaInv
         * @param cudaMult
         */
        void setCudaProperties(cbool cudaInv, cbool cudaMult);

        /**
         * @brief generateMatrixW
         */
        void generateMatrixW();

        /**
         * @brief generateWIn
         * @param dimInput
         */
        void generateWIn(cuint dimInput);

        /**
         * @brief tikhonovRegularization
         * @param xTot
         * @param yTeacher
         * @param dimInput
         */
        void tikhonovRegularization(const cv::Mat &xTot, const cv::Mat &yTeacher, cuint dimInput);

        /**
         * @brief train
         * @param meaningInputTrain
         * @param teacher
         * @param sentencesOutputTrain
         * @param xTot
         */
        void train(const cv::Mat &meaningInputTrain, const cv::Mat &teacher, cv::Mat &sentencesOutputTrain, cv::Mat &xTot);

        /**
         * @brief test
         * @param meaningInputTest
         * @param sentencesOutputTest
         * @param xTot
         */
        void test(const cv::Mat &meaningInputTest, cv::Mat &sentencesOutputTest, cv::Mat &xTot);

        /**
         * @brief Save the current state of internal matrices m_wF m_wInF, m_wOutF
         * @param [in] path : path of the directory where the files m_w.txt, m_wIn.txt, m_wOut.txt will be saved
         */
        void saveTraining(const std::string &path);

        /**
         * @brief loadTraining
         * @param [in]  path       : path of directory containg the files m_w.txt, m_wIn.txt, m_wOut.txt
         */
        void loadTraining(const std::string &path);

        /**
         * @brief loadW
         * @param path
         */
        void loadW(const std::string &path);

        /**
         * @brief loadWIn
         * @param path
         */
        void loadWIn(const std::string &path);

        /**
         * @brief updateMatricesWithLoadedTraining
         */
        void updateMatricesWithLoadedTraining();

        /**
         * @brief setMatricesUse
         * @param useCustomW
         * @param useCustomWIn
         */
        void setMatricesUse(cbool useCustomW, cbool useCustomWIn);

    public slots :

        /**
         * @brief enableMaxOmpThreadNumber
         * @param enable
         */
        void enableMaxOmpThreadNumber(bool enable);

        /**
         * @brief enableDisplay
         * @param enable
         */
        void enableDisplay(bool enable);

        /**
         * @brief updateMatrixXDisplayParameters
         * @param enabled
         * @param randomSentence
         * @param nbRandomNeurons
         * @param startIdNeurons
         * @param endIdNeurons
         */
        void updateMatrixXDisplayParameters(bool enabled, bool randomNeurons, int nbRandomNeurons, int startIdNeurons, int endIdNeurons);

    signals :

        /**
         * @brief sendComputingState
         */
        void sendComputingState(int, int, QString);

        /**
         * @brief sendMatriceImage2Display
         */
        void sendMatriceImage2Display(QImage);

        /**
         * @brief sendXMatriceData
         * @param data
         * @param currentSentenceId
         * @param nbSentence
         */
        void sendXMatriceData(QVector<QVector<double> >*data, int currentSentenceId,int nbSentence);

        /**
         * @brief sendInfoPlot
         */
        void sendInfoPlot(int, int, int, QString);

        /**
         * @brief sendLogInfo
         */
        void sendLogInfo(QString, QColor);

        /**
         * @brief sendLoadedParameters
         */
        void sendLoadedTrainingParameters(QStringList);

        /**
         * @brief sendLoadedWParameters
         */
        void sendLoadedWParameters(QStringList);

        /**
         * @brief sendLoadedWInParameters
         */
        void sendLoadedWInParameters(QStringList);


    private :

        bool m_useW;
        bool m_useWIn;

        bool m_useCudaInversion;        /**< uses cuda inversion matrice ? else uses opencv */
        bool m_useCudaMultiplication;   /**< uses cuda multiplication matrices ? else uses opencv */

        bool m_initialized;             /**< is the reservoir initialized ? */
        bool m_verbose;                 /**< verbose comments */
        int m_nbNeurons;                /**< number of neurons used for building the reservoir */

        clock_t m_oTime;                /**< clock for mesuring the computing time */

        float m_sparcity;               /**< sparcity of the matrice W */
        float m_spectralRadius;         /**< spectral radius used to multiply the matrice W */
        float m_inputScaling;           /**< input scaling used to multiply the matrice W IN */
        float m_leakRate;               /**< leak rate used to build X tot in the training and the test */
        float m_ridge;                  /**< ridge value used in the tychonov regularization */

        cv::Mat m_w;                    /**< W matrice */
        cv::Mat m_wIn;                  /**< W IN matrice */
        cv::Mat m_wOut;                 /**< W OUT matrice */

        cv::Mat m_wLoaded;              /**< loaded W matrice */
        cv::Mat m_wInLoaded;            /**< loaded W IN matrice  */
        cv::Mat m_wOutLoaded;           /**< loaded W OUT matrice */

        int m_numThread;                /**< number of threads to be used by openmp */
        bool m_sendMatrices;            /**< send matrices to be displayed in the interface */


        // TEMP
        bool m_displayEnabled;
        bool m_randomNeurons;
        int m_nbRandomNeurons;
        int m_startIdNeurons;
        int m_endIdNeurons;
};


//template<class T>
///**
// * @brief The Reservoir2 class
// */
//class Reservoir2Qt
//{
//    public :


//        /**
//         * @brief Reservoir
//         */
//        Reservoir2Qt();

//        /**
//         * @brief Reservoir
//         * @param nbNeurons
//         * @param spectralRadius
//         * @param inputScaling
//         * @param leakRate
//         * @param sparcity
//         * @param ridge
//         * @param verbose
//         */
//        Reservoir2Qt(cuint nbNeurons, const T spectralRadius, const T inputScaling, const T leakRate,
//                   const T sparcity = static_cast<T>(-1), const T ridge = static_cast<T>(1e-5), cbool verbose = true);

//        /**
//         * @brief setCudaProperties
//         * @param cudaInv
//         * @param cudaMult
//         */
//        void setCudaProperties(cbool cudaInv, cbool cudaMult);

//        /**
//         * @brief generateMatrixW : generate the main matrix of the reservoir
//         */
//        void generateMatrixW();

//        /**
//         * @brief generateWIn : generate the input matrix of the reservoir
//         * @param dimInput
//         */
//        void generateWIn(cuint dimInput);

//        /**
//         * @brief tikhonovRegularization : computation of the weight of w_out
//         * @param xTot
//         * @param yTeacher
//         * @param dimInput
//         */
//        void tikhonovRegularization(const cv::Mat &xTot, const cv::Mat &yTeacher, cuint dimInput);

//        /**
//         * @brief train : training of the reseroir
//         * @param meaningInputTrain
//         * @param teacher
//         * @param sentencesOutputTrain
//         * @param xTot
//         */
//        void train(const cv::Mat &meaningInputTrain, const cv::Mat &teacher, cv::Mat &sentencesOutputTrain, cv::Mat &xTot);

//        /**
//         * @brief testing of the reservoir, this function works exactly like the training function except w_out is already known
//         * so there is no Thikhonov regularization
//         * @param meaningInputTest
//         * @param sentencesOutputTest
//         * @param xTot
//         */
//        void test(const cv::Mat &meaningInputTest, cv::Mat &sentencesOutputTest, cv::Mat &xTot);

//    private :

//        bool m_useCudaInversion;        /**< ... */
//        bool m_useCudaMultiplication;   /**< ... */

//        bool m_initialized;             /**< ... */
//        bool m_verbose;                 /**< ... */
//        int m_nbNeurons;                /**< ... */

//        T m_sparcity;                   /**< ... */
//        T m_spectralRadius;             /**< ... */
//        T m_inputScaling;               /**< ... */
//        T m_leakRate;                   /**< ... */
//        T m_ridge;                      /**< ... */

//        cv::Mat m_w;                    /**< ... */
//        cv::Mat m_wIn;                  /**< ... */
//        cv::Mat m_wOut;                 /**< ... */

//        clock_t m_oTime;                /**< ... */
//};


//template<class T>
//Reservoir2Qt<T>::Reservoir2Qt()
//{
//    m_initialized = false;
//    m_verbose = true;

//    m_useCudaInversion      = true;
//    m_useCudaMultiplication = false;
//}

//template<class T>
//Reservoir2Qt<T>::Reservoir2Qt(cuint nbNeurons, const T spectralRadius, const T inputScaling, const T leakRate, const T sparcity, const T ridge, cbool verbose)
//    : m_nbNeurons(nbNeurons), m_spectralRadius(spectralRadius), m_inputScaling(inputScaling), m_leakRate(leakRate), m_ridge(ridge), m_verbose(verbose)
//{
//    if(sparcity > 0)
//    {
//        m_sparcity = sparcity;
//    }
//    else
//    {
//        m_sparcity = static_cast<T>(10)/m_nbNeurons;
//    }
//    m_initialized = true;
//}

//template<class T>
//void Reservoir2Qt<T>::setCudaProperties(cbool cudaInv, cbool cudaMult)
//{
//    m_useCudaInversion = cudaInv;
//    m_useCudaMultiplication = cudaMult;
//}


//template<class T>
//void Reservoir2Qt<T>::generateMatrixW()
//{
//    // debug
//    displayTime("START : generate W ", m_oTime, false, m_verbose);

//    // init w matrix [N x N]
//    initMatrix<T>(m_w, m_nbNeurons, m_nbNeurons, true);

//    for(int ii = 0; ii < m_w.rows*m_w.cols;++ii)
//    {
//        if(static_cast <T> (rand()) / static_cast <T> (RAND_MAX) < m_sparcity)
//        {
//            T r = static_cast <T> (rand()) / static_cast <T> (RAND_MAX);
//            m_w.at<T>(ii) = (r - static_cast<T>(0.5)) * m_spectralRadius;
//        }
//    }
//        // debug
//    displayTime("END : generate W ", m_oTime, false, m_verbose);
//}

//template<class T>
//void Reservoir2Qt<T>::generateWIn(cuint dimInput)
//{
//    // debug
//    displayTime("START : generate WIn ", m_oTime, false, m_verbose);

//    // init wIn
//    initMatrix<T>(m_wIn, m_nbNeurons, dimInput + 1, false);

//    // fill wIn matrix with random values [0, 1]
//        cv::MatIterator_<T> it = m_wIn.begin<T>(), it_end = m_wIn.end<T>();
//        T l_randMax = static_cast <T> (RAND_MAX);
//        for(;it != it_end; ++it)
//        {
//            (*it) = (static_cast <T> (rand()) / l_randMax) * m_inputScaling;
//        }

//    // debug
//    displayTime("END : generate WIn ", m_oTime, false, m_verbose);
//}

//template<class T>
//void Reservoir2Qt<T>::tikhonovRegularization(const cv::Mat &xTot, const cv::Mat &yTeacher, cuint dimInput)
//{
//    int l_subdivisionBlocks = 2;
//    if(m_nbNeurons > 3000)
//    {
//        l_subdivisionBlocks = 4;
//    }
//    if(m_nbNeurons > 6000)
//    {
//        l_subdivisionBlocks = 6;
//    }
//    if(m_nbNeurons > 8000)
//    {
//        l_subdivisionBlocks = 8;
//    }

//    displayTime("START : tikhonovRegularization ", m_oTime, false, m_verbose);

//    cv::Mat l_xTotReshaped;
//    initMatrix<T>(l_xTotReshaped, xTot.size[1], xTot.size[0] * xTot.size[2], false);

//    #pragma omp parallel for
//        for(int ii = 0; ii < xTot.size[0]; ++ii)
//        {
//            for(int jj = 0; jj < xTot.size[1]; ++jj)
//            {
//                for(int kk = 0; kk < xTot.size[2]; ++kk)
//                {
//                    l_xTotReshaped.at<T>(jj, ii*xTot.size[2] + kk) = xTot.at<T>(ii,jj,kk);
//                }
//            }
//        }
//    // end pragma

//    displayTime("1 : tikhonovRegularization ", m_oTime, false, m_verbose);
//    cv::Mat l_mat2inv;

//    if(m_useCudaInversion)
//    {
//        swCuda::blockMatrixMultiplication<T>(l_xTotReshaped,l_xTotReshaped.t(), l_mat2inv, l_subdivisionBlocks);
//    }
//    else
//    {
//        l_mat2inv = (l_xTotReshaped * l_xTotReshaped.t());
//    }

//    if(typeid(T) == typeid(float))
//    {
//        l_mat2inv += (cv::Mat::eye(1 + dimInput + m_nbNeurons,1 + dimInput + m_nbNeurons,CV_32FC1) * m_ridge);
//    }
//    else
//    {
//        l_mat2inv += (cv::Mat::eye(1 + dimInput + m_nbNeurons,1 + dimInput + m_nbNeurons,CV_64FC1) * m_ridge);
//    }

//    cv::Mat invCuda, invCV;
//    cv::Mat matCudaS,matCudaU,matCudaVT;

//    if(m_useCudaInversion)
//    {
//        swCuda::squareMatrixSingularValueDecomposition(l_mat2inv,matCudaS,matCudaU,matCudaVT);
//        l_mat2inv.release();

//        displayTime("2 : tikhonovRegularization ", m_oTime, false, m_verbose);

//        for(int ii = 0; ii < matCudaS.rows;++ii)
//        {
//            if(matCudaS.at<T>(ii,ii) > static_cast<T>(1e-6))
//            {
//                matCudaS.at<T>(ii,ii) = 1/matCudaS.at<T>(ii,ii);
//            }
//            else
//            {
//                matCudaS.at<T>(ii,ii) = 0;
//            }
//        }

//        if(m_useCudaMultiplication)
//        {
//            cv::Mat l_tempCudaMult;
//            swCuda::blockMatrixMultiplication<T>(matCudaS, matCudaU.t(), l_tempCudaMult, l_subdivisionBlocks);
//            matCudaS.release();
//            matCudaU.release();

//            swCuda::blockMatrixMultiplication<T>(matCudaVT.t(), l_tempCudaMult, invCuda, l_subdivisionBlocks);
//            matCudaVT.release();
//        }
//        else
//        {
//            invCuda = (matCudaVT.t() * matCudaS * matCudaU.t());
//        }

//        displayTime("3 : tikhonovRegularization ", m_oTime, false, m_verbose);

//        if(m_useCudaMultiplication)
//        {
//            cv::Mat l_tempCudaMult;
//            l_xTotReshaped = l_xTotReshaped.t();

//            swCuda::blockMatrixMultiplication<T>(l_xTotReshaped, invCuda, l_tempCudaMult, l_subdivisionBlocks);
//            invCuda.release();
//            l_xTotReshaped.release();

//            cv::Mat l_yTeacherReshaped;
//            initMatrix<T>(l_yTeacherReshaped, yTeacher.size[0] *yTeacher.size[1], yTeacher.size[2], false);

//            #pragma omp parallel for
//                for(int ii = 0; ii < yTeacher.size[0]; ++ii)
//                {
//                    for(int jj = 0; jj < yTeacher.size[1]; ++jj)
//                    {
//                        for(int kk = 0; kk < yTeacher.size[2]; ++kk)
//                        {
//                            l_yTeacherReshaped.at<T>(ii*yTeacher.size[1] + jj,kk) = yTeacher.at<T>(ii,jj,kk);
//                        }
//                    }
//                }
//            // end pragma

//            m_wOut = l_yTeacherReshaped.t() * l_tempCudaMult;
//        }
//        else
//        {
//            cv::Mat l_yTeacherReshaped;
//            initMatrix<T>(l_yTeacherReshaped, yTeacher.size[0] *yTeacher.size[1], yTeacher.size[2], false);

//            #pragma omp parallel for
//                for(int ii = 0; ii < yTeacher.size[0]; ++ii)
//                {
//                    for(int jj = 0; jj < yTeacher.size[1]; ++jj)
//                    {
//                        for(int kk = 0; kk < yTeacher.size[2]; ++kk)
//                        {
//                            l_yTeacherReshaped.at<T>(ii*yTeacher.size[1] + jj,kk) = yTeacher.at<T>(ii,jj,kk);
//                        }
//                    }
//                }
//            // end pragma

//            m_wOut = l_yTeacherReshaped.t() * l_xTotReshaped.t() * invCuda;
//        }
//    }
//    else
//    {
//        cv::invert(l_mat2inv, invCV, cv::DECOMP_SVD);
//        l_mat2inv.release();

//        displayTime("2-3 : tikhonovRegularization ", m_oTime, false, m_verbose);

//        cv::Mat l_yTeacherReshaped;
//        initMatrix<T>(l_yTeacherReshaped, yTeacher.size[0] *yTeacher.size[1], yTeacher.size[2], false);

//        #pragma omp parallel for
//            for(int ii = 0; ii < yTeacher.size[0]; ++ii)
//            {
//                for(int jj = 0; jj < yTeacher.size[1]; ++jj)
//                {
//                    for(int kk = 0; kk < yTeacher.size[2]; ++kk)
//                    {
//                        l_yTeacherReshaped.at<T>(ii*yTeacher.size[1] + jj,kk) = yTeacher.at<T>(ii,jj,kk);
//                    }
//                }
//            }
//        // end pragma

//        m_wOut = (l_yTeacherReshaped.t() * l_xTotReshaped.t()) * invCV;
//    }


//    displayTime("END : tikhonovRegularization ", m_oTime, false, m_verbose);
//}


//template<class T>
//void Reservoir2Qt<T>::train(const cv::Mat &meaningInputTrain, const cv::Mat &teacher, cv::Mat &sentencesOutputTrain, cv::Mat &xTot)
//{
//    m_oTime = clock();

//    displayTime("START : train ", m_oTime, false, m_verbose);

//    // generate matrices
//        generateMatrixW();
//        generateWIn(meaningInputTrain.size[2]);

//    displayTime("START : sub train ", m_oTime, false, m_verbose);

//        int l_sizeTot[3] = {meaningInputTrain.size[0], 1 + meaningInputTrain.size[2] + m_nbNeurons,  meaningInputTrain.size[1]};
//        int l_size[1] = {m_w.rows}; // OPTI

//        cv::Mat l_X2Copy, l_xPrev2Copy;
//        if(typeid(T) == typeid(float))
//        {
//            xTot = cv::Mat (3,l_sizeTot, CV_32FC1, cv::Scalar(0.f)); //  will contain the internal states of the reservoir for all sentences and all timesteps
//            l_X2Copy = cv::Mat::zeros(1 + meaningInputTrain.size[2] + m_nbNeurons, meaningInputTrain.size[1], CV_32FC1); // OPTI
//            l_xPrev2Copy = cv::Mat(1,l_size, CV_32FC1, cv::Scalar(0.f)); // OPTI
//        }
//        else
//        {
//            xTot = cv::Mat (3,l_sizeTot, CV_64FC1, cv::Scalar(0.0)); //  will contain the internal states of the reservoir for all sentences and all timesteps
//            l_X2Copy = cv::Mat::zeros(1 + meaningInputTrain.size[2] + m_nbNeurons, meaningInputTrain.size[1], CV_64FC1); // OPTI
//            l_xPrev2Copy = cv::Mat(1,l_size, CV_64FC1, cv::Scalar(0.0)); // OPTI
//        }

//        T l_invLeakRate = 1 - m_leakRate;

////        fillRandomMat_(xTot);
////        #pragma omp parallel for num_threads(7)
//        #pragma omp parallel for
//            for(int ii = 0; ii < meaningInputTrain.size[0]; ++ii)
//            {
//                if(m_verbose)
//                {
//                    printf("input train : %d / %d\n", ii+1,meaningInputTrain.size[0]);
//                }

//                cv::Mat l_X = l_X2Copy.clone();
//                cv::Mat l_xPrev, l_x;
//                cv::Mat l_subMean;

//                if(typeid(T) == typeid(float))
//                {
//                    l_subMean = cv::Mat(meaningInputTrain.size[1], meaningInputTrain.size[2], CV_32FC1, meaningInputTrain.data + meaningInputTrain.step[0] *ii);
//                }
//                else
//                {
//                    l_subMean = cv::Mat(meaningInputTrain.size[1], meaningInputTrain.size[2], CV_64FC1, meaningInputTrain.data + meaningInputTrain.step[0] *ii);
//                }

//                for(int jj = 0; jj < meaningInputTrain.size[1]; ++jj)
//                {
//                    // X will contain all the internal states of the reservoir for all timesteps
//                    if(jj == 0)
//                    {
//                        l_xPrev = l_xPrev2Copy;
//                    }
//                    else
//                    {
//                        l_xPrev = l_x;
//                    }

//                    cv::Mat l_u = l_subMean.row(jj);
//                    cv::Mat l_temp;
//                    initMatrix<T>(l_temp, l_subMean.cols+1, 1, false);
//                    l_temp.at<T>(0) = 1;

//                    for(int kk = 0; kk < l_subMean.cols; ++kk)
//                    {
//                        l_temp.at<T>(kk+1) = l_u.at<T>(kk);
//                    }

//                    cv::Mat l_xTemp = (m_wIn * l_temp)+ (m_w * l_xPrev);

//                    cv::MatIterator_<T> it = l_xTemp.begin<T>(), it_end = l_xTemp.end<T>();
//                    for(;it != it_end; ++it)
//                    {
//                        (*it) = static_cast<T>(tanh(*it));
//                    }

//                    l_x = (l_xPrev * l_invLeakRate) + (l_xTemp * m_leakRate);

//                    cv::Mat l_temp2;
//                    initMatrix<T>(l_temp2, l_temp.rows + l_x.rows, 1, false);

//                    for(int kk = 0; kk < l_temp.rows + l_x.rows; ++kk)
//                    {
//                        if(kk < l_temp.rows)
//                        {
//                            l_temp2.at<T>(kk) = l_temp.at<T>(kk);
//                        }
//                        else
//                        {
//                            l_temp2.at<T>(kk) = l_x.at<T>(kk - l_temp.rows);
//                        }
//                    }

//                    l_temp2.copyTo( l_X.col(jj));
//                }

//                for(int jj = 0; jj < l_X.rows; ++jj)
//                {
//                    for(int kk = 0; kk < l_X.cols; ++kk)
//                    {
//                        xTot.at<T>(ii,jj,kk) = l_X.at<T>(jj,kk);
//                    }
//                }

//            }
//        // end pragma

//        l_X2Copy.release();
//        l_xPrev2Copy.release();

//        displayTime("END : sub train ", m_oTime, false, m_verbose);

//        tikhonovRegularization(xTot, teacher, meaningInputTrain.size[2]);

//        sentencesOutputTrain = teacher.clone();
//        sentencesOutputTrain.setTo(0.0);

//        #pragma omp parallel for
//            for(int ii = 0; ii < xTot.size[0]; ++ii)
//            {
//                cv::Mat l_X;
//                initMatrix<T>(l_X, xTot.size[1], xTot.size[2], true);

//                for(int jj = 0; jj < l_X.rows; ++jj)
//                {
//                    for(int kk = 0; kk < l_X.cols; ++kk)
//                    {
//                        l_X.at<T>(jj,kk) = xTot.at<T>(ii,jj,kk);
//                    }
//                }

//                cv::Mat res;

//                res = (m_wOut * l_X).t();

//                for(int jj = 0; jj < sentencesOutputTrain.size[1]; ++jj)
//                {
//                    for(int kk = 0; kk < sentencesOutputTrain.size[2]; ++kk)
//                    {
//                        sentencesOutputTrain.at<T>(ii,jj,kk) = res.at<T>(jj,kk);
//                    }
//                }
//            }
//        // end pragma

//    displayTime("END : train ", m_oTime, false, m_verbose);

//}


//template<class T>
//void Reservoir2Qt<T>::test(const cv::Mat &meaningInputTest, cv::Mat &sentencesOutputTest, cv::Mat &xTot)
//{
//    m_oTime = clock();

//    displayTime("START : test", m_oTime, false, m_verbose);

//    int l_sizeTot[3] = {meaningInputTest.size[0], 1 + meaningInputTest.size[2] + m_nbNeurons,  meaningInputTest.size[1]};
//    int l_sizeOut[3] = {l_sizeTot[0], l_sizeTot[2], m_wOut.rows};


//    if(typeid(T) == typeid(float))
//    {
//        xTot = cv::Mat (3,l_sizeTot, CV_32FC1); //  will contain the internal states of the reservoir for all sentences and all timesteps
//        sentencesOutputTest = cv::Mat(3, l_sizeOut, CV_32FC1);

//    }
//    else
//    {
//        xTot = cv::Mat (3,l_sizeTot, CV_64FC1); //  will contain the internal states of the reservoir for all sentences and all timesteps
//        sentencesOutputTest = cv::Mat(3, l_sizeOut, CV_64FC1);

//    }

//    T l_invLeakRate = 1 - m_leakRate;

//    #pragma omp parallel for
//        for(int ii = 0; ii < meaningInputTest.size[0]; ++ii)
//        {
//            cv::Mat l_X;
//            initMatrix<T>(l_X, 1 + meaningInputTest.size[2] + m_nbNeurons,meaningInputTest.size[1], true);

//            cv::Mat l_xPrev, l_x;
//            cv::Mat l_subMean;

//            if(typeid(T) == typeid(float))
//            {
//                l_subMean = cv::Mat(meaningInputTest.size[1], meaningInputTest.size[2], CV_32FC1, meaningInputTest.data + meaningInputTest.step[0] *ii);
//            }
//            else
//            {
//                l_subMean = cv::Mat(meaningInputTest.size[1], meaningInputTest.size[2], CV_64FC1, meaningInputTest.data + meaningInputTest.step[0] *ii);
//            }

//            for(int jj = 0; jj < meaningInputTest.size[1]; ++jj)
//            {
//                // X will contain all the internal states of the reservoir for all timesteps
//                if(jj == 0)
//                {
//                    int l_size[1] = {m_w.rows};

//                    if(typeid(T) == typeid(float))
//                    {
//                        l_xPrev = cv::Mat(1,l_size, CV_32FC1, cv::Scalar(0.f));
//                    }
//                    else
//                    {
//                        l_xPrev = cv::Mat(1,l_size, CV_64FC1, cv::Scalar(0.0));
//                    }
//                }
//                else
//                {
//                    l_xPrev = l_x;
//                }

//                cv::Mat l_u = l_subMean.row(jj);
//                cv::Mat l_temp;
//                initMatrix<T>(l_temp,l_subMean.cols+1, 1);

//                l_temp.at<T>(0) = 1;
//                for(int kk = 0; kk < l_subMean.cols; ++kk)
//                {
//                    l_temp.at<T>(kk+1) = l_u.at<T>(kk);
//                }

//                cv::Mat l_xTemp = (m_wIn * l_temp) + (m_w * l_xPrev);

//                cv::MatIterator_<T> it = l_xTemp.begin<T>(), it_end = l_xTemp.end<T>();
//                for(;it != it_end; ++it)
//                {
//                    (*it) = tanh(*it);
//                }

//                l_x = (l_xPrev * l_invLeakRate) + (l_xTemp * m_leakRate);

//                cv::Mat l_temp2;
//                initMatrix<T>(l_temp,l_temp.rows + l_x.rows, 1, false);

//                for(int kk = 0; kk < l_temp.rows + l_x.rows; ++kk)
//                {
//                    if(kk < l_temp.rows)
//                    {
//                        l_temp2.at<T>(kk) = l_temp.at<T>(kk);
//                    }
//                    else
//                    {
//                        l_temp2.at<T>(kk) = l_x.at<T>(kk - l_temp.rows);
//                    }
//                }

//                l_temp2.copyTo( l_X.col(jj));

//                cv::Mat l_temp3;
//                initMatrix<T>(l_temp3,l_x.rows + l_subMean.cols+1, 1, false);


//                l_temp3.at<T>(0) = 1;
//                for(int kk = 0; kk < l_subMean.cols; ++kk)
//                {
//                    l_temp3.at<T>(kk+1) = l_u.at<T>(kk);
//                }
//                for(int kk = 0; kk < l_x.rows; ++kk)
//                {
//                    l_temp3.at<T>(kk+l_subMean.cols+1) = l_x.at<T>(kk);
//                }

//                cv::Mat l_y = m_wOut * l_temp3;
//                for(int kk = 0; kk < sentencesOutputTest.size[2]; ++kk)
//                {
//                    sentencesOutputTest.at<T>(ii,jj,kk) = l_y.at<T>(kk);
//                }
//            }
//            for(int jj = 0; jj < l_X.rows; ++jj)
//            {
//                for(int kk = 0; kk < l_X.cols; ++kk)
//                {
//                    xTot.at<T>(ii,jj,kk) = l_X.at<T>(jj,kk);
//                }
//            }
//        }
//    // end omp parallel

//     displayTime("END : test", m_oTime, false, m_verbose);
//}


#endif
