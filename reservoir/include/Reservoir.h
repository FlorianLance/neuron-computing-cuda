

/**
 * \file Reservoir.h
 * \brief defines Reservoir
 * \author Florian Lance
 * \date 02/12/14
 */

/*! \mainpage
 *
 * \section Neuron-computing-cuda
 *
 * \subsection intro_sec Introduction
 *
 * An interface for language learning with neuron computing using GPU acceleration.
 *
 * A Yarp module of the program is also available for remote computing on a CUDA working station.
 *
 * \subsection install_sec Install
 *
 * Go to https://github.com/FlorianLance/neuron-computing-cuda for build procedures.
 *
 *
 * \subsection content_sec Content
 *
 *
 * \subsubsection Tutorial_inteface
 *
 * - https://github.com/FlorianLance/neuron-computing-cuda/wiki/tuto_interface
 *
 * \subsubsection Tutorial_yarp
 *
 * - https://github.com/FlorianLance/neuron-computing-cuda/wiki/tuto_yarp
 *
 * \subsection wiki_sec WIKI
 *
 * Go to https://github.com/FlorianLance/neuron-computing-cuda for more infos.
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
         * @return
         */
        bool train(const cv::Mat &meaningInputTrain, const cv::Mat &teacher, cv::Mat &sentencesOutputTrain, cv::Mat &xTot);

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

        /**
         * @brief saveW
         * @param path
         */
        void saveW(const std::string &path);

        /**
         * @brief saveWIn
         * @param path
         */
        void saveWIn(const std::string &path);

        /**
         * @brief saveParamFile
         * @param path
         */
        void saveParamFile(const std::string &path);

        /**
         * @brief loadParam
         * @param path
         */
        void loadParam(const std::string &path);

    public slots :

        /**
         * @brief enableMaxOmpThreadNumber
         * @param enable
         */
        void enableMaxOmpThreadNumber(bool enable);

        /**
         * @brief stopLoop
         */
        void stopLoop();


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

        bool m_useW;                    /**< uses custom W matrice */
        bool m_useWIn;                  /**< uses custom WIn matrice */

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

        bool m_stopLoop;                /**< is the loop must be stoped ? */
        QReadWriteLock m_stopLocker;    /**< stop loop locker */
};


#endif
