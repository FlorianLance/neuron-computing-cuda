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
 * \file Model.h
 * \brief defines Model
 * \author Florian Lance
 * \date 02/12/14
 */

#ifndef MODEL_H
#define MODEL_H

#include "Reservoir.h"
#include "CorpusProcessing.h"


/**
 * @brief The ModelParameters struct
 * Defines the parameters used by the model.
 */
struct ModelParameters
{
    /**
     * @brief display the current values of the parameters with std::cout
     */
    void display()
    {
        std::cout << "### Parameters ###" << std::endl;
        std::cout << "Corpus          : " << m_corpusFilePath << std::endl;
        std::cout << "Neurons number  : " << m_nbNeurons << std::endl;
        std::cout << "Leak rate       : " << m_leakRate << std::endl;
        std::cout << "Sparcity        : " << m_sparcity << std::endl;
        std::cout << "Input scaling   : " << m_inputScaling << std::endl;
        std::cout << "Ridge           : " << m_ridge << std::endl;
        std::cout << "Spectral radius : " << m_spectralRadius << std::endl;
    }

    /**
     * @brief initialization to be used in the main function in order to retrieve parameters written in the command prompt.
     * will initialize with default values anyway.
     * @param [in] argc : main argc
     * @param [in] argv : main argv
     */
    void init(int argc, char *argv[])
    {
        m_useCudaInv  = true;
        m_useCudaMult = true;
        m_randomSeedNumberGenerator = true;
        m_seedNumberGenerator = 1;
        m_nbNeurons      = 800;
        m_spectralRadius = 3;
        m_inputScaling   = 0.1;
        m_leakRate       = 0.1;
        m_ridge          = 1e-5;

        // retrieve cmd inputs
            if(argc > 1)
            {
                m_nbNeurons = QString(argv[1]).toInt();
            }
            if(argc > 2)
            {
                m_spectralRadius = QString(argv[2]).toInt();
            }
            if(argc > 3)
            {
                m_inputScaling = QString(argv[3]).toDouble();
            }
            if(argc > 4)
            {
                m_leakRate = QString(argv[4]).toDouble();
            }
            if(argc > 5)
            {
                m_sparcity = QString(argv[5]).toDouble();
            }

        m_sparcity       = m_nbNeurons / 10.0;
    }

    bool m_useLoadedTraining;       /**< ...  */
    bool m_useLoadedW;              /**< ...  */
    bool m_useLoadedWIn;            /**< ...  */
    bool m_randomSeedNumberGenerator; /**< ... */
    int  m_seedNumberGenerator;       /**< ... */


    // cuda
    bool m_useCudaInv;              /**< uses the cuda inversion ? */
    bool m_useCudaMult;             /**< uses the cuda multiplication ? */

    // corpus
    std::string m_corpusFilePath;   /**< corpus file path */

    // reservoir
    int m_nbNeurons;                /**< define the efficency and the complexity of the reservoir */
    double m_spectralRadius;        /**< max eigen value of W, affects the entropy of the reservoir */
    double m_inputScaling;          /**< interval repartition of the values of wWin, low value -> great sensibility / hight value -> overfitting */
    double m_leakRate;              /**< capacity of the resvoir for keeping the intern states of the previous times, low value -> stagnation / hight value -> evolvtion too fast */
    double m_sparcity;              /**< sparcity of W */
    double m_ridge;                 /**< ridge... */
};


/**
 * @brief The Model class
 */
class Model : public QObject
{
    Q_OBJECT

    public :

        /**
         * @brief Model default constructor.
         */
        Model();

        /**
         * @brief Model constructor.
         * @param [in] parameters : parameters used for initializeing the reservoirs and the pathes
         */
        Model(const ModelParameters &parameters);


        ~Model();

        /**
         * @brief return the parameters of the model.
         * @return a ModelParameters
         */
        ModelParameters parameters() const;

        /**
         * @brief resetModel
         * @param newParameters
         * @param verbose
         */
        void resetModelParameters(const ModelParameters &newParameters, cbool verbose);

        /**
         * @brief setCCWAndStructure
         * @param [in] CCW       : CCW to be used
         * @param [in] structure : structure to be used
         */
        void setCCWAndStructure(const Sentence &CCW, const Sentence &structure);

        /**
         * @brief launchTraining
         * @return
         */
        bool launchTraining();

        /**
         * @brief launchTests
         * @return
         */
        bool launchTests();


        /**
         * @brief sentences
         * @param trainSentences
         * @param trainResults
         * @param testResults
         */
        void sentences(Sentences &trainSentences, Sentences &trainResults, Sentences &testResults);

        /**
         * @brief displayResults
         * @param trainResults
         * @param testResults
         */
        void displayResults(const bool trainResults = true, const bool testResults = true);

        /**
         * @brief computeResultsData
         * @param trainResults
         * @param diffSizeOCW
         * @param absoluteCCW
         * @param continuousCCW
         * @param absoluteAll
         * @param continuousAll
         * @param meanDiffSizeOCW
         * @param meanAbsoluteCCW
         * @param meanContinuousCCW
         * @param meanAbsoluteAll
         * @param meanContinuousAll
         */
        void computeResultsData(cbool trainResults,
                                std::vector<double> &diffSizeOCW,
                                std::vector<double> &absoluteCCW, std::vector<double> &continuousCCW,
                                std::vector<double> &absoluteAll, std::vector<double> &continuousAll,
                                double &meanDiffSizeOCW,
                                double &meanAbsoluteCCW, double &meanContinuousCCW,
                                double &meanAbsoluteAll, double &meanContinuousAll
                                );

        /**
         * @brief saveResults
         */
        void saveResults();

        /**
         * @brief saveTraining
         * @param pathDirectory
         */
        void saveTraining(const std::string &pathDirectory);

        /**
         * @brief saveReplay
         * @param pathDirectory
         */
        void saveReplay(const std::string &pathDirectory);

        /**
         * @brief loadTraining
         * @param pathDirectory
         */
        void loadTraining(const std::string &pathDirectory);

        /**
         * @brief loadW
         * @param pathDirectory
         */
        void loadW(const std::string &pathDirectory);

        /**
         * @brief loadWIn
         * @param pathDirectory
         */
        void loadWIn(const std::string &pathDirectory);

        /**
         * @brief saveParamFile
         * @param pathDirectory
         */
        void saveParamFile(const std::string &pathDirectory);

        /**
         * @brief saveW
         * @param pathDirectory
         */
        void saveW(const std::string &pathDirectory);

        /**
         * @brief saveWIn
         * @param pathDirectory
         */
        void saveWIn(const std::string &pathDirectory);

        /**
         * @brief reservoir
         * @return
         */
        Reservoir *reservoir();

        /**
         * @brief xTotMatrice
         * @return
         */
        cv::Mat *xTotMatrice();


        Sentences m_recoveredSentencesTrain;    /**< ... */
        Sentences m_recoveredSentencesTest;     /**< ... */
        Sentences m_testSentence;               /**< corpus test sentence */


    signals :

        /**
         * @brief sendTrainInputMatrixSignal
         */
        void sendTrainInputMatrixSignal(cv::Mat, cv::Mat, Sentences);

        /**
         * @brief sendLogInfo
         */
        void sendLogInfo(QString, QColor);

        /**
         * @brief sendOutputMatrix
         */
        void sendOutputMatrix(cv::Mat, Sentences);

    private :

        /**
         * @brief retrieveTrainSentences
         */
        void retrieveTrainSentences();

        /**
         * @brief retrieveTestsSentences
         */
        void retrieveTestsSentences();


    private :

        // states
        bool m_verbose;                         /**< display infos during the processing */
        bool m_trainingSuccess;                 /**< is the training successful */

        Sentence m_CCW;                         /**< CCW used in the corpus */
        Sentence m_structure;                   /**< structure used in the corpus */
        Sentence m_closedClassWords;            /**< closed class words */

        // corpus train data
        Sentences m_trainMeaning;               /**< corpus train meaning    -> ex : gave dog toy girl , chase dog cat  */
        Sentences m_trainInfo;                  /**< corpus train info    -> ex : [A-_-_-P-O-R-_-_][A-P-O-_-_-_-_-_] */
        Sentences m_trainSentence;              /**< corpus train sentence -> ex : the dog that chase the cat gave the toy to the girl */

        // corpus test data
        Sentences m_testMeaning;                /**< corpus test meaning */
        Sentences m_testInfo;                   /**< corpus test info */


        // results of the reservoir
        cv::Mat m_3DMatSentencesOutputTrain;                /**< ... */
        cv::Mat m_3DMatSentencesOutputTest;                 /**< ... */
        cv::Mat m_internalStatesTrain;                      /**< ... */
        std::vector<cv::Mat> m_3DVMatSentencesOutputTrain;  /**< ... */
        std::vector<cv::Mat> m_3DVMatSentencesOutputTest;   /**< ... */

        // desired results
        Sentences m_desiredSentencesTest;       /**< ... */

        // parameters
        ModelParameters m_parameters;           /**< reservoir parameters and corpus pathes */

        // reservoir
        Reservoir *m_reservoir;                  /**< reservoir structure */
};

#endif
