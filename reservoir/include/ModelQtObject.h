/**
 * \file ModelQtObject.h
 * \brief defines Model
 * \author Florian Lance
 * \date 02/12/14
 */

#ifndef MODELQTOBJECT_H
#define MODELQTOBJECT_H


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
class ModelObject : public QObject
{
    public :

        /**
         * @brief Model default constructor.
         */
        ModelObject();

        /**
         * @brief Model constructor.
         * @param [in] parameters : parameters used for initializeing the reservoirs and the pathes
         */
        ModelObject(const ModelParameters &parameters);

        /**
         * @brief return the parameters of the model.
         * @return a ModelParameters
         */
        ModelParameters parameters() const;

        /**
         * @brief resetModelF
         * @param newParameters
         * @param verbose
         */
        void resetModelF(const ModelParameters &newParameters, cbool verbose);

        /**
         * @brief setGrammar
         * @param [in] grammar   : gramar to be used
         * @param [in] structure : structure to be used
         */
        void setGrammar(const Sentence &grammar, const Sentence &structure);

        /**
         * @brief launchTrainingF
         */
        void launchTrainingF();

        /**
         * @brief launchTestsF
         * @param corpusTestFilePath
         */
        void launchTestsF(const std::string &corpusTestFilePath = "");

        /**
         * @brief retrieveTrainSentences
         */
        void retrieveTrainSentences();

        /**
         * @brief retrieveTestsSentences
         */
        void retrieveTestsSentences();

        /**
         * @brief displayResults
         * @param trainResults
         * @param testResults
         */
        void displayResults(const bool trainResults = true, const bool testResults = true);

        /**
         * @brief setResultsTestToCompare
         * @param resultsTestFilePath
         */
        void setResultsTestToCompare(const std::string &resultsTestFilePath);


        /**
         * @brief computeCCWResult
         * @param trainResults
         * @param CCWrightAbsolutePercentage
         * @param CCWcorrectPositionAndWordPercentage
         */
        void computeCCWResult(cbool trainResults, std::vector<double> &CCWrightAbsolutePercentage, std::vector<double> &CCWcorrectPositionAndWordPercentage);

        /**
         * @brief compareResults
         * @param trainResults
         * @param correctPositionAndWordPercentage
         * @param sentenceRightAbsolutePercentage
         * @param sizeDifferencePercentage
         * @param totalWordNumber
         * @param totalWordCorrectNumber
         */
        void compareResults(cbool trainResults, std::vector<double> &correctPositionAndWordPercentage,
                                                std::vector<double> &sentenceRightAbsolutePercentage,
                                                std::vector<double> &sizeDifferencePercentage,
                                                int &totalWordNumber, int &totalWordCorrectNumber);


        /**
         * @brief computeResultsData
         * @param trainResults
         * @param pathSaveAllSentenceRest
         * @param diffSizeOCW
         * @param absoluteCorrectPositionAndWordCCW
         * @param correctPositionAndWordCCW
         * @param meanDiffSizeOCW
         * @param meanAbsoluteCorrectPositionAndWordCCW
         * @param meanCorrectPositionAndWordCCW
         */
        void computeResultsData(cbool trainResults, const std::string &pathSaveAllSentenceRest,
                                std::vector<double> diffSizeOCW, std::vector<double> absoluteCorrectPositionAndWordCCW, std::vector<double> correctPositionAndWordCCW,
                                double &meanDiffSizeOCW, double &meanAbsoluteCorrectPositionAndWordCCW, double &meanCorrectPositionAndWordCCW);



        /**
         * @brief saveResults
         */
        void saveResults();

        Sentences m_recoveredSentencesTrain;    /**< ... */
        Sentences m_recoveredSentencesTest;     /**< ... */
        Sentences m_testSentence;               /**< corpus test sentence */

    private :

        // states
        bool m_verbose;                         /**< display infos during the processing */
        bool m_trainingSuccess;                 /**< is the training successful */

        Sentence m_grammar;                     /**< grammar used in the corpus */
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
        std::vector<cv::Mat> m_3DVMatSentencesOutputTrain;  /**< ... */
        std::vector<cv::Mat> m_3DVMatSentencesOutputTest;   /**< ... */

        // desired results
        Sentences m_desiredSentencesTest;       /**< ... */

        // parameters
        ModelParameters m_parameters;           /**< reservoir parameters and corpus pathes */

        // reservoir
        Reservoir m_reservoir;                  /**< reservoir structure */
};

#endif
