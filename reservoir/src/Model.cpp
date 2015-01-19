

/**
 * \file Model.cpp
 * \brief defines Model
 * \author Florian Lance
 * \date 01/10/14
 */


#include "Model.h"


#include "../moc/moc_Model.cpp"

static int s_numImage = 0;


Model::Model() : m_trainingSuccess(false), m_verbose(true), m_reservoir(new Reservoir()) {}

Model::Model(const ModelParameters &parameters) : m_parameters(parameters), m_trainingSuccess(false), m_verbose(true)
{
    m_reservoir = new Reservoir(m_parameters.m_nbNeurons, m_parameters.m_spectralRadius, m_parameters.m_inputScaling, m_parameters.m_leakRate, m_parameters.m_sparcity, m_parameters.m_ridge, m_verbose);
}

Model::~Model()
{
    delete m_reservoir;
}

ModelParameters Model::parameters() const
{
    return m_parameters;
}

void Model::resetModelParameters(const ModelParameters &newParameters, cbool verbose)
{
    m_parameters = newParameters;
    m_verbose    = verbose;

    // resert training state
        m_trainingSuccess = false;

    // update the matrices with the training loaded
        if(m_parameters.m_useLoadedTraining)
        {
            m_reservoir->updateMatricesWithLoadedTraining();
            m_trainingSuccess = true;
        }

    // matrices custom
        m_reservoir->setMatricesUse(newParameters.m_useLoadedW, newParameters.m_useLoadedWIn);

    // reservoir
        m_reservoir->setParameters(m_parameters.m_nbNeurons, static_cast<float>(m_parameters.m_spectralRadius), static_cast<float>(m_parameters.m_inputScaling)
                              ,static_cast<float>(m_parameters.m_leakRate), static_cast<float>(m_parameters.m_sparcity), static_cast<float>(m_parameters.m_ridge), m_verbose);

    // CUDA
        m_reservoir->setCudaProperties(m_parameters.m_useCudaInv, m_parameters.m_useCudaMult);
}

void Model::setCCWAndStructure(const Sentence &CCW, const Sentence &structure)
{
    m_CCW   = CCW;
    m_structure = structure;
}

void Model::retrieveTrainSentences()
{
    // generate open class word arrays
        Sentences l_trainOCW;
        generateOCWArray(m_trainMeaning, m_trainInfo, l_trainOCW);

    // convert signal
        Sentences l_recoveredConstructionTrain;
        convertLOutputActivityInConstruction(m_3DMatSentencesOutputTrain, m_closedClassWords, l_recoveredConstructionTrain, 1);
        attributeOcwToConstructions(l_recoveredConstructionTrain, l_trainOCW, m_recoveredSentencesTrain, "X");
}

void Model::retrieveTestsSentences()
{
    // generate open class word arrays
        Sentences l_testOCW;
        generateOCWArray(m_testMeaning, m_testInfo, l_testOCW);

        displaySentence(m_testMeaning);
        displaySentence(m_testInfo);
        displaySentence(l_testOCW);

    // convert signal
        Sentences l_recoveredConstructionTest;
        convertLOutputActivityInConstruction(m_3DMatSentencesOutputTest, m_closedClassWords, l_recoveredConstructionTest, 2);
        attributeOcwToConstructions(l_recoveredConstructionTest, l_testOCW, m_recoveredSentencesTest, "X");
}

void Model::sentences(Sentences &trainSentences, Sentences &trainResults, Sentences &testResults)
{
    trainSentences = m_trainSentence;
    trainResults   = m_recoveredSentencesTrain;
    testResults    = m_recoveredSentencesTest;
}


void Model::displayResults(const bool trainResults, const bool testResults)
{
    if(trainResults)
    {
        if(m_verbose)
            std::cout << "Train retrieved sentences : " << std::endl;

        for(int ii = 0; ii < m_recoveredSentencesTrain.size(); ++ii)
        {
            for(int jj = 0; jj < m_trainSentence[ii].size(); ++jj)
            {
                std::cout << m_trainSentence[ii][jj] << " ";
            }
            std::cout << " -> ";
            for(int jj = 0; jj < m_recoveredSentencesTrain[ii].size(); ++jj)
            {
                std::cout << m_recoveredSentencesTrain[ii][jj] << " ";
            }

            std::cout << std::endl;
        }
    }

    if(testResults)
    {
        if(m_verbose)
            std::cout << "\nTest retrieved sentences : " << std::endl;

        for(int ii = 0; ii < m_recoveredSentencesTest.size(); ++ii)
        {
            for(int jj = 0; jj < m_recoveredSentencesTest[ii].size(); ++jj)
            {
                std::cout << m_recoveredSentencesTest[ii][jj] << " ";
            }

            std::cout << std::endl;
        }
    }

    std::cout << std::endl;
}


void Model::computeResultsData(cbool trainResults,
                        std::vector<double> &diffSizeOCW,
                        std::vector<double> &absoluteCCW, std::vector<double> &continuousCCW,
                        std::vector<double> &absoluteAll, std::vector<double> &continuousAll,
                        double &meanDiffSizeOCW,
                        double &meanAbsoluteCCW, double &meanContinuousCCW,
                        double &meanAbsoluteAll, double &meanContinuousAll
                        )
{
    if(m_verbose)
    {
        std::cout << "Start analysing results. " << std::endl;
        sendLogInfo("Start analysing results. \n", QColor(Qt::black));
    }

    Sentences l_goal;
    Sentences l_results;

    if(!trainResults)
    {
        l_results = m_recoveredSentencesTest;

        if(m_desiredSentencesTest.size() == 0)
        {
            l_goal = m_testSentence;
        }
        else
        {
            l_goal = m_desiredSentencesTest;
        }

        if(l_goal.size() != m_recoveredSentencesTest.size())
        {
            std::cerr << "Error compare results : not the same number of sentences. " << std::endl;
            sendLogInfo("Error compare results : not the same number of sentences. \n", QColor(Qt::red));
        }
    }
    else
    {
        l_goal = m_trainSentence;
        l_results = m_recoveredSentencesTrain;
    }

    Sentences l_goalCCWOnly, l_resultsCCWOnly;
    Sentences l_goalAll, l_resultAll;

    std::vector<int> l_diffSizesOCW;

    // suppress OCW from the sentences
    for(int aa = 0; aa < l_goal.size(); ++aa)
    {
        Sentence l_currentGoal = l_goal[aa];
        Sentence l_currentRes  = l_results[aa];
        Sentence l_currentGoalCCWOnly, l_currentResCCWOnly, l_currentGoalOCWOnly, l_currentResultsOCWOnly;
        Sentence l_currentGoalAll, l_currenResultAll;

        // goal
            for(int ii = 0; ii < l_currentGoal.size(); ++ii)
            {
                bool l_addWord = false;

                for(int jj = 0; jj < m_CCW.size(); ++jj)
                {
                    if(l_currentGoal[ii] == m_CCW[jj])
                    {
                        l_addWord = true;
                        break;
                    }
                }

                if(l_addWord)
                {
                    l_currentGoalCCWOnly.push_back(l_currentGoal[ii]);
                    l_currentGoalAll.push_back(l_currentGoal[ii]);
                }
                else
                {
                    l_currentGoalOCWOnly.push_back(l_currentGoal[ii]);
                    l_currentGoalAll.push_back("X");
                }
            }


        // result
            for(int ii = 0; ii < l_currentRes.size(); ++ii)
            {
                bool l_addWord = false;

                for(int jj = 0; jj < m_CCW.size(); ++jj)
                {
                    if(l_currentRes[ii] == m_CCW[jj])
                    {
                        l_addWord = true;
                        break;
                    }
                }

                if(l_addWord)
                {

                    l_currentResCCWOnly.push_back(l_currentRes[ii]);
                    l_currenResultAll.push_back(l_currentRes[ii]);
                }
                else
                {
                    l_currentResultsOCWOnly.push_back(l_currentRes[ii]);
                    l_currenResultAll.push_back("X");
                }
            }

        l_goalCCWOnly.push_back(l_currentGoalCCWOnly);
        l_resultsCCWOnly.push_back(l_currentResCCWOnly);
        l_goalAll.push_back(l_currentGoalAll);
        l_resultAll.push_back(l_currenResultAll);

        l_diffSizesOCW.push_back(static_cast<int>(l_currentResultsOCWOnly.size() - l_currentGoalOCWOnly.size()));
    }

    // reset mean results
        // OCW
        meanDiffSizeOCW = 0.0;
        // CCW
        meanAbsoluteCCW = 0.0;
        meanContinuousCCW = 0.0;
        // all
        meanAbsoluteAll = 0.0;
        meanContinuousAll = 0.0;

    // reset results arrays
        // OCW
        diffSizeOCW.clear();
        // CCW
        absoluteCCW.clear();
        continuousCCW.clear();
        // all
        absoluteAll.clear();
        continuousAll.clear();

    // compute the stats
    for(int ii = 0; ii < l_goalCCWOnly.size(); ++ii)
    {
        Sentence l_currentGoalCCWOnly = l_goalCCWOnly[ii];
        Sentence l_currentResCCWOnly  = l_resultsCCWOnly[ii];
        Sentence l_currentGoalAll     = l_goalAll[ii];
        Sentence l_currentResAll      = l_resultAll[ii];

        int l_correctPositionAndWordCCWOnly = 0;
        int l_correctPositionAndWordAll = 0;
        // sizes
            // CCW
            int l_nbWordsGoalCCWOnly   = static_cast<int>(l_currentGoalCCWOnly.size());
            int l_nbWordsResultCCWOnly = static_cast<int>(l_currentResCCWOnly.size());
            // all
            int l_nbWordsGoalAll = static_cast<int>(l_currentGoalAll.size());
            int l_nbWordsResAll = static_cast<int>(l_currentResAll.size());

        // CCW
        for(int jj = 0; jj < std::min(l_nbWordsGoalCCWOnly, l_nbWordsResultCCWOnly); ++jj)
        {
            if(l_currentGoalCCWOnly[jj] == l_currentResCCWOnly[jj])
            {
                ++l_correctPositionAndWordCCWOnly;
            }
        }
        // all
        for(int jj = 0; jj < std::min(l_nbWordsGoalAll, l_nbWordsResAll); ++jj)
        {
            if(l_currentGoalAll[jj] == l_currentResAll[jj])
            {
                ++l_correctPositionAndWordAll;
            }
        }

        // percent
            // CCW
            double l_percentCorrectPositionAndWordAll     = 100.0 * l_correctPositionAndWordAll     / std::max(l_nbWordsGoalAll,l_nbWordsResAll);
            // all
            double l_percentCorrectPositionAndWordCCWOnly = 100.0 * l_correctPositionAndWordCCWOnly / std::max(l_nbWordsGoalCCWOnly,l_nbWordsResultCCWOnly);

        // absolute
            double l_percentAbsoluteCCWOnly = 0.0;
            double l_percentAbsoluteAll = 0.0;
            // CCW
            if(l_percentCorrectPositionAndWordCCWOnly == 100)
            {
                l_percentAbsoluteCCWOnly = 100.0;
            }
            // all
            if(l_percentCorrectPositionAndWordAll == 100)
            {
                l_percentAbsoluteAll = 100.0;
            }

        // mean
            // CCW
            meanAbsoluteCCW += l_percentAbsoluteCCWOnly;
            meanContinuousCCW         += l_percentCorrectPositionAndWordCCWOnly;
            // OCW
            meanDiffSizeOCW                       += sqrt(static_cast<double>(l_diffSizesOCW[ii]*l_diffSizesOCW[ii]));
            // all
            meanAbsoluteAll += l_percentAbsoluteAll;
            meanContinuousAll         += l_percentCorrectPositionAndWordAll;

        // add results to array
            // OCW
            diffSizeOCW.push_back(l_diffSizesOCW[ii]);
            // CCW
            absoluteCCW.push_back(l_percentAbsoluteCCWOnly);
            continuousCCW.push_back(l_percentCorrectPositionAndWordCCWOnly);
            // all
            absoluteAll.push_back(l_percentAbsoluteAll);
            continuousAll.push_back(l_percentCorrectPositionAndWordAll);

        std::cout.precision(5);
        if(m_verbose)
        {
            std::cout << "|CCW absolute: " << l_percentAbsoluteCCWOnly << "% | All absolute : " << l_percentAbsoluteAll  << std::endl;
        }
    }

    // update mean res values
        // OCW
        meanDiffSizeOCW /= l_goalCCWOnly.size();
        // CCW
        meanContinuousCCW /= l_goalCCWOnly.size();
        meanAbsoluteCCW /= l_goalCCWOnly.size();
        // all
        meanContinuousAll /= l_goalAll.size();
        meanAbsoluteAll /= l_goalAll.size();
}

void Model::saveTraining(const std::string &pathDirectory)
{    
    m_reservoir->saveTraining(pathDirectory);
}

void Model::saveReplay(const std::string &pathDirectory)
{
    save3DMatrixToText(QString::fromStdString(pathDirectory) + "/xTot.txt", m_internalStatesTrain);
}

void Model::loadTraining(const std::string &pathDirectory)
{
    m_reservoir->loadTraining(pathDirectory);
}

void Model::loadW(const std::string &pathDirectory)
{
    m_reservoir->loadW(pathDirectory);
}

void Model::loadWIn(const std::string &pathDirectory)
{
    m_reservoir->loadWIn(pathDirectory);
}

Reservoir *Model::reservoir()
{
    return m_reservoir;
}

cv::Mat *Model::xTotMatrice()
{
    return &m_internalStatesTrain;
}

void Model::launchTraining()
{
    // init time
        clock_t l_trainingTime = clock();
        m_trainingSuccess = false;
        m_3DMatSentencesOutputTrain = cv::Mat();
        m_internalStatesTrain = cv::Mat();

    // generate close class word arrays
        m_closedClassWords.clear();
        if(m_CCW.size() > 0)
        {
            m_closedClassWords = m_CCW;
            m_closedClassWords.push_back("X");
        }
        else
        {
            closedClassWords(m_closedClassWords, "X");
        }

    // generate CCW python argument
        std::string l_CCWPythonArg;
        for(int ii = 0; ii < m_closedClassWords.size()-1; ++ii)
        {
            l_CCWPythonArg += m_closedClassWords[ii];

            if(ii < m_closedClassWords.size() -2)
            {
                l_CCWPythonArg += "_";
            }
        }

    // genertate structure python argument
        std::string l_structurePythonArg;
        for(int ii = 0; ii < m_structure.size(); ++ii)
        {
            l_structurePythonArg += m_structure[ii];

            if(ii < m_structure.size() -1)
            {
                l_structurePythonArg += "_";
            }
        }

    // call python for generating new stim files
        std::string l_pythonCmd("python ../generate_stim.py ");
        std::string l_pythonCall = l_pythonCmd + m_parameters.m_corpusFilePath + " train " + l_CCWPythonArg + " " + l_structurePythonArg;
        sendLogInfo(QString::fromStdString(displayTime("Generate stim files with Python ", l_trainingTime, false, m_verbose)), QColor(Qt::black));
            system(l_pythonCall.c_str());
        sendLogInfo(QString::fromStdString(displayTime("End generation ", l_trainingTime, true, m_verbose)), QColor(Qt::black));

    // init matrices
        cv::Mat l_3DMatStimMeanTrain, l_3DMatStimSentTrain;

    // load input matrices created in the python script)
        load3DMatrixFromNpPythonSaveTextF(QString("../data/input/stim_mean_train.txt"), l_3DMatStimMeanTrain);
        load3DMatrixFromNpPythonSaveTextF(QString("../data/input/stim_sent_train.txt"), l_3DMatStimSentTrain);

    // retrieve corpus train data
        QVector<QStringList> l_trainMeaning,l_trainInfo,l_trainSentence, l_inused;
        extractAllDataFromCorpusFile(m_parameters.m_corpusFilePath.c_str(), l_trainMeaning,l_trainInfo,l_trainSentence, l_inused,l_inused,l_inused);
        convQt2DString2Std2DString(l_trainMeaning, m_trainMeaning);
        convQt2DString2Std2DString(l_trainInfo, m_trainInfo);
        convQt2DString2Std2DString(l_trainSentence, m_trainSentence);

    // send train input matrices to be displayed
        sendTrainInputMatrixSignal(l_3DMatStimMeanTrain,l_3DMatStimSentTrain,m_trainSentence);

    // train reservoir        
        sendLogInfo(QString::fromStdString(displayTime("Start reservoir training ", l_trainingTime, false, m_verbose)), QColor(Qt::black));
            m_reservoir->train(l_3DMatStimMeanTrain, l_3DMatStimSentTrain, m_3DMatSentencesOutputTrain,m_internalStatesTrain);            
        sendLogInfo(QString::fromStdString(displayTime("End reservoir training ", l_trainingTime, true, m_verbose)), QColor(Qt::black));

        retrieveTrainSentences();

        m_trainingSuccess = true;

    // send output matrix for displaying CCW in the interface
        emit sendOutputMatrix(m_3DMatSentencesOutputTrain, m_recoveredSentencesTrain);
}


bool Model::launchTests(const std::string &corpusTestFilePath)
{
    // init time
        clock_t l_testTime = clock();
        m_3DMatSentencesOutputTest = cv::Mat();

    // check training
        if(!m_trainingSuccess)
        {
            std::cerr << "The training must be done before the tests. " << std::endl;
            sendLogInfo("The training must be done before the tests. \n", QColor(Qt::red));
            return false;
        }

    // check corpus path input
        std::string l_corpusFilePath = m_parameters.m_corpusFilePath;
        if(corpusTestFilePath.size() > 0)
        {
            l_corpusFilePath = corpusTestFilePath;
        }

    // generate close class word arrays
        m_closedClassWords.clear();
        if(m_CCW.size() > 0)
        {
            m_closedClassWords = m_CCW;
            m_closedClassWords.push_back("X");
        }
        else
        {
            closedClassWords(m_closedClassWords, "X");
        }

    // retrieve corpus test data
        QVector<QStringList> l_testMeaning,l_testInfo, l_inused;
        extractAllDataFromCorpusFile(l_corpusFilePath.c_str(), l_inused,l_inused,l_inused, l_testMeaning,l_testInfo,l_inused);
        convQt2DString2Std2DString(l_testMeaning, m_testMeaning);
        convQt2DString2Std2DString(l_testInfo, m_testInfo);

        if(m_testMeaning.size() == 0)
        {
            std::cerr << "Corpus test is empty. " << std::endl;
            sendLogInfo("Corpus test is empty. \n", QColor(Qt::red));
            return false;
        }


    // generate CCW python argument
        std::string l_CCWPythonArg;
        for(int ii = 0; ii < m_closedClassWords.size()-1; ++ii)
        {
            l_CCWPythonArg += m_closedClassWords[ii];

            if(ii < m_closedClassWords.size() -2)
            {
                l_CCWPythonArg += "_";
            }
        }

    // genertate structure python argument
        std::string l_structurePythonArg;
        for(int ii = 0; ii < m_structure.size(); ++ii)
        {
            l_structurePythonArg += m_structure[ii];

            if(ii < m_structure.size() -1)
            {
                l_structurePythonArg += "_";
            }
        }

    // call python for generating new stim files
        std::string l_pythonCmd("python ../generate_stim.py ");
        std::string l_pythonCall;
        l_pythonCall = l_pythonCmd + l_corpusFilePath + " test " + l_CCWPythonArg + " " + l_structurePythonArg;

        sendLogInfo(QString::fromStdString(displayTime("Generate stim files with Python ", l_testTime, false, m_verbose)), QColor(Qt::black));
            system(l_pythonCall.c_str());
        sendLogInfo(QString::fromStdString(displayTime("End generation ", l_testTime, true, m_verbose)), QColor(Qt::black));

    // init matrices
        cv::Mat l_3DMatStimMeanTest, l_internalStatesTest;

    // load input matrices created in the python script)
        load3DMatrixFromNpPythonSaveTextF(QString("../data/input/stim_mean_test.txt"),   l_3DMatStimMeanTest);

    // test reservoir
        sendLogInfo(QString::fromStdString(displayTime("Start reservoir testing ", l_testTime, false, m_verbose)), QColor(Qt::black));
            m_reservoir->test(l_3DMatStimMeanTest, m_3DMatSentencesOutputTest, l_internalStatesTest);
        sendLogInfo(QString::fromStdString(displayTime("End reservoir testing ", l_testTime, true, m_verbose)), QColor(Qt::black));

        retrieveTestsSentences();

    // send output matrix for displaying CCW in the interface
        emit sendOutputMatrix(m_3DMatSentencesOutputTest, m_recoveredSentencesTest);

    return true;
}
