

/**
 * \file ModelQtObject.cpp
 * \brief defines Model
 * \author Florian Lance
 * \date 01/10/14
 */


#include "ModelQtObject.h"


#include "../moc/moc_ModelQtObject.cpp"

static int s_numImage = 0;


ModelObject::ModelObject() : m_trainingSuccess(false), m_verbose(true) {}

ModelObject::ModelObject(const ModelParameters &parameters) : m_parameters(parameters), m_trainingSuccess(false), m_verbose(true)
{
     m_reservoir = Reservoir(m_parameters.m_nbNeurons, m_parameters.m_spectralRadius, m_parameters.m_inputScaling, m_parameters.m_leakRate, m_parameters.m_sparcity, m_parameters.m_ridge, m_verbose);


//     cvNamedWindow("reservoir_display", CV_WINDOW_AUTOSIZE | CV_GUI_NORMAL);
//     cvMoveWindow("reservoir_display",200,200);
}

ModelParameters ModelObject::parameters() const
{
    return m_parameters;
}

void ModelObject::resetModelF(const ModelParameters &newParameters, cbool verbose)
{
    m_verbose = verbose;
    m_trainingSuccess = false;
    m_parameters = newParameters;
    m_reservoir = Reservoir(m_parameters.m_nbNeurons, static_cast<float>(m_parameters.m_spectralRadius), static_cast<float>(m_parameters.m_inputScaling)
                            ,  static_cast<float>(m_parameters.m_leakRate), static_cast<float>(m_parameters.m_sparcity), static_cast<float>(m_parameters.m_ridge), m_verbose);
    m_reservoir.setCudaProperties(m_parameters.m_useCudaInv, m_parameters.m_useCudaMult);
}

void ModelObject::setGrammar(const Sentence &grammar, const Sentence &structure)
{
    m_grammar   = grammar;
    m_structure = structure;
}

void ModelObject::retrieveTrainSentences()
{
    // generate open class word arrays
        Sentences l_trainOCW;
        generateOCWArray(m_trainMeaning, m_trainInfo, l_trainOCW);

    // convert signal
        Sentences l_recoveredConstructionTrain;
        convertLOutputActivityInConstruction(m_3DMatSentencesOutputTrain, m_closedClassWords, l_recoveredConstructionTrain, 1);

//        qDebug() << "m_closedClassWords :";
//        displaySentence(m_closedClassWords);
//        qDebug() << "l_recoveredConstructionTrain :";
//        displaySentence(l_recoveredConstructionTrain);
        attributeOcwToConstructions(l_recoveredConstructionTrain, l_trainOCW, m_recoveredSentencesTrain, "X");
//        qDebug() << "m_recoveredSentencesTrain :";
//        displaySentence(m_recoveredSentencesTrain);
}


void ModelObject::retrieveTestsSentences()
{
    // generate open class word arrays
        Sentences l_testOCW;
        generateOCWArray(m_testMeaning, m_testInfo, l_testOCW);

    // convert signal
        Sentences l_recoveredConstructionTest;
        convertLOutputActivityInConstruction(m_3DMatSentencesOutputTest, m_closedClassWords, l_recoveredConstructionTest, 2);
        attributeOcwToConstructions(l_recoveredConstructionTest, l_testOCW, m_recoveredSentencesTest, "X");
}


void ModelObject::displayResults(const bool trainResults, const bool testResults)
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
            for(int jj = 0; jj < m_testSentence[ii].size(); ++jj)
            {
                std::cout << m_testSentence[ii][jj] << " ";
            }
            std::cout << " -> ";
            for(int jj = 0; jj < m_recoveredSentencesTest[ii].size(); ++jj)
            {
                std::cout << m_recoveredSentencesTest[ii][jj] << " ";
            }

            std::cout << std::endl;
        }
//        displaySentence(m_recoveredSentencesTest);
    }

    std::cout << std::endl;
}

void ModelObject::setResultsTestToCompare(const std::string &resultsTestFilePath)
{
    QFile l_file(resultsTestFilePath.c_str());
    if(l_file.open(QIODevice::ReadOnly | QIODevice::Text))
    {
        m_desiredSentencesTest.clear();

        QTextStream in(&l_file), inLine;

        while (!in.atEnd())
        {
            QString l_line = in.readLine();
            inLine.setString(&l_line);

            Sentence l_sentenceTest;

            while(!inLine.atEnd())
            {
                QString l_value;
                inLine >> l_value;
                l_sentenceTest.push_back(l_value.toStdString());
            }

            m_desiredSentencesTest.push_back(l_sentenceTest);
        }
    }
    else
    {
        std::cerr << "Error loading results test file. " << std::endl;
    }
}


void ModelObject::computeCCWResult(cbool trainResults, std::vector<double> &CCWrightAbsolutePercentage, std::vector<double> &CCWcorrectPositionAndWordPercentage)
{
    if(m_verbose)
    {
        std::cout << "Start analysing results. " << std::endl;
    }

    Sentences l_goal, l_results;
    Sentences l_goalCCWOnly, l_resultsCCWOnly;

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
        }
    }
    else
    {
        l_goal = m_trainSentence;
        l_results = m_recoveredSentencesTrain;
    }

    // suppress OCW from the sentences
    for(int aa = 0; aa < l_goal.size(); ++aa)
    {
        Sentence l_currentGoal = l_goal[aa];
        Sentence l_currentRes  = l_results[aa];
        Sentence l_currentGoalCCWOnly, l_currentResCCWOnly;

        // goal
            for(int ii = 0; ii < l_currentGoal.size(); ++ii)
            {
                bool l_addWord = false;



                for(int jj = 0; jj < m_grammar.size(); ++jj)
                {
                    if(l_currentGoal[ii] == m_grammar[jj])
                    {
                        l_addWord = true;
                        break;
                    }
                }

                if(l_addWord)
                {
                    l_currentGoalCCWOnly.push_back(l_currentGoal[ii]);
                }
            }


        // result
            for(int ii = 0; ii < l_currentRes.size(); ++ii)
            {
                bool l_addWord = false;

                for(int jj = 0; jj < m_grammar.size(); ++jj)
                {
                    if(l_currentRes[ii] == m_grammar[jj])
                    {
                        l_addWord = true;
                        break;
                    }
                }

                if(l_addWord)
                {

                    l_currentResCCWOnly.push_back(l_currentRes[ii]);
                }
            }

        l_goalCCWOnly.push_back(l_currentGoalCCWOnly);
        l_resultsCCWOnly.push_back(l_currentResCCWOnly);
    }


    // compute the percentages
    for(int ii = 0; ii < l_goalCCWOnly.size(); ++ii)
    {
        Sentence l_currenGoalCCWOnly = l_goalCCWOnly[ii];
        Sentence l_currenResCCWOnly  = l_resultsCCWOnly[ii];

        int l_correctPositionAndWord = 0;
        int l_nbWordsGoal   = static_cast<int>(l_currenGoalCCWOnly.size());
        int l_nbWordsResult = static_cast<int>(l_currenResCCWOnly.size());

        for(int jj = 0; jj < std::min(l_nbWordsGoal, l_nbWordsResult); ++jj)
        {
            if(l_currenGoalCCWOnly[jj] == l_currenResCCWOnly[jj])
            {
                ++l_correctPositionAndWord;
            }
        }

        double l_percentCorrectPositionAndWord = 100.0 * l_correctPositionAndWord / l_nbWordsGoal;

        double l_percentAbsolute = 0.0;
        if(l_percentCorrectPositionAndWord == 100)
        {
            l_percentAbsolute = 100.0;
        }

        std::cout.precision(5);
        if(m_verbose)
        {

            std::cout << "|CPW : " << l_percentCorrectPositionAndWord << "% | Abs : " << l_percentAbsolute << "% |  -> ";

            if(trainResults)
            {
                for(int jj = 0; jj < m_recoveredSentencesTrain[ii].size(); ++jj)
                {
                    std::cout << m_recoveredSentencesTrain[ii][jj] << " ";
                }
            }
            else
            {
                for(int jj = 0; jj < m_recoveredSentencesTest[ii].size(); ++jj)
                {
                    std::cout << m_recoveredSentencesTest[ii][jj] << " ";
                }
            }
            std::cout << std::endl;
        }

        CCWrightAbsolutePercentage.push_back(l_percentCorrectPositionAndWord);
        CCWcorrectPositionAndWordPercentage.push_back(l_percentAbsolute);
    }

    if(m_verbose)
    {
        std::cout << "End analysing CCW results. " << std::endl;
    }
}


void ModelObject::computeResultsData(cbool trainResults, const std::string &pathSaveAllSentenceRest,
                               std::vector<double> diffSizeOCW, std::vector<double> absoluteCorrectPositionAndWordCCW, std::vector<double> correctPositionAndWordCCW,
                               double &meanDiffSizeOCW, double &meanAbsoluteCorrectPositionAndWordCCW, double &meanCorrectPositionAndWordCCW)
{
    if(m_verbose)
    {
        std::cout << "Start analysing results. " << std::endl;
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
        }
    }
    else
    {
        l_goal = m_trainSentence;
        l_results = m_recoveredSentencesTrain;
    }

    Sentences l_goalCCWOnly, l_resultsCCWOnly;
    Sentences l_goalOCWOnly, l_resultsOCWOnly;

    std::vector<int> l_diffSizesOCW;

    // suppress OCW from the sentences
    for(int aa = 0; aa < l_goal.size(); ++aa)
    {
        Sentence l_currentGoal = l_goal[aa];
        Sentence l_currentRes  = l_results[aa];
        Sentence l_currentGoalCCWOnly, l_currentResCCWOnly, l_currentGoalOCWOnly, l_currentResultsOCWOnly;

        // goal
            for(int ii = 0; ii < l_currentGoal.size(); ++ii)
            {
                bool l_addWord = false;

                for(int jj = 0; jj < m_grammar.size(); ++jj)
                {
                    if(l_currentGoal[ii] == m_grammar[jj])
                    {
                        l_addWord = true;
                        break;
                    }
                }

                if(l_addWord)
                {
                    l_currentGoalCCWOnly.push_back(l_currentGoal[ii]);
                }
                else
                {
                    l_currentGoalOCWOnly.push_back(l_currentGoal[ii]);
                }
            }


        // result
            for(int ii = 0; ii < l_currentRes.size(); ++ii)
            {
                bool l_addWord = false;

                for(int jj = 0; jj < m_grammar.size(); ++jj)
                {
                    if(l_currentRes[ii] == m_grammar[jj])
                    {
                        l_addWord = true;
                        break;
                    }
                }

                if(l_addWord)
                {

                    l_currentResCCWOnly.push_back(l_currentRes[ii]);
                }
                else
                {
                    l_currentResultsOCWOnly.push_back(l_currentRes[ii]);
                }
            }

        l_goalCCWOnly.push_back(l_currentGoalCCWOnly);
        l_resultsCCWOnly.push_back(l_currentResCCWOnly);
        l_goalOCWOnly.push_back(l_currentGoalOCWOnly);
        l_resultsOCWOnly.push_back(l_currentResultsOCWOnly);

        l_diffSizesOCW.push_back(static_cast<int>(l_currentResultsOCWOnly.size() - l_currentGoalOCWOnly.size()));
    }


    // mean results
    meanDiffSizeOCW = 0.0;
    meanAbsoluteCorrectPositionAndWordCCW = 0.0;
    meanCorrectPositionAndWordCCW = 0.0;

    diffSizeOCW.clear();
    absoluteCorrectPositionAndWordCCW.clear();
    correctPositionAndWordCCW.clear();

    std::ofstream l_flowResFile(pathSaveAllSentenceRest);

    if(!l_flowResFile)
    {
        std::cerr << "Error log file. " << std::endl;
    }

    // compute the percentages
    for(int ii = 0; ii < l_goalCCWOnly.size(); ++ii)
    {
        Sentence l_currenGoalCCWOnly = l_goalCCWOnly[ii];
        Sentence l_currenResCCWOnly  = l_resultsCCWOnly[ii];

        int l_correctPositionAndWord = 0;
        int l_nbWordsGoal   = static_cast<int>(l_currenGoalCCWOnly.size());
        int l_nbWordsResult = static_cast<int>(l_currenResCCWOnly.size());

        for(int jj = 0; jj < std::min(l_nbWordsGoal, l_nbWordsResult); ++jj)
        {
            if(l_currenGoalCCWOnly[jj] == l_currenResCCWOnly[jj])
            {
                ++l_correctPositionAndWord;
            }
        }

        double l_percentCorrectPositionAndWord = 100.0 * l_correctPositionAndWord / std::max(l_nbWordsGoal,l_nbWordsResult);

        double l_percentAbsolute = 0.0;
        if(l_percentCorrectPositionAndWord == 100)
        {
            l_percentAbsolute = 100.0;
        }

        meanAbsoluteCorrectPositionAndWordCCW += l_percentAbsolute;
        meanCorrectPositionAndWordCCW         += l_percentCorrectPositionAndWord;
        meanDiffSizeOCW                       += sqrt(static_cast<double>(l_diffSizesOCW[ii]*l_diffSizesOCW[ii]));


        std::ostringstream l_os1,l_os2,l_os3;
        l_os1 << l_percentCorrectPositionAndWord;
        l_os2 << l_percentAbsolute;
        l_os3 << l_diffSizesOCW[ii];
        l_flowResFile << l_os1.str() << " " << l_os2.str() << " " << l_os3.str() << std::endl;


        diffSizeOCW.push_back(l_percentCorrectPositionAndWord);
        absoluteCorrectPositionAndWordCCW.push_back(l_percentAbsolute);
        correctPositionAndWordCCW.push_back(l_diffSizesOCW[ii]);

        std::cout.precision(5);
        if(m_verbose)
        {
            std::cout << "|CCW absolute: " << l_percentAbsolute << "% | CCW : " << l_percentCorrectPositionAndWord << "% | OCW diff size : " << l_diffSizesOCW[ii] << std::endl;
        }
    }


    // update mean res values
        meanDiffSizeOCW /= l_goalCCWOnly.size();
        meanCorrectPositionAndWordCCW /= l_goalCCWOnly.size();
        meanAbsoluteCorrectPositionAndWordCCW /= l_goalCCWOnly.size();
}



void ModelObject::compareResults(cbool trainResults,std::vector<double> &correctPositionAndWordPercentage,
                                              std::vector<double> &sentenceRightAbsolutePercentage,
                                              std::vector<double> &sizeDifferencePercentage,
                                              int &totalWordNumber, int &totalWordCorrectNumber)
{
    if(m_verbose)
    {
        std::cout << "Start analysing results. " << std::endl;
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
        }
    }
    else
    {
        l_goal = m_trainSentence;
        l_results = m_recoveredSentencesTrain;
    }

    correctPositionAndWordPercentage.clear();
    sentenceRightAbsolutePercentage.clear();
    sizeDifferencePercentage.clear();
    totalWordNumber        = 0;
    totalWordCorrectNumber = 0;

    for(int ii = 0; ii < l_goal.size(); ++ii)
    {
        int l_nbWordsGoal   = static_cast<int>(l_goal[ii].size());
        int l_nbWordsResult = static_cast<int>(l_results[ii].size());
        totalWordNumber += l_nbWordsGoal;

        int l_correctPositionAndWord = 0;
        int l_wordsNotAllocated = 0;
        int l_sizeDifference = (l_nbWordsGoal - l_nbWordsResult);
        if(l_sizeDifference < 0)
        {
            l_sizeDifference *= -1;
        }

        for(int jj = 0; jj < std::min(l_goal[ii].size(), l_results[ii].size()); ++jj)
        {
            if(l_goal[ii][jj] == l_results[ii][jj])
            {
                ++l_correctPositionAndWord;
            }
        }
        totalWordCorrectNumber += l_correctPositionAndWord;

        for(int jj = 0; jj < l_results[ii].size(); ++jj)
        {
            if( l_results[ii][jj] == "X")
            {
                ++l_wordsNotAllocated;
            }
        }

        double l_percentCorrectPositionAndWord = 100.0 * l_correctPositionAndWord / l_nbWordsGoal;
        double l_percentNotAllocated = 100.0 * l_wordsNotAllocated / l_nbWordsGoal;
        double l_percentSizeDifference = 100.0 * l_sizeDifference / l_nbWordsGoal;
        double l_percentAbsolute = 0.0;
        if(l_percentCorrectPositionAndWord == 100)
        {
            l_percentAbsolute = 100.0;
        }

        std::cout.precision(5);
        if(m_verbose)
        {

            std::cout << "|CPW : " << l_percentCorrectPositionAndWord << "% | Abs : " << l_percentAbsolute << "% | SD : " << l_percentSizeDifference << "% | -> ";

            if(trainResults)
            {
                for(int jj = 0; jj < m_recoveredSentencesTrain[ii].size(); ++jj)
                {
                    std::cout << m_recoveredSentencesTrain[ii][jj] << " ";
                }
            }
            else
            {
                for(int jj = 0; jj < m_recoveredSentencesTest[ii].size(); ++jj)
                {
                    std::cout << m_recoveredSentencesTest[ii][jj] << " ";
                }
            }
            std::cout << std::endl;
        }

        correctPositionAndWordPercentage.push_back(l_percentCorrectPositionAndWord);
        sentenceRightAbsolutePercentage.push_back(l_percentAbsolute);
        sizeDifferencePercentage.push_back(l_percentSizeDifference);

    }

    if(m_verbose)
    {
        std::cout << "End analysing results. " << std::endl;
    }
}




// ######################################## TESTS FLOAT


void ModelObject::launchTrainingF()
{
    // init time
        clock_t l_trainingTime = clock();
        m_trainingSuccess = false;
        m_3DMatSentencesOutputTrain = cv::Mat();

    // generate close class word arrays
        if(m_grammar.size() > 0)
        {
            m_closedClassWords = m_grammar;
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
        displayTime("Generate stim files with Python ", l_trainingTime, false, m_verbose);
            system(l_pythonCall.c_str());
        displayTime("End generation ", l_trainingTime, true, m_verbose);


    // init matrices
        cv::Mat l_3DMatStimMeanTrain, l_3DMatStimSentTrain, l_internalStatesTrain;
        std::vector<cv::Mat> l_3DVMatStimMeanTrain, l_3DVMatStimSentTrain, l_internalStatesTrainV; // TEST


    // load input matrices created in the python script)
        load3DMatrixFromNpPythonSaveTextF(QString("../data/input/stim_mean_train.txt"), l_3DMatStimMeanTrain);
        load3DMatrixFromNpPythonSaveTextF(QString("../data/input/stim_sent_train.txt"), l_3DMatStimSentTrain);
        load3DMatrixFromNpPythonSaveTextF(QString("../data/input/stim_mean_train.txt"), l_3DVMatStimMeanTrain);
        load3DMatrixFromNpPythonSaveTextF(QString("../data/input/stim_sent_train.txt"), l_3DVMatStimSentTrain);

    // train reservoir
        displayTime("Start reservoir training ", l_trainingTime, false, m_verbose);
            m_reservoir.train(l_3DMatStimMeanTrain, l_3DMatStimSentTrain, m_3DMatSentencesOutputTrain, l_internalStatesTrain);
        displayTime("End reservoir training ", l_trainingTime, true, m_verbose);

    // save matrices
        cv::Mat l_subRes(m_3DMatSentencesOutputTrain.size[1], m_3DMatSentencesOutputTrain.size[0]*m_3DMatSentencesOutputTrain.size[2], CV_32FC3);

            for(int ii = 0; ii < m_3DMatSentencesOutputTrain.size[0]; ++ii)
            {
                for(int jj = 0; jj < m_3DMatSentencesOutputTrain.size[1]; ++jj)
                {
                    for(int kk = 0; kk < m_3DMatSentencesOutputTrain.size[2]; ++kk)
                    {
                        cv::Vec3f l_value;

                        if(ii % 3 == 0)
                        {
                            l_value = cv::Vec3f(m_3DMatSentencesOutputTrain.at<float>(ii,jj,kk),0,m_3DMatSentencesOutputTrain.at<float>(ii,jj,kk));
                        }
                        else if(ii % 3 == 1)
                        {
                            l_value = cv::Vec3f(0,m_3DMatSentencesOutputTrain.at<float>(ii,jj,kk),0);
                        }
                        else
                        {
                            l_value = cv::Vec3f(m_3DMatSentencesOutputTrain.at<float>(ii,jj,kk),0,0);
                        }


                        l_subRes.at<cv::Vec3f>(jj, kk + ii*m_3DMatSentencesOutputTrain.size[2]) = l_value;
                    }
                }
            }

        std::ostringstream l_os1;
        l_os1 << s_numImage;


        double min, max;
        cv::minMaxLoc(l_subRes, &min, &max);
        save3Channel2DMatrixToTextStd("../data/Results/mat_out/img_before_" + l_os1.str() + ".txt", l_subRes);


        for(int ii = 0; ii < l_subRes.rows * l_subRes.cols; ++ii)
        {
//            if(l_subRes.at<cv::Vec3f>(ii) != cv::Vec3f(0,0,0))
            {
                l_subRes.at<cv::Vec3f>(ii) += cv::Vec3f(-min,-min,-min);
            }
//            else
            {
//                l_subRes.at<cv::Vec3f>(ii) = cv::Vec3f(0,0,0);
            }

            cv::Vec3f l_value = l_subRes.at<cv::Vec3f>(ii);
            l_value[0] *= 255./max;
            l_value[1] *= 255./max;
            l_value[2] *= 255./max;

            l_subRes.at<cv::Vec3f>(ii) = l_value;
        }




        save3Channel2DMatrixToTextStd("../data/Results/mat_out/img_color_" + l_os1.str() + ".txt", l_subRes);

        cv::imwrite("../data/Results/mat_out/sentence_output_color.png_" + l_os1.str() + ".png", l_subRes);
        cv::cvtColor(l_subRes, l_subRes, CV_BGR2GRAY );
        cv::imwrite("../data/Results/mat_out/sentence_output_gray.png_" + l_os1.str() + ".png", l_subRes);

        ++s_numImage;


    // save data for creating curves

        cv::Mat l_curve(m_3DMatSentencesOutputTrain.size[0]*m_3DMatSentencesOutputTrain.size[1], m_3DMatSentencesOutputTrain.size[2]+2, CV_32FC1);

        for(int ii = 0; ii < m_3DMatSentencesOutputTrain.size[0]; ++ii)
        {
            for(int jj = 0; jj < m_3DMatSentencesOutputTrain.size[1]; ++jj)
            {
                l_curve.at<float>(ii*m_3DMatSentencesOutputTrain.size[1] + jj, 1) = jj;

                for(int kk = 0; kk < m_3DMatSentencesOutputTrain.size[2]; ++kk)
                {
                    l_curve.at<float>(ii*m_3DMatSentencesOutputTrain.size[1] + jj,kk+2) = m_3DMatSentencesOutputTrain.at<float>(ii,jj,kk);
                }
            }
        }

        for(int ii = 0; ii < l_curve.rows; ++ii)
        {
            l_curve.at<float>(ii,0) = ii / m_3DMatSentencesOutputTrain.size[1];
        }

        save2DMatrixToTextStd("../data/Results/mat_out/curve_" +  l_os1.str() + ".txt", l_curve);


//        cv::imshow("reservoir_display",l_subRes);
//        cv::waitKey(5000);


//        cvtColor( l_subRes, l_subRes, CV_GRAY2RGB);
//        cv::imshow("reservoir_display",l_subRes);
//        cv::waitKey(5000);





    // retrieve corpus train data
        QVector<QStringList> l_trainMeaning,l_trainInfo,l_trainSentence, l_inused;
        extractAllDataFromCorpusFile(m_parameters.m_corpusFilePath.c_str(), l_trainMeaning,l_trainInfo,l_trainSentence, l_inused,l_inused,l_inused);
        convQt2DString2Std2DString(l_trainMeaning, m_trainMeaning);
        convQt2DString2Std2DString(l_trainInfo, m_trainInfo);
        convQt2DString2Std2DString(l_trainSentence, m_trainSentence);

        m_trainingSuccess = true;
}


void ModelObject::launchTestsF(const std::string &corpusTestFilePath)
{
    // init time
        clock_t l_testTime = clock();
        m_3DMatSentencesOutputTest = cv::Mat();

    // check training
        if(!m_trainingSuccess)
        {
            std::cerr << "The training must be done before the tests. " << std::endl;
            return;
        }

    // check corpus path input
        std::string l_corpusFilePath = m_parameters.m_corpusFilePath;
        if(corpusTestFilePath.size() > 0)
        {
            l_corpusFilePath = corpusTestFilePath;
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

        displayTime("Generate stim files with Python ", l_testTime, false, m_verbose);
            system(l_pythonCall.c_str());
        displayTime("End generation ", l_testTime, true, m_verbose);

    // init matrices
        cv::Mat l_3DMatStimMeanTest, l_internalStatesTest;

    // load input matrices created in the python script)
        load3DMatrixFromNpPythonSaveTextF(QString("../data/input/stim_mean_test.txt"),   l_3DMatStimMeanTest);

    // test reservoir
        displayTime("Start reservoir testing ", l_testTime, false, m_verbose);
            m_reservoir.test(l_3DMatStimMeanTest, m_3DMatSentencesOutputTest, l_internalStatesTest);
        displayTime("End reservoir testing ", l_testTime, true, m_verbose);

    // retrieve corpus test data
        QVector<QStringList> l_testMeaning,l_testInfo, l_inused;
        extractAllDataFromCorpusFile(l_corpusFilePath.c_str(), l_inused,l_inused,l_inused, l_testMeaning,l_testInfo,l_inused);
        convQt2DString2Std2DString(l_testMeaning, m_testMeaning);
        convQt2DString2Std2DString(l_testInfo, m_testInfo);
}
