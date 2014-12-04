
/**
 * \file Generalization.cpp
 * \brief defines Generalization
 * \author Florian Lance
 * \date 01/10/14
 */

#include "Generalization.h"

Generalization::Generalization(Model &model) : m_model(&model)
{}



void Generalization::retrieveRandomSentenceList(cint sizeCorpus, cint nbSentence, std::vector<int> &randomSentenceList)
{
    randomSentenceList.clear();

    while(randomSentenceList.size() < nbSentence)
    {
        int l_id = rand()%sizeCorpus;

        bool l_adSentence = true;

        for(int ii = 0; ii < randomSentenceList.size(); ++ii)
        {
            if(randomSentenceList[ii] == l_id)
            {
                l_adSentence = false;
                break;
            }
        }

        if(l_adSentence)
        {
            randomSentenceList.push_back(l_id);
        }
    }
}

void Generalization::retrieveSubSentenceCorpusRandomized(std::vector<int> &randomSentenceList, QVector<QStringList> &subMeaning, QVector<QStringList> &subInfo, QVector<QStringList>& subSentence)
{
//    for(int ii = 0; ii < randomSentenceList.size(); ++ii)
//    {
//        QStringList l_randomSubSentence = m_trainSentence[randomSentenceList[ii]];
//        std::random_shuffle(l_randomSubSentence.begin(), l_randomSubSentence.end());

//        subMeaning.push_back(m_trainMeaning[randomSentenceList[ii]]);
//        subInfo.push_back(m_trainInfo[randomSentenceList[ii]]);
//        subSentence.push_back(l_randomSubSentence);
//    }

//    for(int ii = 0; ii < m_trainSentence.size(); ++ii)
//    {
//        bool l_addSentence = true;

//        for(int jj = 0; jj < randomSentenceList.size(); ++jj)
//        {
//            if(randomSentenceList[jj] == ii)
//            {
//                l_addSentence = false;
//                break;
//            }
//        }

//        if(l_addSentence)
//        {
//            subMeaning.push_back(m_trainMeaning[ii]);
//            subInfo.push_back(m_trainInfo[ii]);
//            subSentence.push_back(m_trainSentence[ii]);
//        }
//    }

    for(int ii = 0; ii < m_trainSentence.size(); ++ii)
    {
        bool l_addSentence = true;

        for(int jj = 0; jj < randomSentenceList.size(); ++jj)
        {
            if(randomSentenceList[jj] == ii)
            {
                l_addSentence = false;

                QStringList l_randomSubSentence = m_trainSentence[randomSentenceList[jj]];
                std::random_shuffle(l_randomSubSentence.begin(), l_randomSubSentence.end());
                subMeaning.push_back(m_trainMeaning[randomSentenceList[jj]]);
                subInfo.push_back(m_trainInfo[randomSentenceList[jj]]);
                subSentence.push_back(l_randomSubSentence);

                break;
            }
        }


        if(l_addSentence)
        {
            subMeaning.push_back(m_trainMeaning[ii]);
            subInfo.push_back(m_trainInfo[ii]);
            subSentence.push_back(m_trainSentence[ii]);
        }
    }

}

void Generalization::retrieveSubMeaningCorpusRandomized(std::vector<int> &randomSentenceList, QVector<QStringList> &subMeaning, QVector<QStringList> &subInfo, QVector<QStringList>& subSentence)
{
    for(int ii = 0; ii < randomSentenceList.size(); ++ii)
    {
        QStringList l_randomSubMeaning = m_trainMeaning[randomSentenceList[ii]];

        // detect "," string
            int l_comaId = 0; // no coma in the sentence
            for(int jj = 0; jj < l_randomSubMeaning.size(); ++jj)
            {
                if(l_randomSubMeaning[jj] == ",")
                {
                    l_comaId = jj;
                }
            }

        // split sequence
            if(l_comaId != 0)
            {
                QStringList  l_part1;
                QStringList  l_part2;
                for(int jj = 0; jj < l_randomSubMeaning.size(); ++jj)
                {
                    if(jj < l_comaId)
                    {
                        l_part1 << l_randomSubMeaning[jj];
                    }
                    else if(jj > l_comaId)
                    {
                        l_part2 << l_randomSubMeaning[jj];
                    }
                }

            // shuffle list
                std::random_shuffle(l_part1.begin(), l_part1.end());
                std::random_shuffle(l_part2.begin(), l_part2.end());
                l_randomSubMeaning.clear();
                l_randomSubMeaning << l_part1 << "," << l_part2;
            }
            else
            {
                std::random_shuffle(l_randomSubMeaning.begin(), l_randomSubMeaning.end());
            }

        subMeaning.push_back(l_randomSubMeaning);
        subInfo.push_back(m_trainInfo[randomSentenceList[ii]]);
        subSentence.push_back(m_trainSentence[randomSentenceList[ii]]);
    }

    for(int ii = 0; ii < m_trainMeaning.size(); ++ii)
    {
        bool l_addSentence = true;

        for(int jj = 0; jj < randomSentenceList.size(); ++jj)
        {
            if(randomSentenceList[jj] == ii)
            {
                l_addSentence = false;
                break;
            }
        }

        if(l_addSentence)
        {
            subMeaning.push_back(m_trainMeaning[ii]);
            subInfo.push_back(m_trainInfo[ii]);
            subSentence.push_back(m_trainSentence[ii]);
        }
    }

}

void Generalization::randomChangeCorpusGeneralization(cint numberRandomSentences, const QString pathRandomCorpus, cbool randomMeaning)
{
    QVector<QStringList> l_inused;

    // extract data
        m_trainMeaning.clear(); m_trainInfo.clear(); m_trainSentence.clear();
        extractAllDataFromCorpusFile(m_model->parameters().m_corpusFilePath.c_str(), m_trainMeaning,m_trainInfo,m_trainSentence, l_inused,l_inused,l_inused);

    // create random sentence list
        int l_sizeCorpus = m_trainMeaning.size();
        std::vector<int> l_randomIdSentences;
        retrieveRandomSentenceList(l_sizeCorpus, numberRandomSentences, l_randomIdSentences);

        QVector<QStringList> l_subData, l_subInfo, l_subMeaning;
        if(randomMeaning)
        {
            retrieveSubMeaningCorpusRandomized(l_randomIdSentences, l_subData, l_subInfo, l_subMeaning);
        }
        else
        {
            retrieveSubSentenceCorpusRandomized(l_randomIdSentences, l_subData, l_subInfo, l_subMeaning);
        }

        generateCorpus(pathRandomCorpus, l_subData, l_subInfo, l_subMeaning, l_inused, l_inused);
}

void Generalization::startXVerification(const std::string &xCheckTrainPath, const std::string &xCheckTestPath, const std::string &xCheckTestSentencesPath)
{
    QVector<QStringList> l_inused;
    extractAllDataFromCorpusFile(m_model->parameters().m_corpusFilePath.c_str(), m_trainMeaning,m_trainInfo,m_trainSentence, l_inused,l_inused,l_inused);

    ModelParameters l_currentParameters  = m_model->parameters();
    l_currentParameters.m_corpusFilePath = "../data/input/Corpus/XCheck.txt";

    l_currentParameters.display();    

    std::ofstream l_flowXCheckTrain(xCheckTrainPath);
    std::ofstream l_flowXCheckTest(xCheckTestPath);
    std::ofstream l_flowXCheckTestSentences(xCheckTestSentencesPath);

    l_flowXCheckTrain << "\nRES 1 : CCW correct position and word ABSOLUTE (0% or 100%) -> ex : goal : the , the  that -s -ed it | res : that, the the -s -ed it\n";
    l_flowXCheckTrain << "RES 2 : CCW correct position and word (between 0% and 100%) \n";
    l_flowXCheckTrain << "RES 3 : ALL correct position and word ABSOLUTE (0% or 100%) -> ex : goal : the X , the X that X -s X -ed it | res : the X X X, the the that X -s X X -ed it \n";
    l_flowXCheckTrain << "RES 4 : ALL correct position and word (between 0% and 100%) \n";
    l_flowXCheckTrain << "\n CORPUS ID | NEURONS | LEAK RATE | SPARCITY | INPUT SCALING |  RIDGE  | SPECTRAL RADIUS |   TIME   |   RES 1   |   RES 2   |   RES 3   |   RES 4   |\n";

    l_flowXCheckTest << "\nRES 1 : CCW correct position and word ABSOLUTE (0% or 100%) -> ex : goal : the , the  that -s -ed it | res : that, the the -s -ed it\n";
    l_flowXCheckTest << "RES 2 : CCW correct position and word (between 0% and 100%) \n";
    l_flowXCheckTest << "RES 3 : ALL correct position and word ABSOLUTE (0% or 100%) -> ex : goal : the X , the X that X -s X -ed it | res : the X X X, the the that X -s X X -ed it \n";
    l_flowXCheckTest << "RES 4 : ALL correct position and word (between 0% and 100%) \n";
    l_flowXCheckTest  << "\n CORPUS ID | NEURONS | LEAK RATE | SPARCITY | INPUT SCALING |  RIDGE  | SPECTRAL RADIUS |   TIME   |   RES 1   |   RES 2   |   RES 3   |   RES 4   |\n";

    int l_nbCharParams[] = {11,9,11,10,15,9,17,10,11,11,11,11};

    int aa = 0; // corpus number

    for(int ii = 0; ii < m_trainMeaning.size(); ++ii)
    {
        m_model->m_testSentence.clear(); // temp

        // creation of the new corpus
            QVector<QStringList> l_trainDataM, l_trainInfoM, l_trainMeaningM, l_testData, l_testInfo, l_testMeaning;
            for(int jj = 0; jj < m_trainMeaning.size(); ++jj)
            {
                if(ii != jj)
                {
                    l_trainDataM.push_back(m_trainMeaning[jj]);
                    l_trainInfoM.push_back(m_trainInfo[jj]);
                    l_trainMeaningM.push_back(m_trainSentence[jj]);
                }
                else
                {
                    l_testData.push_back(m_trainMeaning[jj]);
                    l_testInfo.push_back(m_trainInfo[jj]);

                    Sentence l_testMeaning;
                    for(int kk = 0;kk < m_trainSentence[jj].size(); ++kk)
                    {
                        l_testMeaning.push_back(m_trainSentence[jj][kk].toStdString());
                    }
                    m_model->m_testSentence.push_back(l_testMeaning); // temp
                }
            }

            generateCorpus(l_currentParameters.m_corpusFilePath.c_str(), l_trainDataM, l_trainInfoM, l_trainMeaningM, l_testData, l_testInfo);

            m_model->resetModelParameters(l_currentParameters, false);

            clock_t l_timeTraining = clock();

            m_model->launchTraining();

            double l_time = static_cast<double>((clock() - l_timeTraining)) / CLOCKS_PER_SEC;
            m_model->retrieveTrainSentences();


            std::vector<double> l_diffSizeOCW, l_absoluteCorrectPositionAndWordCCW, l_correctPositionAndWordCCW, l_absoluteCorrectPositionAndWordAll, l_correctPositionAndWordAll;
            double l_meanDiffSizeOCW, l_meanCorrectPositionAndWordCCW, l_meanAbsoluteCorrectPositionAndWordCCW, l_meanCorrectPositionAndWordAll, l_meanAbsoluteCorrectPositionAndWordAll;

            m_model->computeResultsData(true,"../data/Results/generalization_train.txt", l_diffSizeOCW,
                                        l_absoluteCorrectPositionAndWordCCW, l_correctPositionAndWordCCW,
                                        l_absoluteCorrectPositionAndWordAll, l_correctPositionAndWordAll,
                                        l_meanDiffSizeOCW,
                                        l_meanAbsoluteCorrectPositionAndWordCCW, l_meanCorrectPositionAndWordCCW,
                                        l_meanAbsoluteCorrectPositionAndWordAll, l_meanCorrectPositionAndWordAll
                                        );

            double l_res1 = l_meanAbsoluteCorrectPositionAndWordCCW, l_res2 = l_meanCorrectPositionAndWordCCW;
            double l_res3 = l_meanAbsoluteCorrectPositionAndWordAll, l_res4 = l_meanCorrectPositionAndWordAll;

            // retrieve string values from parameters
            std::vector<std::string> l_parameters;
            {
                std::ostringstream l_os1,l_os2,l_os3,l_os4,l_os5,l_os6,l_os7,l_os8,l_os9,l_os10,l_os11, l_os12;//, l_os13;
                l_os4.precision(4);l_os8.precision(6),l_os9.precision(3); l_os10.precision(3); l_os11.precision(3),l_os12.precision(3); // l_os13.precision(3);
                l_os1 << aa; l_os2 << l_currentParameters.m_nbNeurons; l_os3 <<  l_currentParameters.m_leakRate;
                l_os4 << l_currentParameters.m_sparcity; l_os5 << l_currentParameters.m_inputScaling; l_os6 << l_currentParameters.m_ridge;
                l_os7 << l_currentParameters.m_spectralRadius; l_os8 << l_time; l_os9 << l_res1; l_os10 << l_res2; l_os11 << l_res3;
                l_os12 << l_res4;// l_os13 << l_res5;


                l_parameters.push_back(l_os1.str()); l_parameters.push_back(l_os2.str()); l_parameters.push_back(l_os3.str()); l_parameters.push_back(l_os4.str());
                l_parameters.push_back(l_os5.str()); l_parameters.push_back(l_os6.str()); l_parameters.push_back(l_os7.str()); l_parameters.push_back(l_os8.str());
                l_parameters.push_back(l_os9.str()); l_parameters.push_back(l_os10.str()); l_parameters.push_back(l_os11.str());
                l_parameters.push_back(l_os12.str()); //l_parameters.push_back(l_os13.str());
            }


            // read readable data
                int l_nbSpaces,l_nbDivSpaces1,l_nbDivSpaces2;
                std::string l_spaces;

                for(int oo = 0; oo < l_parameters.size(); ++oo)
                {
                    l_nbSpaces = l_nbCharParams[oo] - static_cast<int>(l_parameters[oo].size());
                    l_nbDivSpaces1 = l_nbSpaces/2;
                    l_nbDivSpaces2 = l_nbSpaces/2 + l_nbSpaces%2;
                    l_spaces.append(l_nbDivSpaces1, ' ');
                    l_flowXCheckTrain << l_spaces; l_spaces.clear();
                    l_flowXCheckTrain << l_parameters[oo];
                    l_spaces.append(l_nbDivSpaces2, ' ');
                    l_flowXCheckTrain << l_spaces << "|"; l_spaces.clear();
                }

                l_flowXCheckTrain << std::endl;

            // create test stats
                m_model->launchTests();
                m_model->retrieveTestsSentences();

                m_model->computeResultsData(false,"../data/Results/generalization_test.txt", l_diffSizeOCW,
                                            l_absoluteCorrectPositionAndWordCCW, l_correctPositionAndWordCCW,
                                            l_absoluteCorrectPositionAndWordAll, l_correctPositionAndWordAll,
                                            l_meanDiffSizeOCW,
                                            l_meanAbsoluteCorrectPositionAndWordCCW, l_meanCorrectPositionAndWordCCW,
                                            l_meanAbsoluteCorrectPositionAndWordAll, l_meanCorrectPositionAndWordAll
                                            );

                l_res1 = l_meanAbsoluteCorrectPositionAndWordCCW; l_res2 = l_meanCorrectPositionAndWordCCW;
                l_res3 = l_meanAbsoluteCorrectPositionAndWordAll; l_res4 = l_meanCorrectPositionAndWordAll;

                l_parameters.clear();
                // retrieve string values from parameters
                {
                    std::ostringstream l_os1,l_os2,l_os3,l_os4,l_os5,l_os6,l_os7,l_os8,l_os9,l_os10,l_os11, l_os12;//, l_os13;
                    l_os4.precision(4);l_os8.precision(6),l_os9.precision(3); l_os10.precision(3); l_os11.precision(3),l_os12.precision(3); // l_os13.precision(3);
                    l_os1 << aa; l_os2 << l_currentParameters.m_nbNeurons; l_os3 <<  l_currentParameters.m_leakRate;
                    l_os4 << l_currentParameters.m_sparcity; l_os5 << l_currentParameters.m_inputScaling; l_os6 << l_currentParameters.m_ridge;
                    l_os7 << l_currentParameters.m_spectralRadius; l_os8 << l_time; l_os9 << l_res1; l_os10 << l_res2; l_os11 << l_res3;
                    l_os12 << l_res4;// l_os13 << l_res5;

                    l_parameters.push_back(l_os1.str()); l_parameters.push_back(l_os2.str()); l_parameters.push_back(l_os3.str()); l_parameters.push_back(l_os4.str());
                    l_parameters.push_back(l_os5.str()); l_parameters.push_back(l_os6.str()); l_parameters.push_back(l_os7.str()); l_parameters.push_back(l_os8.str());
                    l_parameters.push_back(l_os9.str()); l_parameters.push_back(l_os10.str()); l_parameters.push_back(l_os11.str());
                    l_parameters.push_back(l_os12.str()); //l_parameters.push_back(l_os13.str());
                }



                for(int oo = 0; oo < l_parameters.size(); ++oo)
                {
                    l_nbSpaces = l_nbCharParams[oo] - static_cast<int>(l_parameters[oo].size());
                    l_nbDivSpaces1 = l_nbSpaces/2;
                    l_nbDivSpaces2 = l_nbSpaces/2 + l_nbSpaces%2;
                    l_spaces.append(l_nbDivSpaces1, ' ');
                    l_flowXCheckTest << l_spaces; l_spaces.clear();
                    l_flowXCheckTest << l_parameters[oo];
                    l_spaces.append(l_nbDivSpaces2, ' ');
                    l_flowXCheckTest << l_spaces << "|"; l_spaces.clear();
                }

                l_flowXCheckTest << std::endl;


            for(int jj = 0; jj < m_model->m_recoveredSentencesTest.size(); ++jj)
            {
                for(int kk = 0; kk < m_model->m_recoveredSentencesTest[jj].size(); ++kk)
                {
                    l_flowXCheckTestSentences << m_model->m_recoveredSentencesTest[jj][kk] << " ";
                }
            }

            l_flowXCheckTestSentences << "\n ----> ";

            for(int jj = 0; jj < m_trainSentence[ii].size(); ++jj)
            {
                l_flowXCheckTestSentences << m_trainSentence[ii][jj].toStdString() << " ";
            }

            l_flowXCheckTestSentences << std::endl;

            m_model->displayResults(false,true);
    }
}


