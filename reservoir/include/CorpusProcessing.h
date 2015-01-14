
/**
 * \file CorpusProcessing.h
 * \brief defines useful function for manipuling a corpus.
 * \author Florian Lance
 * \date 01/10/14
 */

#ifndef CORPUSPROCESSING_H
#define CORPUSPROCESSING_H

#include "Utility.h"


/**
 * @brief return a default list of closed class words
 * @param [out] constructionWords : CCW list to be returned
 * @param [in]  endStringToAdd    : word to add at the end of the CCW list
 */
static void closedClassWords(Sentence &constructionWords, const std::string endStringToAdd = "")
{
    constructionWords.clear();
    std::string l_words[] ={"and","is","of","the","to",".","-ed","-ing","-s","by","it","that","was","did",",","from"};
    constructionWords = Sentence(l_words, l_words + sizeof(l_words) / sizeof(std::string));

    if(endStringToAdd.size() > 0)
    {
        constructionWords.push_back(endStringToAdd);
    }
}

/**
 * @brief Generate a corpus file with input data vectors
 */
static void generateCorpus(const QString pathFileNewCorpus, const QVector<QStringList> &trainData, const QVector<QStringList> &trainInfo, const QVector<QStringList> &trainMeaning,
                                                            const QVector<QStringList> &testData, const QVector<QStringList> &testInfo)
{
    QFile l_file(pathFileNewCorpus);
    if(l_file.open(QIODevice::WriteOnly | QIODevice::Text))
    {
        QTextStream out(&l_file);

        out << "<train data>\n";

        for(int ii = 0; ii < trainData.size(); ++ii)
        {
            QString l_line = trainData[ii].join(QString(" ")) + " <o> " +
                    trainInfo[ii].join(QString(" ")) + " <o>; " + trainMeaning[ii].join(QString(" "));
            out << l_line << "\n";
        }

        out << "</train data>\n<test data>\n";

        for(int ii = 0; ii < testData.size(); ++ii)
        {
            QString l_line = testData[ii].join(QString(" ")) + " <o> " + testInfo[ii].join(QString(" ")) + " <o> ";
            out << l_line << "\n";
        }

        out << "</test data>\n";
    }
    else
    {
        std::cerr << "Error generateCorpus : cannot open output corpus file. " << std::endl;
    }
}


/**
 * @brief Extract the CCW and the structure from a setting file.
 * ex : and s of the to . -ed -ing -s by it that was did , from -> CCW to be extracted
 *      P0 A1 O2 R3 -> structure to be extracted
 */
static void extractDataFromSettingFile(const QString pathFileSettings, QStringList &CCW, QStringList &structure)
{
    QFile l_file(pathFileSettings);

    if(l_file.open(QIODevice::ReadOnly | QIODevice::Text))
    {
        QTextStream in(&l_file), inLine;

        QString l_CCW       = in.readLine();
        QString l_Structure = in.readLine();

        CCW       = l_CCW.split(" ");
        structure = l_Structure.split(" ");
    }
    else
    {
        std::cerr << "Can not open settings file. " << std::endl;
    }
}

/**
 * @brief Parse a corpus file and save the data in the inputs vectors.
 */
static void extractAllDataFromCorpusFile(const QString pathFileCorpus,
                                         QVector<QStringList> &dataTrain, QVector<QStringList> &sendFormInfoTrain, QVector<QStringList> &meaningTrain,
                                         QVector<QStringList> &dataTest,  QVector<QStringList> &sendFormInfoTest,  QVector<QStringList> &meaningTest)
{
    bool l_trainData = false;
    bool l_testData = false;

    QFile l_file(pathFileCorpus);
    if(l_file.open(QIODevice::ReadOnly | QIODevice::Text))
    {
        QTextStream in(&l_file), inLine;

        while (!in.atEnd())
        {
            QString l_line = in.readLine();

            if(l_line.contains("<train data>"))
            {
                l_trainData = true;
                continue;
            }
            else if(l_line.contains("</train data>"))
            {
                l_trainData = false;
                continue;
            }
            else if(l_line.contains("<test data>"))
            {
                l_testData = true;
                continue;
            }
            else if(l_line.contains("</test data>"))
            {
                l_testData = false;
                continue;
            }

            inLine.setString(&l_line);

            int l_step = 0;
            QStringList l_data;
            QStringList l_sendFormInfo;
            QStringList l_meaning;

            while(!inLine.atEnd())
            {
                QString l_value;
                inLine >> l_value;

                if(l_value == "<o>")
                {
                    ++l_step;
                    continue;
                }

                if(l_value == "<o>;")
                {
                    ++l_step;
                    continue;
                }

                if(l_step == 0)
                {
                    l_data << l_value;
                }
                else if(l_step == 1)
                {
                    l_sendFormInfo << l_value;
                }
                else
                {
                    l_meaning << l_value;
                }

                if(l_value[0] == '\0')
                {
                    continue;
                }
            }

            if(l_trainData)
            {
                dataTrain.push_back(l_data);
                sendFormInfoTrain.push_back(l_sendFormInfo);
                meaningTrain.push_back(l_meaning);
            }

            if(l_testData)
            {
                dataTest.push_back(l_data);
                sendFormInfoTest.push_back(l_sendFormInfo);
                meaningTest.push_back(l_meaning);
            }
        }
    }
    else
    {
        std::cerr << "Can not open corpus file. " << std::endl;
    }
}

/**
 * @brief Parse a corpus file and save the data in the inputs vectors.
 */
static void extractAllDataFromCorpusFile(const QString pathFileCorpus, QVector<QStringList> &data, QVector<QStringList> &sendFormInfo, QVector<QStringList> &meaning)
{
    bool l_trainData = false;
    bool l_testData = false;

    QFile l_file(pathFileCorpus);
    if(l_file.open(QIODevice::ReadOnly | QIODevice::Text))
    {
        QTextStream in(&l_file), inLine;

        while (!in.atEnd())
        {
            QString l_line = in.readLine();

            if(l_line.contains("<train data>"))
            {
                l_trainData = true;
                continue;
            }
            else if(l_line.contains("</train data>"))
            {
                l_trainData = false;
                continue;
            }
            else if(l_line.contains("<test data>"))
            {
                l_testData = true;
                continue;
            }
            else if(l_line.contains("</test data>"))
            {
                l_testData = false;
                continue;
            }

            inLine.setString(&l_line);

            int l_step = 0;
            QStringList l_data;
            QStringList l_sendFormInfo;
            QStringList l_meaning;

            while(!inLine.atEnd())
            {
                QString l_value;
                inLine >> l_value;

                if(l_value == "<o>")
                {
                    ++l_step;
                    continue;
                }

                if(l_value == "<o>;")
                {
                    ++l_step;
                    continue;
                }

                if(l_step == 0)
                {
                    l_data << l_value;
                }
                else if(l_step == 1)
                {
                    l_sendFormInfo << l_value;
                }
                else
                {
                    l_meaning << l_value;
                }

                if(l_value[0] == '\0')
                {
                    continue;
                }
            }

            data.push_back(l_data);
            sendFormInfo.push_back(l_sendFormInfo);
            meaning.push_back(l_meaning);
        }
    }
    else
    {
        std::cerr << "Can not open corpus file. " << std::endl;
    }
}


/**
 * @brief Generate an OCW array with computed results and the list of words (OCW)
 */
static void generateOCWArray(const Sentences &dataArray, const Sentences &infoPAORArray, Sentences &OCWArray)
{
    for(int ii = 0; ii < infoPAORArray.size(); ++ii)
    {
        Sentence l_meaningSplit1, l_meaningSplit2, l_meaning = dataArray[ii], l_OCW;
        std::string l_info = infoPAORArray[ii][0];

        // split
        bool l_coma = false;
        for(int jj = 0; jj < l_meaning.size(); ++jj)
        {
            if(l_meaning[jj] == "," && !l_coma)
            {
                l_coma = true;
                continue;
            }

            if(!l_coma)
            {
                l_meaningSplit1.push_back(l_meaning[jj]);
            }
            else
            {
                l_meaningSplit2.push_back(l_meaning[jj]);
            }
        }

        int l_nbSeparators = 0;
        for(int jj = 0; jj < l_info.size(); ++jj)
        {
            if(l_info[jj] == '-')
            {
                ++l_nbSeparators;
            }
        }

        int l_nbCodeLetters;

        // case 1 : [_-_-_-_]
        if(l_meaningSplit2.size() == 0)
        {
            l_nbCodeLetters = l_nbSeparators + 1;

            for(int jj = 0; jj < l_nbCodeLetters; ++jj)
            {
                char l_codeLetter = l_info[1 + 2 * jj];

                if (l_codeLetter == 'P')
                {
                    l_OCW.push_back(l_meaningSplit1[0]);
                }
                else if(l_codeLetter == 'A')
                {
                    l_OCW.push_back(l_meaningSplit1[1]);
                }
                else if(l_codeLetter == 'O')
                {
                    l_OCW.push_back(l_meaningSplit1[2]);
                }
                else if(l_codeLetter == 'R')
                {
                    l_OCW.push_back(l_meaningSplit1[3]);
                }
                else if(l_codeLetter == 'Q') // TEST
                {
                    l_OCW.push_back(l_meaningSplit1[0]);
                }
            }
        }
        // case 2 : [_-_-_-_-_-_-_-_][_-_-_-_-_-_-_-_]
        else
        {
            l_nbCodeLetters = (l_nbSeparators + 2) /2 ;

            for(int jj = 0; jj < l_nbCodeLetters; ++jj)
            {
                char l_codeLetter1 = l_info[1 + 2 * jj];
                char l_codeLetter2 = l_info[(l_nbCodeLetters + l_nbSeparators/2 + 2) + (1 + 2 * jj)];


                if(l_codeLetter1 != '_')
                {
                    if (l_codeLetter1 == 'P')
                    {
                        l_OCW.push_back(l_meaningSplit1[0]);
                    }
                    else if(l_codeLetter1 == 'A')
                    {
                        l_OCW.push_back(l_meaningSplit1[1]);
                    }
                    else if(l_codeLetter1 == 'O')
                    {
                        l_OCW.push_back(l_meaningSplit1[2]);
                    }
                    else if(l_codeLetter1 == 'R')
                    {
                        l_OCW.push_back(l_meaningSplit1[3]);
                    }
                    else if(l_codeLetter1 == 'Q') // TEST
                    {
                        l_OCW.push_back(l_meaningSplit1[0]);
                    }
                }
                else
                {
                    if (l_codeLetter2 == 'P')
                    {
                        l_OCW.push_back(l_meaningSplit2[0]);
                    }
                    else if(l_codeLetter2 == 'A')
                    {
                        l_OCW.push_back(l_meaningSplit2[1]);
                    }
                    else if(l_codeLetter2 == 'O')
                    {
                        l_OCW.push_back(l_meaningSplit2[2]);
                    }
                    else if(l_codeLetter2 == 'R')
                    {
                        l_OCW.push_back(l_meaningSplit2[3]);
                    }
                    else if(l_codeLetter2 == 'Q') // TEST
                    {
                        l_OCW.push_back(l_meaningSplit2[0]);
                    }
                }
            }
        }

        OCWArray.push_back(l_OCW);
    }
}

template<typename T>
/**
 * @brief Convert the output activity of the reservoir in a signal.
 */
static void convertOutputActivityInSignalIdxMax(cv::Mat &outAct, cv::Mat &signalIndicesMax, const T thres = 0.4, const T eps = 1e-12)
{
    signalIndicesMax = cv::Mat();

    for(int ii = 0; ii < outAct.rows * outAct.cols; ++ii)
    {
        if(outAct.at<T>(ii) < thres)
        {
            outAct.at<T>(ii) = static_cast<T>(0);
        }
    }

    for(int ii = 0; ii < outAct.size[0]; ++ii)
    {
        int l_idx;

        cv::Mat l_currentRow = outAct.row(ii).clone();

        if(l_currentRow.depth() == CV_32FC1)
        {
            l_currentRow.convertTo(l_currentRow,CV_64FC1);
        }

        double l_max;
        cv::Point l_maxLocation;
        cv::minMaxLoc(l_currentRow, NULL, &l_max, NULL, &l_maxLocation);

        if(l_max < eps)
        {
            l_idx = -1;
        }
        else
        {
            l_idx = l_maxLocation.x;

            // remove max value
            l_currentRow.at<double>(l_idx) = -DBL_MAX;

            // check for the second max value
            double l_max2;
            cv::minMaxLoc(l_currentRow, NULL, &l_max2);

            if(l_max == l_max2) // there is at least 2 values that are equal to maximum
            {
                l_idx = -2;
            }
        }

        signalIndicesMax.push_back(l_idx);
    }
}

template<typename T>
/**
 * @brief convertOneOutputActivityInConstruction
 */
static void convertOneOutputActivityInConstruction(cv::Mat &outAct, const Sentence &constructionWords, Sentence &sent, cint minNbValUpperThres = 1)
{
    cv::Mat l_signalIndicesMax;
    convertOutputActivityInSignalIdxMax<T>(outAct, l_signalIndicesMax, static_cast<T>(0.4), static_cast<T>(1e-12));

    int l_previous = -1;
    int l_keepInMemory = -1;
    int l_nbOccurrenceSameIndex = 0;
    sent.clear();

    for(int ii = 0; ii < l_signalIndicesMax.rows; ++ii)
    {
        if(l_signalIndicesMax.at<int>(ii) != l_previous)
        {
            // if the new signal was the same that the one kept in memory
            if(l_signalIndicesMax.at<int>(ii) == l_keepInMemory)
            {
                // increment the counter
                ++l_nbOccurrenceSameIndex;
            }

            if(minNbValUpperThres - 1 - l_nbOccurrenceSameIndex > 0)
            {
                // keep the index in memory
                l_keepInMemory = l_signalIndicesMax.at<int>(ii);
            }
            else
            {
                // add the word corresponding to this index in the final sentence
                if(l_signalIndicesMax.at<int>(ii) != -1)
                {
                    sent.push_back(constructionWords[l_signalIndicesMax.at<int>(ii)]);
                }

                l_previous = l_signalIndicesMax.at<int>(ii);

                // reinit temp variables
                l_nbOccurrenceSameIndex = 0;
                l_keepInMemory = -1;
            }
        }
    }
}

/**
 * @brief convertLOutputActivityInConstruction
 */
static void convertLOutputActivityInConstruction(cv::Mat &outAct, const Sentence &constructionWords, Sentences &sent, cint minNbValUpperThres = 1)
{
    sent.clear();

    for(int ii = 0; ii < outAct.size[0]; ++ii)
    {
        Sentence l_sent;
        cv::Mat l_subOutAct;

        if(outAct.depth() == CV_32FC1)
        {
            l_subOutAct = cv::Mat(outAct.size[1], outAct.size[2], CV_32FC1);
            for(int jj = 0; jj < outAct.size[1]; ++jj)
            {
                for(int kk = 0; kk < outAct.size[2]; ++kk)
                {
                    l_subOutAct.at<float>(jj,kk) = outAct.at<float>(ii,jj,kk);
                }
            }

            convertOneOutputActivityInConstruction<float>(l_subOutAct, constructionWords, l_sent, minNbValUpperThres);
        }
        else
        {
            l_subOutAct = cv::Mat(outAct.size[1], outAct.size[2], CV_64FC1);

            for(int jj = 0; jj < outAct.size[1]; ++jj)
            {
                for(int kk = 0; kk < outAct.size[2]; ++kk)
                {
                    l_subOutAct.at<double>(jj,kk) = outAct.at<double>(ii,jj,kk);
                }
            }

            convertOneOutputActivityInConstruction<double>(l_subOutAct, constructionWords, l_sent, minNbValUpperThres);
        }


        sent.push_back(l_sent);
    }
}

/**
 * @brief Return the number of OCW in a construction sentence.
 */
static int nbOCWInConstruction(const Sentence &construction, const std::string _OCW)
{
    int l_nbOCW = 0;

    for(int ii = 0; ii < construction.size(); ++ii)
    {
        if(construction[ii] == _OCW)
        {
            ++l_nbOCW;
        }
    }

    return l_nbOCW;
}


/**
 * @brief Attribute openc class words to constructions.
 */
static void attributeOcwToConstructions(const Sentences &constructions, const Sentences &OCW,
                                           Sentences &sent, const std::string _OCW = "X")
{
    sent.clear();

    for(int ii = 0; ii < constructions.size(); ++ii)
    {
        Sentence l_sent;
        Sentence l_ocw = OCW[ii];

        int l_nbOCW = nbOCWInConstruction(constructions[ii], _OCW);

        if(!(l_nbOCW == l_ocw.size()))
        {
            int l_diff = l_nbOCW - static_cast<int>(l_ocw.size());

            if(l_diff > 0)
            {
                for(int jj = 0; jj < l_diff; ++jj)
                {
                    l_ocw.push_back("X");
                }
            }
        }

        int l_offset = 0;
        for(int jj = 0; jj < constructions[ii].size(); ++jj)
        {
            if(constructions[ii][jj] == _OCW)
            {
                l_sent.push_back(l_ocw[l_offset]);
                ++l_offset;
            }
            else
            {
                l_sent.push_back(constructions[ii][jj]);
            }
        }
        sent.push_back(l_sent);
    }
}

/**
 * @brief convertLineCorpusInfo
 */
static void convertLineCorpusInfo(const QStringList &data, const QStringList &infoTrain, const QStringList &trainMeaning, QString &newLineCorpusTrain)
{    
    // identify CCW in meaning and create a new train meaning string list
    QStringList l_newTrainMeaning;
    for(int ii = 0; ii < trainMeaning.size(); ++ii)
    {
        bool l_wordInsideData = false;
        for(int jj = 0; jj < data.size(); ++jj)
        {
            if(trainMeaning[ii] == data[jj])
            {
                l_wordInsideData = true;
                break;
            }
        }

        if(l_wordInsideData)
        {
            l_newTrainMeaning <<  trainMeaning[ii];
        }
    }

    // subdivide data between the 2 sentences
    bool l_sentence1 = true;
    QStringList l_data1, l_data2;
    for(int ii = 0; ii < data.size(); ++ii)
    {
        if(data[ii] == ",")
        {
            l_sentence1 = false;
            continue;
        }

        if(l_sentence1)
        {
            l_data1 << data[ii];
        }
        else
        {
            l_data2 << data[ii];
        }

    }

    QStringList l_PAOR, l_infoSentence1, l_infoSentence2;
    l_PAOR << "P" << "A" << "O" << "R";

    // find info PAOR
    for(int ii = 0; ii < l_newTrainMeaning.size(); ++ii)
    {
        int l_positionWordSentence1 = -1;
        int l_positionWordSentence2 = -1;

        for(int jj = 0; jj < l_data1.size(); ++jj)
        {
            if(l_data1[jj] == l_newTrainMeaning[ii])
            {
                l_positionWordSentence1 = jj;
                break;
            }
        }

        for(int jj = 0; jj < l_data2.size(); ++jj)
        {
            if(l_data2[jj] == l_newTrainMeaning[ii])
            {
                l_positionWordSentence2 = jj;
                break;
            }
        }

        if(l_positionWordSentence1 != -1)
        {
            l_infoSentence1 << l_PAOR[l_positionWordSentence1];
        }
        else
        {
            l_infoSentence1 << "_";
        }

        if(l_positionWordSentence2 != -1)
        {
            l_infoSentence2 << l_PAOR[l_positionWordSentence2];
        }
        else
        {
            l_infoSentence2 << "_";
        }
    }

    // build the info structure
    QString l_newInfoTrain1 = "[", l_newInfoTrain2 = "[";
    for(int ii = 0; ii < 9; ++ii)
    {
        if(ii < l_infoSentence1.size())
        {
            l_newInfoTrain1 += l_infoSentence1[ii] + "-";
        }
        else
        {
            l_newInfoTrain1 += "_-";
        }

        if(ii < l_infoSentence2.size())
        {
            l_newInfoTrain2 += l_infoSentence2[ii] + "-";
        }
        else
        {
            l_newInfoTrain2 += "_-";
        }
    }

    l_newInfoTrain1[l_newInfoTrain1.size()-1] = ']';
    l_newInfoTrain2[l_newInfoTrain2.size()-1] = ']';

    // create new corpus train line
    newLineCorpusTrain = data.join(QString(" ")) + " <o> " + l_newInfoTrain1 + l_newInfoTrain2 + " <o>; " + trainMeaning.join(QString(" "));
}

/**
 * @brief Convert an old formatted PAOR corpus to the new PAOR encoding.
 */
static void convertCorpus(QString pathCorpusOldFormat, QString pathCorpusNewFormat)
{
    QVector<QStringList> trainData, infoTrain, trainMeaning;
    extractAllDataFromCorpusFile(pathCorpusOldFormat, trainData, infoTrain, trainMeaning);


    QFile l_file(pathCorpusNewFormat);
    if(l_file.open(QIODevice::WriteOnly | QIODevice::Text))
    {
        QTextStream out(&l_file);

        out << "<train data>\n";

        for(int ii = 0; ii < trainData.size(); ++ii)
        {
            QString l_line;
            convertLineCorpusInfo(trainData[ii], infoTrain[ii], trainMeaning[ii], l_line);

            out << l_line << "\n";
        }

        out << "</train data>\n<test data>\n</test data>";
    }
}


/**
 * @brief generateSubRandomCorpus
 */
static void generateSubRandomCorpus(const QString pathFileCorpus, const QString pathFileNewCorpus, const int nbOfSentencesToKeepRandomly)
{
    QVector<QStringList> l_dataTrain, l_sendFormInfoTrain, l_meaningTrain;
    QVector<QStringList> l_dataTest, l_sendFormInfoTest, l_meaningTest;
    extractAllDataFromCorpusFile(pathFileCorpus, l_dataTrain, l_sendFormInfoTrain, l_meaningTrain, l_dataTest, l_sendFormInfoTest, l_meaningTest);

    if(nbOfSentencesToKeepRandomly > l_dataTrain.size())
    {
        std::cerr << "Error generateSubRandomCorpus : bad input nbOfSentencesToKeepRandomly. " << std::endl;
        return;
    }

    std::vector<int> l_sentencesId;

    while(l_sentencesId.size() < nbOfSentencesToKeepRandomly)
    {
        int l_index = rand()%l_dataTrain.size();

        // check if index alreay choosen
        bool l_addIndex = true;
        for(int ii = 0; ii < l_sentencesId.size(); ++ii)
        {
            if(l_sentencesId[ii] == l_index)
            {
                l_addIndex = false;
                break;
            }
        }

        if(l_addIndex)
        {
            l_sentencesId.push_back(l_index);
        }
    }


    QFile l_file(pathFileNewCorpus);
    if(l_file.open(QIODevice::WriteOnly | QIODevice::Text))
    {
        QTextStream out(&l_file);

        out << "<train data>\n";

        for(int ii = 0; ii < l_sentencesId.size(); ++ii)
        {
            QString l_line = l_dataTrain[l_sentencesId[ii]].join(QString(" ")) + " <o> " +
                    l_sendFormInfoTrain[l_sentencesId[ii]].join(QString(" ")) + " <o>; " + l_meaningTrain[l_sentencesId[ii]].join(QString(" "));
            out << l_line << "\n";
        }

        out << "</train data>\n<test data>\n";

        for(int ii = 0; ii < l_dataTest.size(); ++ii)
        {
            QString l_line = l_dataTest[ii].join(QString(" ")) + " <o> " + l_sendFormInfoTest[ii].join(QString(" ")) + " <o> " + l_meaningTest[ii].join(QString(" "));
            out << l_line << "\n";
        }

        out << "</test data>\n";
    }
    else
    {
        std::cerr << "Error generateSubRandomCorpus : cannot open output corpus file. " << std::endl;
    }
}

#endif
