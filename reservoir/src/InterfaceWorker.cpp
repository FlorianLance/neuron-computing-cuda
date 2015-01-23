


/**
 * \file InterfaceWorker.cpp
 * \brief Defines InterfaceWorker
 * \author Florian Lance
 * \date 01/12/14
 */

#include "InterfaceWorker.h"
#include "../moc/moc_InterfaceWorker.cpp"


InterfaceWorker::InterfaceWorker(QString absolutePath) : m_gridSearch(new GridSearch(m_model)), m_nbOfCorpus(0)
{
    qRegisterMetaType<ReplayParameters>("ReplayParameters");
    qRegisterMetaType<ReservoirParameters>("ReservoirParameters");
    qRegisterMetaType<LanguageParameters>("LanguageParameters");
    qRegisterMetaType<ModelParameters>("ModelParameters");
    qRegisterMetaType<ResultsDisplayReservoir>("ResultsDisplayReservoir");
    qRegisterMetaType<QVector<QVector<double> > >("QVector<QVector<double> >");
    qRegisterMetaType<QVector<int> >("QVector<int>");
    qRegisterMetaType<cv::Mat>("cv::Mat");
    qRegisterMetaType<Sentences >("Sentences");
}

InterfaceWorker::~InterfaceWorker()
{
    delete m_gridSearch;
}

GridSearch *InterfaceWorker::gridSearch() const
{
    return m_gridSearch;
}

Model *InterfaceWorker::model()
{
    return &m_model;
}

LanguageParameters InterfaceWorker::languageParameters() const
{
    return m_languageParameters;
}

void InterfaceWorker::addCorpus(QString corpusPath)
{
    m_corpusList.push_back(corpusPath);

    std::vector<std::string> l_stringListCorpus;

    for(int ii = 0; ii < m_corpusList.size(); ++ii)
    {
        l_stringListCorpus.push_back(m_corpusList[ii].toStdString());
    }

    m_gridSearch->setCorpusList(l_stringListCorpus);
    ++m_nbOfCorpus;
}

void InterfaceWorker::removeCorpus(int index)
{
    if(index == -1)
    {
        return;
    }

    m_corpusList.removeAt(index);

    std::vector<std::string> l_stringListCorpus;

    for(int ii = 0; ii < m_corpusList.size(); ++ii)
    {
        l_stringListCorpus.push_back(m_corpusList[ii].toStdString());
    }

    m_gridSearch->setCorpusList(l_stringListCorpus);
    --m_nbOfCorpus;
}

void InterfaceWorker::updateReservoirParameters(ReservoirParameters newParams)
{
    m_reservoirParameters = newParams;
}

void InterfaceWorker::updateLanguageParameters(LanguageParameters newParams)
{
    m_languageParameters = newParams;
}

void InterfaceWorker::updateReplayParameters(ReplayParameters newParams)
{
    m_replayParameters.m_randomNeurons          = newParams.m_randomNeurons;
    m_replayParameters.m_randomSentence         = newParams.m_randomSentence;
    m_replayParameters.m_randomNeuronsNumber    = newParams.m_randomNeuronsNumber;
    m_replayParameters.m_randomSentencesNumber  = newParams.m_randomSentencesNumber;
    m_replayParameters.m_rangeSentencesStart    = newParams.m_rangeSentencesStart;
    m_replayParameters.m_rangeSentencesEnd      = newParams.m_rangeSentencesEnd;
    m_replayParameters.m_rangeNeuronsStart      = newParams.m_rangeNeuronsStart;
    m_replayParameters.m_rangeNeuronsEnd        = newParams.m_rangeNeuronsEnd;
    m_replayParameters.m_useLastTraining        = newParams.m_useLastTraining;
}

void InterfaceWorker::start()
{
    if(m_nbOfCorpus <= 0)
    {
        QString l_message = "Cannot start, no corpus is defined. \n";
        emit sendLogInfo(l_message, QColor(Qt::red));
        std::cerr << l_message.toStdString() << std::endl;
        return;
    }

    lockInterfaceSignal(true);

    // define language parameters
        Sentence l_CCW, l_structure;
        QStringList l_CCWList = m_languageParameters.m_CCW.split(" ");
        for(int ii = 0; ii < l_CCWList.size(); ++ii)
        {
            l_CCW.push_back(l_CCWList[ii].toStdString());
        }
        QStringList l_structureList = m_languageParameters.m_structure.split(" ");
        for(int ii = 0; ii < l_structureList.size(); ++ii)
        {
            l_structure.push_back(l_structureList[ii].toStdString());
        }

        m_model.setCCWAndStructure(l_CCW, l_structure);

    // define all grid search parameters
        m_gridSearch->deleteParameterValues();
        m_gridSearch->setCudaParameters(m_reservoirParameters.m_useCuda, m_reservoirParameters.m_useCuda);

        bool l_operationValid;
        int l_OperationInvalid = 0;


        bool l_onlyStartValue = m_reservoirParameters.m_useOnlyStartValue;

        if(m_reservoirParameters.m_neuronsEnabled)
        {
            l_operationValid = m_gridSearch->setParameterValues(GridSearch::NEURONS_NB, m_reservoirParameters.m_neuronsStart, m_reservoirParameters.m_neuronsEnd,
                                                                m_reservoirParameters.m_neuronsOperation.toStdString(), l_onlyStartValue, m_reservoirParameters.m_neuronsNbOfUses);
            emit displayValidityOperationSignal(l_operationValid, GridSearch::NEURONS_NB);
            if(!l_operationValid)
            {
                ++l_OperationInvalid;
            }
        }
        if(m_reservoirParameters.m_leakRateEnabled)
        {
            l_operationValid = m_gridSearch->setParameterValues(GridSearch::LEAK_RATE,m_reservoirParameters.m_leakRateStart, m_reservoirParameters.m_leakRateEnd,
                                                                m_reservoirParameters.m_leakRateOperation.toStdString(), l_onlyStartValue, m_reservoirParameters.m_leakRateNbOfUses);
            emit displayValidityOperationSignal(l_operationValid, GridSearch::LEAK_RATE);
            if(!l_operationValid)
            {
                ++l_OperationInvalid;
            }
        }
        if(m_reservoirParameters.m_issEnabled)
        {
            l_operationValid = m_gridSearch->setParameterValues(GridSearch::INPUT_SCALING,m_reservoirParameters.m_issStart, m_reservoirParameters.m_issEnd,
                                                                m_reservoirParameters.m_issOperation.toStdString(), l_onlyStartValue, m_reservoirParameters.m_issNbOfUses);
            emit displayValidityOperationSignal(l_operationValid, GridSearch::INPUT_SCALING);
            if(!l_operationValid)
            {
                ++l_OperationInvalid;
            }
        }
        if(m_reservoirParameters.m_spectralRadiusEnabled)
        {
            l_operationValid = m_gridSearch->setParameterValues(GridSearch::SPECTRAL_RADIUS,    m_reservoirParameters.m_spectralRadiusStart,  m_reservoirParameters.m_spectralRadiusEnd,
                                                                m_reservoirParameters.m_spectralRadiusOperation.toStdString(), l_onlyStartValue, m_reservoirParameters.m_spectralRadiusNbOfUses);
            emit displayValidityOperationSignal(l_operationValid, GridSearch::SPECTRAL_RADIUS);
            if(!l_operationValid)
            {
                ++l_OperationInvalid;
            }
        }
        if(m_reservoirParameters.m_ridgeEnabled)
        {
            l_operationValid = m_gridSearch->setParameterValues(GridSearch::RIDGE,m_reservoirParameters.m_ridgeStart, m_reservoirParameters.m_ridgeEnd,
                                                                m_reservoirParameters.m_ridgeOperation.toStdString(), l_onlyStartValue, m_reservoirParameters.m_ridgeNbOfUses);
            emit displayValidityOperationSignal(l_operationValid, GridSearch::RIDGE);
            if(!l_operationValid)
            {
                ++l_OperationInvalid;
            }
        }
        if(m_reservoirParameters.m_sparcityEnabled)
        {
            l_operationValid = m_gridSearch->setParameterValues(GridSearch::SPARCITY, m_reservoirParameters.m_sparcityStart,  m_reservoirParameters.m_sparcityEnd,
                                                                m_reservoirParameters.m_sparcityOperation.toStdString(), l_onlyStartValue, m_reservoirParameters.m_sparcityNbOfUses);
            emit displayValidityOperationSignal(l_operationValid, GridSearch::SPARCITY);
            if(!l_operationValid)
            {
                ++l_OperationInvalid;
            }
        }

        if(l_OperationInvalid != 0)
        {
            QString l_message = "\nCannot start, " + QString::number(l_OperationInvalid) + " operation are invalid. (Displayed in red)\n";
            sendLogInfo(l_message, QColor(Qt::red));
            std::cerr << l_message.toStdString() << std::endl;
            lockInterfaceSignal(false);
            return;
        }

        bool l_doTrain, l_doTest;

        switch(m_reservoirParameters.m_action)
        {
            case TRAINING_RES :
                l_doTrain = true;
                l_doTest = false;
            break;
            case TEST_RES :
                l_doTrain = false;
                l_doTest = true;
            break;
            case BOTH_RES :
                l_doTrain = true;
                l_doTest = true;
            break;
        }

    QDateTime l_dateTime;
    l_dateTime = l_dateTime.currentDateTime();
    QDate l_date = l_dateTime.date();
    QTime l_time = QTime::currentTime();
    QString l_uniqueName = l_date.toString("dd_MM_yyyy") + "_" +  QString::number(l_time.hour()) + "h" + QString::number(l_time.minute()) + "m" + QString::number(l_time.second()) + "s.txt";
    QString l_pathRes    =  "../data/Results/res_"  + l_uniqueName;
    QString l_pathResRaw =  "../data/Results/raw/res_" + l_uniqueName;

    if(m_reservoirParameters.m_useLoadedTraining)
    {
        if(m_parametersTrainingLoaded.size() > 0)
        {
            m_gridSearch->setParameterValues(GridSearch::NEURONS_NB,      m_parametersTrainingLoaded[0].toInt(),    m_parametersTrainingLoaded[0].toInt(),   "+0", true, 1);
            m_gridSearch->setParameterValues(GridSearch::SPARCITY,        m_parametersTrainingLoaded[1].toDouble(), m_parametersTrainingLoaded[1].toDouble(),"+0", true, 1);
            m_gridSearch->setParameterValues(GridSearch::SPECTRAL_RADIUS, m_parametersTrainingLoaded[2].toDouble(), m_parametersTrainingLoaded[2].toDouble(),"+0", true, 1);
            m_gridSearch->setParameterValues(GridSearch::INPUT_SCALING,   m_parametersTrainingLoaded[3].toDouble(), m_parametersTrainingLoaded[3].toDouble(),"+0", true, 1);
            m_gridSearch->setParameterValues(GridSearch::LEAK_RATE,       m_parametersTrainingLoaded[4].toDouble(), m_parametersTrainingLoaded[4].toDouble(),"+0", true, 1);
            m_gridSearch->setParameterValues(GridSearch::RIDGE,           m_parametersTrainingLoaded[5].toDouble(), m_parametersTrainingLoaded[5].toDouble(),"+0", true, 1);
        }
    }

    if(m_reservoirParameters.m_useLoadedW && !m_reservoirParameters.m_useLoadedWIn)
    {
        if(m_parametersWLoaded.size() > 0)
        {
            m_gridSearch->setParameterValues(GridSearch::NEURONS_NB,      m_parametersWLoaded[0].toInt(),    m_parametersWLoaded[0].toInt(),   "+0", true, 1);
            m_gridSearch->setParameterValues(GridSearch::SPARCITY,        m_parametersWLoaded[1].toDouble(), m_parametersWLoaded[1].toDouble(),"+0", true, 1);
            m_gridSearch->setParameterValues(GridSearch::SPECTRAL_RADIUS, m_parametersWLoaded[2].toDouble(), m_parametersWLoaded[2].toDouble(),"+0", true, 1);
            m_gridSearch->setParameterValues(GridSearch::INPUT_SCALING,   m_parametersWLoaded[3].toDouble(), m_parametersWLoaded[3].toDouble(),"+0", true, 1);
            m_gridSearch->setParameterValues(GridSearch::LEAK_RATE,       m_parametersWLoaded[4].toDouble(), m_parametersWLoaded[4].toDouble(),"+0", true, 1);
            m_gridSearch->setParameterValues(GridSearch::RIDGE,           m_parametersWLoaded[5].toDouble(), m_parametersWLoaded[5].toDouble(),"+0", true, 1);
        }
    }
    else if(m_reservoirParameters.m_useLoadedWIn & !m_reservoirParameters.m_useLoadedW)
    {
        if(m_parametersWInLoaded.size() > 0)
        {
            m_gridSearch->setParameterValues(GridSearch::NEURONS_NB,      m_parametersWInLoaded[0].toInt(),    m_parametersWInLoaded[0].toInt(),   "+0", true, 1);
            m_gridSearch->setParameterValues(GridSearch::SPARCITY,        m_parametersWInLoaded[1].toDouble(), m_parametersWInLoaded[1].toDouble(),"+0", true, 1);
            m_gridSearch->setParameterValues(GridSearch::SPECTRAL_RADIUS, m_parametersWInLoaded[2].toDouble(), m_parametersWInLoaded[2].toDouble(),"+0", true, 1);
            m_gridSearch->setParameterValues(GridSearch::INPUT_SCALING,   m_parametersWInLoaded[3].toDouble(), m_parametersWInLoaded[3].toDouble(),"+0", true, 1);
            m_gridSearch->setParameterValues(GridSearch::LEAK_RATE,       m_parametersWInLoaded[4].toDouble(), m_parametersWInLoaded[4].toDouble(),"+0", true, 1);
            m_gridSearch->setParameterValues(GridSearch::RIDGE,           m_parametersWInLoaded[5].toDouble(), m_parametersWInLoaded[5].toDouble(),"+0", true, 1);
        }
    }
    else if(m_reservoirParameters.m_useLoadedW && m_reservoirParameters.m_useLoadedWIn)
    {
        bool l_paramValid = true;
        if(m_parametersWLoaded.size() == m_parametersWInLoaded.size())
        {
            for(int ii = 0; ii < m_parametersWLoaded.size(); ++ii)
            {
                if(m_parametersWLoaded[ii] != m_parametersWInLoaded[ii])
                {
                    l_paramValid = false;
                    break;
                }
            }
        }
        else
        {
            l_paramValid = false;
        }

        if(m_parametersWInLoaded.size() > 0 && l_paramValid)
        {
            m_gridSearch->setParameterValues(GridSearch::NEURONS_NB,      m_parametersWInLoaded[0].toInt(),    m_parametersWInLoaded[0].toInt(),   "+0", true, 1);
            m_gridSearch->setParameterValues(GridSearch::SPARCITY,        m_parametersWInLoaded[1].toDouble(), m_parametersWInLoaded[1].toDouble(),"+0", true, 1);
            m_gridSearch->setParameterValues(GridSearch::SPECTRAL_RADIUS, m_parametersWInLoaded[2].toDouble(), m_parametersWInLoaded[2].toDouble(),"+0", true, 1);
            m_gridSearch->setParameterValues(GridSearch::INPUT_SCALING,   m_parametersWInLoaded[3].toDouble(), m_parametersWInLoaded[3].toDouble(),"+0", true, 1);
            m_gridSearch->setParameterValues(GridSearch::LEAK_RATE,       m_parametersWInLoaded[4].toDouble(), m_parametersWInLoaded[4].toDouble(),"+0", true, 1);
            m_gridSearch->setParameterValues(GridSearch::RIDGE,           m_parametersWInLoaded[5].toDouble(), m_parametersWInLoaded[5].toDouble(),"+0", true, 1);
        }

        if(!l_paramValid)
        {
            QString l_message = "W and WIn doesn't have the same parameter file. \n";
            emit sendLogInfo(l_message, QColor(Qt::red));
            std::cerr << l_message.toStdString() << std::endl;
            lockInterfaceSignal(false);
            return;
        }
    }



    // launch reservoir computing
    m_gridSearch->launchTrainWithAllParameters(l_pathRes.toStdString(), l_pathResRaw.toStdString(), l_doTrain, l_doTest, m_reservoirParameters.m_useLoadedTraining,
                                                   m_reservoirParameters.m_useLoadedW, m_reservoirParameters.m_useLoadedWIn);
    lockInterfaceSignal(false);

    emit endTrainingSignal(true);
}

void InterfaceWorker::stop()
{
//    lockInterfaceSignal(false);
}

void InterfaceWorker::saveLastTraining(QString pathDirectory)
{
    if(pathDirectory.size() > 0)
    {
        m_model.saveTraining(pathDirectory.toStdString());
        sendLogInfo("Training saved in the directory : " + pathDirectory + "\n", QColor(0,0,255));
    }
}

void InterfaceWorker::saveLastReplay(QString pathDirectory)
{
    if(pathDirectory.size() > 0)
    {
        m_model.saveReplay(pathDirectory.toStdString());
        sendLogInfo("Replay saved in the directory : " + pathDirectory + "\n", QColor(0,0,255));
    }
}

void InterfaceWorker::loadTraining(QString pathDirectory)
{
    if(pathDirectory.size() > 0)
    {
        m_model.loadTraining(pathDirectory.toStdString());
        sendLogInfo("Training loaded in the directory : " + pathDirectory + "\n", QColor(Qt::blue));
    }
}

void InterfaceWorker::loadW(QString pathDirectory)
{
    if(pathDirectory.size() > 0)
    {
        m_model.loadW(pathDirectory.toStdString());
        sendLogInfo("W matrice loaded in the directory : " + pathDirectory + "\n", QColor(Qt::blue));
    }
}

void InterfaceWorker::loadWIn(QString pathDirectory)
{
    if(pathDirectory.size() > 0)
    {
        m_model.loadWIn(pathDirectory.toStdString());
        sendLogInfo("WIn matrice loaded in the directory : " + pathDirectory + "\n", QColor(Qt::blue));
    }
}

void InterfaceWorker::setLoadedTrainingParameters(QStringList loadedParams)
{
    m_parametersTrainingLoaded = loadedParams;
}

void InterfaceWorker::setLoadedWParameters(QStringList loadedParams)
{
    m_parametersWLoaded = loadedParams;
}

void InterfaceWorker::setLoadedWInParameters(QStringList loadedParams)
{
    m_parametersWInLoaded = loadedParams;
}

void InterfaceWorker::loadReplay(QString pathReplay)
{
    if(load3DMatrixFromNpPythonSaveTextF(pathReplay + "/xTot.txt", m_xTot))
    {
        sendLogInfo("Replay matrice xTot loaded in the directory : " + pathReplay + "\n", QColor(Qt::blue));
        emit replayLoaded();
    }
}

void InterfaceWorker::startReplay()
{
    cv::Mat *l_xTot = NULL;

    if(m_replayParameters.m_useLastTraining)
    {
        l_xTot = model()->xTotMatrice();
    }
    else
    {
        l_xTot = &m_xTot;
    }

    int l_nbNeurons   = l_xTot->size[1];
    int l_nbSentences = l_xTot->size[0];

    int l_startIdNeurons    = m_replayParameters.m_rangeNeuronsStart;
    int l_endIdNeurons      = m_replayParameters.m_rangeNeuronsEnd;
    int l_startIdSentences  = m_replayParameters.m_rangeSentencesStart;
    int l_endIdSentences    = m_replayParameters.m_rangeSentencesEnd;

    int l_nbRandomNeurons = m_replayParameters.m_randomNeuronsNumber;
    if(l_nbRandomNeurons > l_nbNeurons-1)
    {
        l_nbRandomNeurons = l_nbNeurons-1;
    }

    int l_nbRandomSentences = m_replayParameters.m_randomSentencesNumber;
    if(l_nbRandomSentences > l_nbSentences-1)
    {
        l_nbRandomSentences = l_nbSentences-1;
    }

    if(l_endIdNeurons >  l_nbNeurons-1)
    {
        l_endIdNeurons =  l_nbNeurons-1;
    }

    if(l_endIdSentences >  l_nbSentences-1)
    {
        l_endIdSentences =  l_nbSentences-1;
    }

    // create id of the neurons/sentences activities to be displayed
    QVector<int> l_idNeurons;
    QVector<int> l_idSentences;

    if(m_replayParameters.m_randomNeurons)
    {
        while(l_idNeurons.size() < l_nbRandomNeurons)
        {
            bool l_addId = true;
            int l_idNeuron = rand()%l_nbNeurons;

            for(int ii = 0; ii < l_idNeurons.size(); ++ii)
            {
                if(l_idNeuron == l_idNeurons[ii])
                {
                    l_addId = false;
                    break;
                }
            }

            if(l_addId)
            {
                l_idNeurons << l_idNeuron;
            }
        }
    }
    else
    {
        if(l_endIdNeurons < l_startIdNeurons)
        {
            l_endIdNeurons = l_startIdNeurons;
        }

        for(int ii = l_startIdNeurons; ii <= l_endIdNeurons; ++ii)
        {
            l_idNeurons << ii;
        }
    }

    if(m_replayParameters.m_randomSentence)
    {
        while(l_idSentences.size() < l_nbRandomSentences)
        {
            bool l_addId = true;
            int l_idSentence = rand()%l_nbSentences;

            for(int ii = 0; ii < l_idSentences.size(); ++ii)
            {
                if(l_idSentence == l_idSentences[ii])
                {
                    l_addId = false;
                    break;
                }
            }

            if(l_addId)
            {
                l_idSentences << l_idSentence;
            }
        }
    }
    else
    {
        if(l_endIdSentences < l_startIdSentences)
        {
            l_endIdSentences = l_startIdSentences;
        }

        for(int ii = l_startIdSentences; ii <= l_endIdSentences; ++ii)
        {
            l_idSentences << ii;
        }
    }

    QVector<QVector<double> > l_data;

    for(int ii = 0; ii < l_idNeurons.size(); ++ii)
    {
        QVector<double> l_neuronValues;
        int l_idCurrentNeuron = l_idNeurons[ii];
        for(int jj = 0; jj < l_idSentences.size(); ++jj)
        {
            int l_idCurrentSentence = l_idSentences[jj];
            for(int kk = 0; kk < l_xTot->size[2]; ++kk)
            {
                l_neuronValues << static_cast<double>(l_xTot->at<float>(l_idCurrentSentence,l_idCurrentNeuron,kk));
            }
        }

        l_data << l_neuronValues;
    }


    emit sendReplayData(l_data, l_idNeurons, l_idSentences);
}
