

#include <iostream>
#include "YarpInterface.h"



#include "../moc/moc_YarpInterface.cpp"

using namespace yarp::os;


int main(int argc, char* argv[])
{
    // initialize yarp network
    yarp::os::Network l_oYarp;
    if (!l_oYarp.checkNetwork())
    {
        std::cerr << "-ERROR: Problem connecting to YARP server" << std::endl;
        return -1;
    }

    QCoreApplication l_oApp(argc, argv);
    ReservoirInterface l_interface(&l_oApp);
    return l_oApp.exec();
}



ReservoirInterface::ReservoirInterface(QCoreApplication *parent)
{
    srand(1);
    culaWarmup(1);

    // set absolute path
        m_absolutePath = QDir::currentPath() + "/";

    // create folders
        QVector<QDir> l_dirs;
        l_dirs.push_back(QDir(m_absolutePath     + "../data/input/Corpus"));
        l_dirs.push_back(QDir(m_absolutePath     + "../data/input/Settings"));
        l_dirs.push_back(QDir(m_absolutePath     + "../data/input/Matrices/W"));
        l_dirs.push_back(QDir(m_absolutePath     + "../data/input/Matrices/WIn"));
        l_dirs.push_back(QDir(m_absolutePath     + "../data/input/Settings"));
        l_dirs.push_back(QDir(m_absolutePath     + "../data/Results/raw"));
        l_dirs.push_back(QDir(m_absolutePath     + "../data/training"));
        l_dirs.push_back(QDir(m_absolutePath     + "../data/replay"));
        l_dirs.push_back(QDir(m_absolutePath     + "../data/images"));
        l_dirs.push_back(QDir(m_absolutePath     + "../log"));

        for(int ii = 0; ii < l_dirs.size(); ++ii)
        {
            if(!l_dirs[ii].exists())
            {
                l_dirs[ii].mkpath(".");
            }
        }

    // init worker
    m_yarpWorker = new YarpInterfaceWorker(m_absolutePath);

    // init connections
    QObject::connect(this, SIGNAL(start()), m_yarpWorker, SLOT(doLoop()));
    QObject::connect(this, SIGNAL(stop()), m_yarpWorker, SLOT(stopLoop()));
    QObject::connect(m_yarpWorker, SIGNAL(sendDataToReservoirSignal(int, ModelParameters,Sentence,Sentence)),this, SLOT(startReservoir(int, ModelParameters,Sentence,Sentence)));
    QObject::connect(this, SIGNAL(endReservoirComputing(QVector<std::vector<double> >, QVector<std::vector<double> >, Sentences,Sentences,Sentences)),
                     m_yarpWorker, SLOT(updateResultsFromReservoir(QVector<std::vector<double> >,QVector<std::vector<double> >,Sentences,Sentences,Sentences)));

    // init thread
    m_yarpWorker->moveToThread(&m_yarpWorkerThread);
    m_yarpWorkerThread.start();

    // start loop
    emit start();
}

ReservoirInterface::~ReservoirInterface()
{
    emit stop();

    QTime l_dieTime = QTime::currentTime().addMSecs(200);
    while( QTime::currentTime() < l_dieTime)
    {
        QCoreApplication::processEvents(QEventLoop::AllEvents, 100);
    }

    m_yarpWorkerThread.quit();
    m_yarpWorkerThread.wait();
    delete m_yarpWorker;
}

void ReservoirInterface::startReservoir(int actionToDo, ModelParameters parameters, Sentence CCW, Sentence structure)
{
    // set CCW / structure
        m_model.setCCWAndStructure(CCW,structure);
    // set parameters
        m_model.resetModelParameters(parameters,true);

    QVector<std::vector<double> > l_resultsTrain, l_resultsTests;

    if(actionToDo == 0 || actionToDo == 2)
    {
        m_model.launchTraining();

        std::vector<double> l_diffSizeOCW, l_absoluteCCW, l_continuousCCW, l_absoluteAll, l_continuousAll;
        double l_meanDiffSizeOCW, l_meanContinuousCCW, l_meanAbsoluteCCW, l_meanContinuousAll, l_meanAbsoluteAll;

        m_model.computeResultsData(true, l_diffSizeOCW,
                                    l_absoluteCCW, l_continuousCCW,
                                    l_absoluteAll, l_continuousAll,
                                    l_meanDiffSizeOCW,
                                    l_meanAbsoluteCCW, l_meanContinuousCCW,
                                    l_meanAbsoluteAll, l_meanContinuousAll
                                    );

        l_resultsTrain << l_continuousCCW;
        l_resultsTrain << l_continuousAll;
    }

    if(actionToDo == 1 || actionToDo == 2)
    {
        bool l_error = !m_model.launchTests();

        std::vector<double> l_diffSizeOCW, l_absoluteCCW, l_continuousCCW, l_absoluteAll, l_continuousAll;
        double l_meanDiffSizeOCW, l_meanContinuousCCW, l_meanAbsoluteCCW, l_meanContinuousAll, l_meanAbsoluteAll;

        if(!l_error)
        {
            m_model.computeResultsData(false, l_diffSizeOCW,
                                        l_absoluteCCW, l_continuousCCW,
                                        l_absoluteAll, l_continuousAll,
                                        l_meanDiffSizeOCW,
                                        l_meanAbsoluteCCW, l_meanContinuousCCW,
                                        l_meanAbsoluteAll, l_meanContinuousAll
                                        );
        }
        l_resultsTests << l_continuousCCW;
        l_resultsTests << l_continuousAll;
    }

    Sentences l_trainSentences, l_trainResults, l_testResults;
    m_model.sentences(l_trainSentences, l_trainResults, l_testResults);
    emit endReservoirComputing(l_resultsTrain, l_resultsTests,l_trainSentences, l_trainResults, l_testResults);
}





YarpInterfaceWorker::YarpInterfaceWorker(QString absolutePath) : m_doLoop(true), m_reservoirIsRunning(false), m_isParameters(false), m_isData(false), m_absolutePath(absolutePath)
{
    qRegisterMetaType<ModelParameters>("ModelParameters");
    qRegisterMetaType<Sentence>("Sentence");
    qRegisterMetaType<Sentences>("Sentences");
    qRegisterMetaType<QVector<std::vector<double> >>("QVector<std::vector<double> >");

     // init yarp ports
    m_dataPort.open("/reservoir/data/in");
    m_parametersPort.open("/reservoir/parameters/in");
    m_controlPort.open("/reservoir/control/in");

    m_resultsPort.open("/reservoir/results/out");
}

YarpInterfaceWorker::~YarpInterfaceWorker()
{
    m_dataPort.close();
    m_parametersPort.close();
    m_controlPort.close();

    m_resultsPort.close();
}

void YarpInterfaceWorker::doLoop()
{
    bool l_doLoop = m_doLoop;
    yarp::os::Bottle *l_parametersBottle = NULL;
    yarp::os::Bottle *l_dataBottle = NULL;
    yarp::os::Bottle *l_controlBottle = NULL;

    while(l_doLoop)
    {
        // manage events
        QTime l_dieTime = QTime::currentTime().addMSecs(10);
        while( QTime::currentTime() < l_dieTime)
        {
            QCoreApplication::processEvents(QEventLoop::AllEvents, 10);
        }

        // check if reservoir is busy
        m_reservoirLock.lockForRead();
            bool l_reservoirIsRunning = m_reservoirIsRunning;
        m_reservoirLock.unlock();

        if(!l_reservoirIsRunning)
        {                
            // check parameter port
            l_parametersBottle = m_parametersPort.read(false);
            if(l_parametersBottle)
            {
                readParameters(l_parametersBottle);
            }
            // check data port
            l_dataBottle       = m_dataPort.read(false);
            if(l_dataBottle)
            {
                readData(l_dataBottle);
            }
            // check control port
            l_controlBottle = m_controlPort.read(false);
            if(l_controlBottle)
            {
                m_startReservoir = (l_controlBottle->get(0).asInt()==1); // 0 -> START RESERVOIR (int) (if 1 start, else do nothing)
            }



            if(m_startReservoir && m_isParameters && !l_reservoirIsRunning)
            {
                if(m_isData)
                {
                    if(m_currentModelParameters.m_useLoadedTraining && !m_currentModelParameters.m_useLoadedW && !m_currentModelParameters.m_useLoadedWIn)
                    {
                        // ...
                    }
                    if(m_currentModelParameters.m_useLoadedW && !m_currentModelParameters.m_useLoadedTraining)
                    {
                        // ...
                    }
                    if(m_currentModelParameters.m_useLoadedWIn && !m_currentModelParameters.m_useLoadedTraining)
                    {
                        // ...
                    }
                }

                emit sendDataToReservoirSignal(m_actionToDo, m_currentModelParameters, m_CCWSentence, m_structureSentence);
                m_startReservoir = false;
            }
        }

        // check results
        // ...


        // check if must leave the loop
        m_loopLock.lockForRead();
            l_doLoop = m_doLoop;
        m_loopLock.unlock();
    }
}

void YarpInterfaceWorker::stopLoop()
{
    m_loopLock.lockForWrite();
        m_doLoop = false;
    m_loopLock.unlock();
}


void YarpInterfaceWorker::readParameters(yarp::os::Bottle *parametersBottle)
{   
    m_actionToDo        = parametersBottle->get(0).asInt();                             // 0 -> action to do : 0 train / 1 test / 2 the both
    QString l_corpus    = QString::fromStdString(parametersBottle->get(1).asString());  // 1 -> corpus (string)
    QString l_structure = QString::fromStdString(parametersBottle->get(2).asString());  // 2 -> structure (P0 A1 O2 R3) (string)
    QString l_CCW       = QString::fromStdString(parametersBottle->get(3).asString());  // 3 -> CCW (string)
    m_currentModelParameters.m_nbNeurons         = parametersBottle->get(4).asInt();    // 4 -> NEURONS (int) (if -1 -> default value)
    m_currentModelParameters.m_leakRate          = parametersBottle->get(5).asDouble(); // 5 -> LEAKRATE (double) (if -1 -> default value)
    m_currentModelParameters.m_inputScaling      = parametersBottle->get(6).asDouble(); // 6 -> INPUT SCALING (double) (if -1 -> default value)
    m_currentModelParameters.m_spectralRadius    = parametersBottle->get(7).asDouble(); // 7 -> SPECTRAL RADIUS (double) (if -1 -> default value)
    m_currentModelParameters.m_ridge             = parametersBottle->get(8).asDouble(); // 8 -> RIDGE(double)(if -1 -> default value)
    m_currentModelParameters.m_sparcity          = parametersBottle->get(9).asDouble(); // 9 -> SPARCITY (double)   (if -1 -> automatic recommanded value )
    bool l_useCuda                               = (parametersBottle->get(10).asInt()==1);  // 10 -> USE CUDA (int) (1 -> true / else false)
    m_currentModelParameters.m_useCudaInv        = l_useCuda;
    m_currentModelParameters.m_useCudaMult       = l_useCuda;
    m_currentModelParameters.m_useLoadedTraining = (parametersBottle->get(11).asInt()==1);  // 11 -> use training file from the data port (int) (1 -> true / else false)
    m_currentModelParameters.m_useLoadedW        = (parametersBottle->get(12).asInt()==1);  // 12 -> use loaded w file from the data port (int) (1 -> true / else false)
    m_currentModelParameters.m_useLoadedWIn      = (parametersBottle->get(13).asInt()==1);  // 13 -> use loaded wIn file from the data port (int) (1 -> true / else false)

    // transform strings
    QFile l_fileCorpus(m_absolutePath + "../data/input/Corpus/received.txt");
    if(l_fileCorpus.open(QIODevice::WriteOnly | QIODevice::Text))
    {
        QTextStream in(&l_fileCorpus);
        in << l_corpus;
        m_currentModelParameters.m_corpusFilePath = (m_absolutePath + "../data/input/Corpus/received.txt").toStdString();
    }

    QStringList l_CCWList = l_CCW.split(" ");
    QStringList l_structureList = l_structure.split(" ");

    for(QStringList::iterator ii = l_CCWList.begin(); ii != l_CCWList.end(); ++ii)
    {
        m_CCWSentence.push_back((*ii).toStdString());
    }
    for(QStringList::iterator ii = l_structureList.begin(); ii != l_structureList.end(); ++ii)
    {
        m_structureSentence.push_back((*ii).toStdString());
    }

    m_isParameters = true;
}

void YarpInterfaceWorker::readData(yarp::os::Bottle *dataBottle)
{
    int l_sizesW[2], l_sizesWIn[2], l_sizesWOut[2];
    l_sizesW[0]         = dataBottle->get(0).asInt();   // 3 -> W matrice rows (int) (if 0 -> W matrice not used)
    l_sizesW[1]         = dataBottle->get(1).asInt();   // 4 -> W matrice cols (int) (if 0 -> W matrice not used)
    l_sizesWIn[0]       = dataBottle->get(2).asInt();   // 5 -> WIn matrice rows (int) (if 0 -> Win matrice not used)
    l_sizesWIn[1]       = dataBottle->get(3).asInt();   // 6 -> WIn matrice cols (int) (if 0 -> WIn matrice not used)
    l_sizesWOut[0]      = dataBottle->get(4).asInt();   // 7 -> WOut matrice rows (int) (if 0 -> WOut matrice not used)
    l_sizesWOut[1]      = dataBottle->get(5).asInt();   // 8 -> WOut matrice cols (int) (if 0 -> WOut matrice not used)

    int l_offset1 = l_sizesW[0]*l_sizesW[1]+6;
    if(l_sizesW[0] > 0)
    {
        m_W = cv::Mat(l_sizesW[0], l_sizesW[1], CV_32FC1);
        for(int ii = 7; ii < l_offset1; ++ii)
        {
            m_W.at<float>(ii) = static_cast<float>(dataBottle->get(ii).asDouble());
        }
    }
    int l_offset2 = l_offset1 + l_sizesWIn[0]*l_sizesWIn[1];
    if(l_sizesWIn[0] > 0)
    {
        m_WIn = cv::Mat(l_sizesWIn[0], l_sizesWIn[1], CV_32FC1);
        for(int ii = l_offset1; ii < l_offset2; ++ii)
        {
            m_WIn.at<float>(ii) = static_cast<float>(dataBottle->get(ii).asDouble());
        }
    }
    int l_offset3 = l_offset2 + l_sizesWOut[0]*l_sizesWOut[1];
    if(l_sizesWOut[0] > 0)
    {
        m_WOut = cv::Mat(l_sizesWOut[0], l_sizesWOut[1], CV_32FC1);
        for(int ii = l_offset2; ii < l_offset3; ++ii)
        {
            m_WOut.at<float>(ii) = static_cast<float>(dataBottle->get(ii).asDouble());
        }
    }

    m_isData = true;

    // send data to the reservoir

    // ...
}

void YarpInterfaceWorker::updateResultsFromReservoir(QVector<std::vector<double> > resultsTrain, QVector<std::vector<double> > resultsTest, Sentences trainSentences, Sentences trainResults, Sentences testResults)
{
    QString l_trainSentences, l_trainResults, l_testResults;
    for(int ii = 0; ii < trainSentences.size(); ++ii)
    {
        for(int jj = 0; jj < trainSentences[0].size(); ++jj)
        {
            l_trainSentences += QString::fromStdString(trainSentences[ii][jj]) + " ";
            l_trainResults   += QString::fromStdString(trainResults[ii][jj]) + " ";
        }

        l_trainSentences += "\n";
        l_trainResults += "\n";
    }

    for(int ii = 0; ii < testResults.size(); ++ii)
    {
        for(int jj = 0; jj < testResults[0].size(); ++jj)
        {
            l_testResults += QString::fromStdString(testResults[ii][jj]) + " ";
        }

        l_testResults += "\n";
    }

    std::vector<double> l_trainCCWContinuous, l_trainAllContinuous, l_testsCCWContinuous, l_testsAllContinuous;
    if(resultsTrain.size() > 0)
    {
        l_trainCCWContinuous = resultsTrain[0];
        l_trainAllContinuous = resultsTrain[1];
    }
    if(resultsTest.size() > 0)
    {
        l_testsCCWContinuous = resultsTest[0];
        l_testsAllContinuous = resultsTest[1];
    }
    QString l_trainCCWContinuousString, l_trainAllContinuousString, l_testsCCWContinuousString, l_testsAllContinuousString;
    for(int ii = 0; ii < l_trainCCWContinuous.size(); ++ii)
    {
        l_trainCCWContinuousString += QString::number(l_trainCCWContinuous[ii]) + " ";
        l_trainAllContinuousString += QString::number(l_trainAllContinuous[ii]) + " ";
    }
    for(int ii = 0; ii < l_testsCCWContinuousString.size(); ++ii)
    {
        l_testsCCWContinuousString += QString::number(l_testsCCWContinuous[ii]) + " ";
        l_testsAllContinuousString += QString::number(l_testsAllContinuous[ii]) + " ";
    }

    yarp::os::Bottle &l_resultsBottle = m_resultsPort.prepare();
    l_resultsBottle.addString(l_trainSentences.toStdString());
    l_resultsBottle.addString(l_trainResults.toStdString());
    l_resultsBottle.addString(l_testResults.toStdString());
    l_resultsBottle.addString(l_trainCCWContinuousString.toStdString());
    l_resultsBottle.addString(l_trainAllContinuousString.toStdString());
    l_resultsBottle.addString(l_testsCCWContinuousString.toStdString());
    l_resultsBottle.addString(l_testsAllContinuousString.toStdString());

    m_resultsPort.write();

    m_reservoirLock.lockForWrite();
        m_reservoirIsRunning = false;
    m_reservoirLock.unlock();
}
