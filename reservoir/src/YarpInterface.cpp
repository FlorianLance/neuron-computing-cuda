

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

void ReservoirInterface::updateParameters(ModelParameters parameters)
{
    m_model.resetModelParameters(parameters, false);
}





YarpInterfaceWorker::YarpInterfaceWorker(QString absolutePath) : m_doLoop(true), m_reservoirIsRunning(false), m_isParameters(false), m_isData(false), m_absolutePath(absolutePath)
{
     // init yarp ports
    m_dataPort.open("/reservoir/data/in");
    m_parametersPort.open("/reservoir/parameters/in");
    m_controlPort.open("/reservoir/control/in");
}

YarpInterfaceWorker::~YarpInterfaceWorker()
{
    m_dataPort.close();
    m_parametersPort.close();
    m_controlPort.close();
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

        // check if reservoir is busy
        m_reservoirLock.lockForRead();
            bool l_reservoirIsRunning = m_reservoirIsRunning;
        m_reservoirLock.unlock();

        if(!l_reservoirIsRunning)
        {
            if(m_startReservoir && m_isParameters)
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

                emit sendDataToReservoirSignal(m_currentModelParameters, m_CCWSentence, m_structureSentence);
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
    QString l_corpus    = QString::fromStdString(parametersBottle->get(0).asString());  // 0 -> corpus (string)
    QString l_structure = QString::fromStdString(parametersBottle->get(1).asString());  // 1 -> structure (P0 A1 O2 R3) (string)
    QString l_CCW       = QString::fromStdString(parametersBottle->get(2).asString());  // 2 -> CCW (string)
    m_currentModelParameters.m_nbNeurons         = parametersBottle->get(3).asInt();    // 3 -> NEURONS (int) (if -1 -> default value)
    m_currentModelParameters.m_leakRate          = parametersBottle->get(4).asDouble(); // 4 -> LEAKRATE (double) (if -1 -> default value)
    m_currentModelParameters.m_inputScaling      = parametersBottle->get(5).asDouble(); // 5 -> INPUT SCALING (double) (if -1 -> default value)
    m_currentModelParameters.m_spectralRadius    = parametersBottle->get(6).asDouble(); // 6 -> SPECTRAL RADIUS (double) (if -1 -> default value)
    m_currentModelParameters.m_ridge             = parametersBottle->get(7).asDouble(); // 7 -> RIDGE(double)(if -1 -> default value)
    m_currentModelParameters.m_sparcity          = parametersBottle->get(8).asDouble(); // 8 -> SPARCITY (double)   (if -1 -> automatic recommanded value )
    bool l_useCuda                               = (parametersBottle->get(9).asInt()==1);   // 9 -> USE CUDA (int) (1 -> true / else false)
    m_currentModelParameters.m_useCudaInv  = l_useCuda;
    m_currentModelParameters.m_useCudaMult = l_useCuda;
    m_currentModelParameters.m_useLoadedTraining = (parametersBottle->get(10).asInt()==1);  // 10 -> use training file from the data port (int) (1 -> true / else false)
    m_currentModelParameters.m_useLoadedW        = (parametersBottle->get(11).asInt()==1);  // 11 -> use loaded w file from the data port (int) (1 -> true / else false)
    m_currentModelParameters.m_useLoadedWIn      = (parametersBottle->get(12).asInt()==1);  // 12 -> use loaded wIn file from the data port (int) (1 -> true / else false)


    // display :
    qDebug() << l_corpus << " " << l_structure << " " << l_CCW;
    qDebug() << m_currentModelParameters.m_nbNeurons << " " << m_currentModelParameters.m_leakRate << " " << m_currentModelParameters.m_inputScaling;
    qDebug() << m_currentModelParameters.m_spectralRadius << " " << m_currentModelParameters.m_ridge << " " << m_currentModelParameters.m_sparcity;
    qDebug() << m_currentModelParameters.m_spectralRadius << " " << m_currentModelParameters.m_ridge << " " << m_currentModelParameters.m_sparcity;
    qDebug() << m_currentModelParameters.m_useCudaInv << " " << m_currentModelParameters.m_useCudaMult << " " << m_currentModelParameters.m_useLoadedTraining;
    qDebug() << m_currentModelParameters.m_useLoadedW << " " << m_currentModelParameters.m_useLoadedWIn;


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

    qDebug() << "display sentences";
    displaySentence(m_CCWSentence);
    displaySentence(m_structureSentence);


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