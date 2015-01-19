
/**
 * \file YarpInterface.h
 * \author Florian Lance
 * \date 14-01-2015
 * \brief Define YarpInterface class
 */

#ifndef RESERVOIRRFMODULE_H
#define RESERVOIRRFMODULE_H


// Reservoir
#include "Model.h"

// Qt
#include <QtCore>

// YARP
#include <yarp/os/all.h>
#include <yarp/os/Network.h>



class YarpInterfaceWorker;

class ReservoirInterface : public QObject
{
    Q_OBJECT


    public :

        ReservoirInterface(QCoreApplication *parent);

        ~ReservoirInterface();


    public slots :


        void startReservoir(int actionToDo, ModelParameters parameters,Sentence CCW,Sentence structure);


    signals :

        void start();

        void stop();

        void endReservoirComputing(QVector<std::vector<double> > , QVector<std::vector<double> >, Sentences, Sentences, Sentences);

    private :

        YarpInterfaceWorker  *m_yarpWorker;
        QThread         m_yarpWorkerThread;

        Model m_model;
        QString m_absolutePath;
};


class YarpInterfaceWorker : public QObject
{
    Q_OBJECT

    public :

        /**
         * \brief Constructor of YarpInterfaceWorker
         */
        YarpInterfaceWorker(QString absolutePath);

        ~YarpInterfaceWorker();

    private :

        void readParameters(yarp::os::Bottle *parametersBottle);

        void readData(yarp::os::Bottle *dataBottle);


    public slots:

        void doLoop();

        void stopLoop();

        void updateResultsFromReservoir(QVector<std::vector<double> > resultsTrain, QVector<std::vector<double> > resultsTests, Sentences trainSentences, Sentences trainResults, Sentences testResults);


    signals :


        void sendDataToReservoirSignal(int, ModelParameters, Sentence, Sentence);


    private :

        bool m_doLoop;
        bool m_isParameters;
        bool m_isData;
        bool m_reservoirIsRunning;
        bool m_startReservoir;

        QString m_absolutePath;

        QReadWriteLock m_loopLock;
        QReadWriteLock m_reservoirLock;

        yarp::os::BufferedPort<yarp::os::Bottle> m_controlPort;
        yarp::os::BufferedPort<yarp::os::Bottle> m_dataPort;
        yarp::os::BufferedPort<yarp::os::Bottle> m_parametersPort;

        yarp::os::BufferedPort<yarp::os::Bottle> m_resultsPort;

        int m_actionToDo;
        ModelParameters m_currentModelParameters;
        Sentence m_CCWSentence, m_structureSentence;

        cv::Mat m_W;
        cv::Mat m_WIn;
        cv::Mat m_WOut;

};



#endif

