
#include <iostream>

// YARP
#include <yarp/os/all.h>
#include <yarp/os/Network.h>


#include "GridSearch.h"

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

    QString l_path = QDir::currentPath() + "/";
    qDebug()  << l_path;

    QString l_pathCorpus   = l_path + "../data/input/Corpus/10.txt";
    QFile l_fileCorpus(l_pathCorpus);
    QString l_corpusString;

    QString l_structure = "P0 A1 O2 R3";
    QString l_CCW = "and s of the to . -ed -ing -s by it that was did , from";

    if(l_fileCorpus.open(QIODevice::ReadOnly | QIODevice::Text))
    {
        QTextStream in(&l_fileCorpus);
        l_corpusString = in.readAll();
    }

    yarp::os::BufferedPort<yarp::os::Bottle> l_controlPort;
    yarp::os::BufferedPort<yarp::os::Bottle> l_dataPort;
    yarp::os::BufferedPort<yarp::os::Bottle> l_parametersPort;
    yarp::os::BufferedPort<yarp::os::Bottle> l_resultsPort;
    l_dataPort.open("/reservoir/data/out");
    l_parametersPort.open("/reservoir/parameters/out");
    l_controlPort.open("/reservoir/control/out");

    l_resultsPort.open("/reservoir/results/in");

    qDebug() << "Enter a key and press enter to start.";
    std::string l_s;
    std::cin >> l_s;

    {
        // send parameters
        yarp::os::Bottle &l_parametersBottle = l_parametersPort.prepare();
        l_parametersBottle.addInt(0);                                           // 0 -> action to do : 0 -> train / 1 -> test / 2 -> both
        l_parametersBottle.addString(l_corpusString.toStdString());             // 1 -> corpus (string)
        l_parametersBottle.addString(l_structure.toStdString());                // 2 -> structure (P0 A1 O2 R3) (string)
        l_parametersBottle.addString(l_CCW.toStdString());                      // 3 -> CCW (string)
        l_parametersBottle.addInt(500);                                         // 4 -> NEURONS (int) (if -1 -> default value)
        l_parametersBottle.addDouble(0.5);                                      // 5 -> LEAKRATE (double) (if -1 -> default value)
        l_parametersBottle.addDouble(0.2);                                      // 6 -> INPUT SCALING (double) (if -1 -> default value)
        l_parametersBottle.addDouble(4.0);                                      // 7 -> SPECTRAL RADIUS (double) (if -1 -> default value)
        l_parametersBottle.addDouble(-1);                                       // 8 -> RIDGE(double)(if -1 -> default value)
        l_parametersBottle.addDouble(-1);                                       // 9 -> SPARCITY (double)   (if -1 -> automatic recommanded value)
        l_parametersBottle.addInt(1);                                           // 10-> USE CUDA (int) (if 1, use CUDA else do not use)
        l_parametersBottle.addInt(-1);                                          // 11-> use training file from the data port (int) (if 1, use training file else do not use)
        l_parametersBottle.addInt(-1);                                          // 12-> use loaded w file from the data port  (int) (if 1, use loaded w fil eelse do not use)
        l_parametersBottle.addInt(-1);                                          // 13-> use loaded wIn file from the data port (int) (if 1, use loaded wIn file else do not use)
        l_parametersPort.write();

        // start the reservoir
        yarp::os::Bottle &l_controlBottle = l_controlPort.prepare();
        l_controlBottle.addInt(1);                                              // 0 -> START RESERVOIR (int) (if 1 start, else do nothing)
        l_controlPort.write();
    }

    // retrieve results
        bool l_resultsReceived = false;
        while(!l_resultsReceived)
        {
            yarp::os::Bottle *l_resultsBottle = l_resultsPort.read(false);

            if(l_resultsBottle)
            {
                l_resultsReceived = true;
                QString l_trainSentences = QString::fromStdString(l_resultsBottle->get(0).asString()); // 0 -> train sentences (string)
                QString l_trainResults   = QString::fromStdString(l_resultsBottle->get(1).asString()); // 1 -> train results (string)
                QString l_testsResults   = QString::fromStdString(l_resultsBottle->get(2).asString()); // 2 -> test results (string)
                QString l_trainCCW       = QString::fromStdString(l_resultsBottle->get(3).asString()); // 3 -> train CCW (string)
                QString l_trainAll       = QString::fromStdString(l_resultsBottle->get(4).asString()); // 4 -> train All (string)
                QString l_testsCCW       = QString::fromStdString(l_resultsBottle->get(5).asString()); // 5 -> test CCW (string)
                QString l_testsAll       = QString::fromStdString(l_resultsBottle->get(6).asString()); // 6 -> test All (string)
                qDebug() << l_trainSentences;
                qDebug() << l_trainResults;
                qDebug() << l_testsResults;
                qDebug() << l_trainCCW;
                qDebug() << l_trainAll;
                qDebug() << l_testsCCW;
                qDebug() << l_testsAll;
            }

            QTime l_dieTime = QTime::currentTime().addMSecs(100);
            while( QTime::currentTime() < l_dieTime)
            {
                QCoreApplication::processEvents(QEventLoop::AllEvents, 100);
            }
        }

    // in order to change the parameter, just send another parameter bottle, and a new control bottle
//    {
//        yarp::os::Bottle &l_parametersBottle = l_parametersPort.prepare();
//        l_parametersBottle.addInt(0);                                           // 0 -> action to do : 0 -> train / 1 -> test / 2 -> both
//        l_parametersBottle.addString(l_corpusString.toStdString());             // 1 -> corpus (string)
//        l_parametersBottle.addString(l_structure.toStdString());                // 2 -> structure (P0 A1 O2 R3) (string)
//        l_parametersBottle.addString(l_CCW.toStdString());                      // 3 -> CCW (string)
//        l_parametersBottle.addInt(1000);                                         // 4 -> NEURONS (int) (if -1 -> default value)
//        l_parametersBottle.addDouble(0.5);                                      // 5 -> LEAKRATE (double) (if -1 -> default value)
//        l_parametersBottle.addDouble(0.2);                                      // 6 -> INPUT SCALING (double) (if -1 -> default value)
//        l_parametersBottle.addDouble(4.0);                                      // 7 -> SPECTRAL RADIUS (double) (if -1 -> default value)
//        l_parametersBottle.addDouble(-1);                                       // 8 -> RIDGE(double)(if -1 -> default value)
//        l_parametersBottle.addDouble(-1);                                       // 9 -> SPARCITY (double)   (if -1 -> automatic recommanded value)
//        l_parametersBottle.addInt(1);                                           // 10-> USE CUDA (int) (if 1, use CUDA else do not use)
//        l_parametersBottle.addInt(-1);                                          // 11-> use training file from the data port (int) (if 1, use training file else do not use)
//        l_parametersBottle.addInt(-1);                                          // 12-> use loaded w file from the data port  (int) (if 1, use loaded w fil eelse do not use)
//        l_parametersBottle.addInt(-1);                                          // 13-> use loaded wIn file from the data port (int) (if 1, use loaded wIn file else do not use)
//        l_parametersPort.write();

//        yarp::os::Bottle &l_controlBottle = l_controlPort.prepare();
//        l_controlBottle.addInt(1);                                              // 0 -> START RESERVOIR (int) (if 1 start, else do nothing)
//        l_controlPort.write();
//    }



    l_dataPort.close();
    l_parametersPort.close();
    l_controlPort.close();

    return 0;
}



