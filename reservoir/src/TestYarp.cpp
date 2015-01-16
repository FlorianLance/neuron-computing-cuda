
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
    l_dataPort.open("/reservoir/data/out");
    l_parametersPort.open("/reservoir/parameters/out");
    l_controlPort.open("/reservoir/control/out");

    qDebug() << "Start ?";
    std::string l_s;
    std::cin >> l_s;

    // fill bottles
    yarp::os::Bottle &l_parametersBottle = l_parametersPort.prepare();
    l_parametersBottle.addString(l_corpusString.toStdString());             // 0 -> corpus (string)
    l_parametersBottle.addString(l_CCW.toStdString());                      // 1 -> structure (P0 A1 O2 R3) (string)
    l_parametersBottle.addString(l_structure.toStdString());                // 2 -> CCW (string)
    l_parametersBottle.addInt(500);                                         // 3 -> NEURONS (int) (if -1 -> default value)
    l_parametersBottle.addDouble(0.5);                                      // 4 -> LEAKRATE (double) (if -1 -> default value)
    l_parametersBottle.addDouble(0.2);                                      // 5 -> INPUT SCALING (double) (if -1 -> default value)
    l_parametersBottle.addDouble(4.0);                                      // 6 -> SPECTRAL RADIUS (double) (if -1 -> default value)
    l_parametersBottle.addDouble(-1);                                       // 7 -> RIDGE(double)(if -1 -> default value)
    l_parametersBottle.addDouble(-1);                                       // 8 -> SPARCITY (double)   (if -1 -> automatic recommanded value)
    l_parametersBottle.addInt(1);                                           // 9 -> USE CUDA (int) (if 1, use CUDA else do not use)
    l_parametersBottle.addInt(-1);                                          // 10-> use training file from the data port (int) (if 1, use training file else do not use)
    l_parametersBottle.addInt(-1);                                          // 11-> use loaded w file from the data port  (int) (if 1, use loaded w fil eelse do not use)
    l_parametersBottle.addInt(-1);                                          // 12-> use loaded wIn file from the data port (int) (if 1, use loaded wIn file else do not use)
    l_parametersPort.write();




    l_dataPort.close();
    l_parametersPort.close();
    l_controlPort.close();

    return 0;
}



