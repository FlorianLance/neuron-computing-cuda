
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


        void updateParameters(ModelParameters parameters);


    signals :

        void start();

        void stop();

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
        YarpInterfaceWorker();

        ~YarpInterfaceWorker();

    private :

        void readParameters(yarp::os::Bottle *parametersBottle);

        void readData(yarp::os::Bottle *dataBottle);


    public slots:

        void doLoop();


        void stopLoop();


    signals :


    private :

        bool m_doLoop;
        bool m_isParameters;
        bool m_isData;
        bool m_reservoirIsRunning;
        bool m_startReservoir;

        QReadWriteLock m_loopLock;

        yarp::os::BufferedPort<yarp::os::Bottle> m_controlPort;
        yarp::os::BufferedPort<yarp::os::Bottle> m_dataPort;
        yarp::os::BufferedPort<yarp::os::Bottle> m_parametersPort;

        ModelParameters m_currentModelParameters;
        cv::Mat m_W;
        cv::Mat m_WIn;
        cv::Mat m_WOut;

};


/**
 * \class ReservoirRFModule
 * \author Florian Lance
 * \date 14-01-2015
 * \brief A RFModule interface for the neuron computing reservoir.
 */
//class ReservoirRFModule : public RFModule
//{
//    public:

//        /**
//         * \brief ReservoirRFModule constructor
//         */
//        ReservoirRFModule();

//        /**
//         * \brief ReservoirRFModule destructor
//         */
//        ~ReservoirRFModule();

//        /**
//         * \brief The configure function loads the config options.
//         *
//         * This function loads the config options.
//         * \param rf: the resource finder  address
//         * \return true if the configure step was successfull
//         */
//        bool configure(ResourceFinder &rf);

//        /**
//         * \brief The close function terminates the connection and listening
//         *
//         * This function terminates the listening mechanism and related processes
//         * to stop the eye-tracker connection.
//         * \return true if the closing step was ok
//         */
//        bool close();

//        /**
//         * \brief The updateModule function update the module.
//         *
//         * This function updates the module.
//         * \return true if the update step was successfull
//         */
//        bool updateModule();

//        /**
//         * \brief The interruptModule function interrupts the module.
//         *
//         * This function interrupts the module.
//         * \return true if the interrupt module step was successfull
//         */
//        bool interruptModule();

//        /**
//         * \brief The getPeriod function to choose the period of update.
//         *
//         * This function gets the period of update.
//         * \return a value in second which correponds to the period of calling th upDateModule() method
//         */
//        double getPeriod();



//    private:

//        bool m_bIsRunning;                      /**<  Whether the thread is running */
//        bool m_bHeadInitialized;                /**< ... */
//        bool m_bTorsoInitialized;               /**< ... */
//        bool m_bLeftArmInitialized;               /**< ... */
//        bool m_bRightArmInitialized;               /**< ... */

//        int m_i32Fps;                           /**< fps (define the period for calling updateModule) */

//        // Config variables retrieved from the ini file
//        std::string m_sModuleName;              /**< name of the mondule (config) */
//        std::string m_sRobotName;               /**< name of the robot (config) */

//};

#endif

