
/**
 * \file ReservoirRFModule.h
 * \author Florian Lance
 * \date 14-01-2015
 * \brief Define ReservoirRFModule class
 */

#ifndef RESERVOIRRFMODULE_H
#define RESERVOIRRFMODULE_H



// YARP
#include <yarp/os/all.h>
#include <yarp/os/Network.h>

//#include <yarp/os/RFModule.h>
//#include <yarp/os/Time.h>
//#include <yarp/os/Port.h>
//#include <yarp/os/Bottle.h>
//#include <yarp/os/Property.h>

//#include <yarp/sig/Vector.h>

//#include <yarp/math/Math.h>

//#include <yarp/dev/Drivers.h>
//#include <yarp/dev/CartesianControl.h>
//#include <yarp/dev/PolyDriver.h>
//#include <yarp/dev/ControlBoardInterfaces.h>


//using namespace yarp::os;
//using namespace yarp::dev;
//using namespace yarp::sig;
//using namespace yarp::math;


/**
 * \class ReservoirRFModule
 * \author Florian Lance
 * \date 14-01-2015
 * \brief A RFModule interface for the neuron computing reservoir.
 */
class ReservoirRFModule : public RFModule
{
    public:

        /**
         * \brief ReservoirRFModule constructor
         */
        ReservoirRFModule();

        /**
         * \brief ReservoirRFModule destructor
         */
        ~ReservoirRFModule();

        /**
         * \brief The configure function loads the config options.
         *
         * This function loads the config options.
         * \param rf: the resource finder  address
         * \return true if the configure step was successfull
         */
        bool configure(ResourceFinder &rf);

        /**
         * \brief The close function terminates the connection and listening
         *
         * This function terminates the listening mechanism and related processes
         * to stop the eye-tracker connection.
         * \return true if the closing step was ok
         */
        bool close();

        /**
         * \brief The updateModule function update the module.
         *
         * This function updates the module.
         * \return true if the update step was successfull
         */
        bool updateModule();

        /**
         * \brief The interruptModule function interrupts the module.
         *
         * This function interrupts the module.
         * \return true if the interrupt module step was successfull
         */
        bool interruptModule();

        /**
         * \brief The getPeriod function to choose the period of update.
         *
         * This function gets the period of update.
         * \return a value in second which correponds to the period of calling th upDateModule() method
         */
        double getPeriod();



    private:

        bool m_bIsRunning;                      /**<  Whether the thread is running */
        bool m_bHeadInitialized;                /**< ... */
        bool m_bTorsoInitialized;               /**< ... */
        bool m_bLeftArmInitialized;               /**< ... */
        bool m_bRightArmInitialized;               /**< ... */

        int m_i32Fps;                           /**< fps (define the period for calling updateModule) */

        // Config variables retrieved from the ini file
        std::string m_sModuleName;              /**< name of the mondule (config) */
        std::string m_sRobotName;               /**< name of the robot (config) */

};

#endif

