/*******************************************************************************
**                                                                            **
**  Language Learning - Reservoir Computing - GPU                             **
**  An interface for language learning with neuron computing using GPU        **
**  acceleration.                                                             **
**                                                                            **
**  This program is free software: you can redistribute it and/or modify      **
**  it under the terms of the GNU Lesser General Public License as published  **
**  by the Free Software Foundation, either version 3 of the License, or      **
**  (at your option) any later version.                                       **
**                                                                            **
**  This program is distributed in the hope that it will be useful,           **
**  but WITHOUT ANY WARRANTY; without even the implied warranty of            **
**  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the             **
**  GNU Lesser General Public License for more details.                       **
**                                                                            **
**  You should have received a copy of the GNU Lesser General Public License  **
**  along with Foobar.  If not, see <http://www.gnu.org/licenses/>.           **
**                                                                            **
********************************************************************************/


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

/**
 * @brief The ReservoirInterface class
 */
class ReservoirInterface : public QObject
{
    Q_OBJECT


    public :

        /**
         * @brief Constructor of ReservoirInterface
         * @param parent
         */
        ReservoirInterface(QCoreApplication *parent);

        /**
         * @brief Destructor of ReservoirInterface
         */
        ~ReservoirInterface();


    public slots :

        /**
         * @brief startReservoir
         * @param actionToDo
         * @param parameters
         * @param CCW
         * @param structure
         * @param pathTrainingFileToBeSaved
         * @param pathWMatriceFileToBeSaved
         * @param pathWInMatriceFileToBeSaved
         * @param pathTrainingFileToBeLoaded
         * @param pathWMatriceFileToBeLoaded
         * @param pathWInMatriceFileToBeLoaded
         */
        void startReservoir(int actionToDo, ModelParameters parameters,Sentence CCW,Sentence structure,
                            QString pathTrainingFileToBeSaved, QString pathWMatriceFileToBeSaved, QString pathWInMatriceFileToBeSaved,
                            QString pathTrainingFileToBeLoaded, QString pathWMatriceFileToBeLoaded, QString pathWInMatriceFileToBeLoaded);


    signals :

        /**
         * @brief start
         */
        void start();

        /**
         * @brief stop
         */
        void stop();

        /**
         * @brief endReservoirComputing
         */
        void endReservoirComputing(QVector<std::vector<double> > , QVector<std::vector<double> >, Sentences, Sentences, Sentences);

    private :

        YarpInterfaceWorker  *m_yarpWorker; /**< yarp worker */
        QThread         m_yarpWorkerThread; /**< yarp worker thread */

        Model m_model;                      /**< model of the reservoir */
        QString m_absolutePath;             /**< absolute path initialied at the launching */
};

/**
 * @brief The YarpInterfaceWorker class
 */
class YarpInterfaceWorker : public QObject
{
    Q_OBJECT

    public :

        /**
         * \brief Constructor of YarpInterfaceWorker
         */
        YarpInterfaceWorker(QString absolutePath);


        /**
         * @brief Destructor of YarpInterfaceWorker
         */
        ~YarpInterfaceWorker();

    private :

        /**
         * @brief readParameters
         * @param parametersBottle
         */
        void readParameters(yarp::os::Bottle *parametersBottle);

    public slots:

        /**
         * @brief doLoop
         */
        void doLoop();

        /**
         * @brief stopLoop
         */
        void stopLoop();

        /**
         * @brief updateResultsFromReservoir
         * @param resultsTrain
         * @param resultsTests
         * @param trainSentences
         * @param trainResults
         * @param testResults
         */
        void updateResultsFromReservoir(QVector<std::vector<double> > resultsTrain, QVector<std::vector<double> > resultsTests, Sentences trainSentences, Sentences trainResults, Sentences testResults);


    signals :

        /**
         * @brief sendDataToReservoirSignal
         */
        void sendDataToReservoirSignal(int, ModelParameters, Sentence, Sentence, QString, QString, QString, QString, QString, QString);


    private :

        bool m_doLoop;              /**< do the main loop ? */
        bool m_isParameters;        /**< is parameters received ? */
        bool m_reservoirIsRunning;  /**< is the reservoir running ? */
        bool m_startReservoir;      /**< start the reservoir ? */

        QString m_absolutePath;     /**< absolute path initialized at the launching */

        QReadWriteLock m_loopLock;      /**< mutex lock for the loop */
        QReadWriteLock m_reservoirLock; /**< mutex lock for the reservoir */

        yarp::os::BufferedPort<yarp::os::Bottle> m_controlPort;     /**< port for receiving control data */
        yarp::os::BufferedPort<yarp::os::Bottle> m_parametersPort;  /**< port for receiving parameters data */
        yarp::os::BufferedPort<yarp::os::Bottle> m_resultsPort;     /**< port for sending results data */

        int m_actionToDo;                               /**< what to do ? train 0 / test 1 / both 2 */
        ModelParameters m_currentModelParameters;       /**< last parameters received for the model */
        Sentence m_CCWSentence;                         /**< current CCW sentence */
        Sentence m_structureSentence;                   /**< current structure sentence */
        QString m_pathTrainingToBeSaved;                /**< path for saving the training */
        QString m_pathWToBeSaved;                       /**< path for saving the W matrice */
        QString m_pathWInToBeSaved;                     /**< path for saving the WIn matrice */

        QString m_pathTrainingToBeLoaded;               /**< path for loading the training */
        QString m_pathWToBeLoaded;                      /**< path for loading the W matrice */
        QString m_pathWInToBeLoaded;                    /**< path for loading the WIn matrice */
};



#endif

