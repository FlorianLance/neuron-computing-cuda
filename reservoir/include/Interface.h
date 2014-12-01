
/**
 * \file Interface.h
 * \brief Defines SWViewerInterface
 * \author Florian Lance
 * \date 01/12/14
 */

#ifndef _INTERFACE_
#define _INTERFACE_

#include <QMainWindow>
#include <QThread>

#include <QtGui>

//#include "SWViewerWorker.h"

#include "../genUI/UI_Interface.h"

//#include "interface/QtWidgets/SWGLMultiObjectWidget.h"

//#include "animation/SWAnimation.h"



namespace Ui {
    class UI_Reservoir;
}


class InterfaceWorker;

struct ReservoirParameter
{
    int m_neuronsStart;
    double m_leakRateStart;
    double m_issStart;
    double m_spectralRadiusStart;
    double m_ridgeStart;
    double m_sparcityStart;

    int m_neuronsEnd;
    double m_leakRateEnd;
    double m_issEnd;
    double m_spectralRadiusEnd;
    double m_ridgeEnd;
    double m_sparcityEnd;

    bool m_neuronsEnabled;
    bool m_leakRateEnabled;
    bool m_issEnabled;
    bool m_spectralRadiusEnabled;
    bool m_ridgeEnabled;
    bool m_sparcityEnabled;

    QString m_neuronsOperation;
    QString m_leakRateOperation;
    QString m_issOperation;
    QString m_spectralRadiusOperation;
    QString m_ridgeOperation;
    QString m_sparcityOperation;
};


/**
 * \class Interface
 * \brief ...
 * \author Florian Lance
 * \date 01/12/14
 */
class Interface : public QMainWindow
{
    Q_OBJECT

    public :


        // ############################################# CONSTRUCTORS / DESTRUCTORS

        /**
         * \brief Constructor of Interface
         */
        Interface();

        /**
         * \brief Destructor of Interface
         */
        ~Interface();


        // ############################################# METHODS

    public slots:

        /**
         * @brief closeEvent
         * @param event
         */
        void closeEvent(QCloseEvent *event);

        /**
         * @brief addCorpus
         */
        void addCorpus();

        /**
         * @brief removeCorpus
         */
        void removeCorpus();

        /**
         * @brief updateParamters
         */
        void updateReservoirParameters();


        void updateReservoirParameters(int value);

        void updateReservoirParameters(double value);

        void updateReservoirParameters(QString value);

        void updateReservoirParameters(bool value);

    signals:

        /**
         * @brief addCorpusSignal
         */
        void addCorpusSignal(QString);

        /**
         * @brief removeCorpusSignal
         */
        void removeCorpusSignal(int);

        /**
         * @brief sendReservoirParametersSignal
         */
        void sendReservoirParametersSignal(ReservoirParameter);


    private :


    public :

        // widgets & ui
        Ui::UI_Reservoir* m_uiInterface;   /**< qt main window */

        // threads & workers
        InterfaceWorker  *m_pWInterface;    /**< viewer worker */
        QThread         m_TInterface;    /**< viewer thread */

};



/**
 * @brief The SWViewerWorker class
 */
class InterfaceWorker : public QObject
{
    Q_OBJECT

    public :

        /**
         * \brief constructor of InterfaceWorker
         */
        InterfaceWorker();


    public slots:


    signals:


    private :


//        QReadWriteLock m_oMutex;                /**< ... */
};


#endif
