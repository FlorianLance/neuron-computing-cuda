
/**
 * \file Interface.h
 * \brief Defines SWViewerInterface
 * \author Florian Lance
 * \date 01/12/14
 */

#ifndef _INTERFACE_
#define _INTERFACE_

//#include <QtWidgets/qmainwindow.h>
//#include <QtCore/qthread.h>
#include <QMainWindow>
#include <QThread>
#include <QtGui>

//#include "SWViewerWorker.h"

#include "../genUI/UI_Interface.h"


//#include "GridSearch.h"
#include "GridSearchQtObject.h"


#include "DisplayImageWidget.h"

#include "qcustomplot.h"


namespace Ui {
    class UI_Reservoir;
}


class InterfaceWorker;


enum ActionToDo
{
    TRAINING_RES,TEST_RES,BOTH_RES
};

struct LanguageParameters
{
    QString m_structure;
    QString m_CCW;
};

struct ReservoirParameters
{
    bool m_useCuda;

    bool m_useLoadedTraining;
    bool m_useOnlyStartValue;    

    ActionToDo m_action;

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

    int m_neuronsNbOfUses;
    int m_leakRateNbOfUses;
    int m_issNbOfUses;
    int m_spectralRadiusNbOfUses;
    int m_ridgeNbOfUses;
    int m_sparcityNbOfUses;

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
        Interface(QApplication *parent);

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
         * @brief saveTraining
         */
        void saveTraining();

        /**
         * @brief loadTraining
         */
        void loadTraining();

        /**
         * @brief updateParamters
         */
        void updateReservoirParameters();

        void updateReservoirParameters(int value);

        void updateReservoirParameters(double value);

        void updateReservoirParameters(QString value);

        /**
         * @brief lockInterface
         * @param lock
         */
        void lockInterface(bool lock);

        /**
         * @brief displayValidityOperation
         * @param operationValid
         * @param indexParameter
         */
        void displayValidityOperation(bool operationValid, int indexParameter);

        /**
         * @brief displayCurrentParameters
         */
        void displayCurrentParameters(ModelParametersQt params);

        /**
         * @brief displayCurrentResults
         * @param results
         */
        void displayCurrentResults(ResultsDisplayReservoir results);

        /**
         * @brief updateProgressBar
         * @param currentValue
         * @param valueMax
         * @param text
         */
        void updateProgressBar(int currentValue, int valueMax, QString text);

        /**
         * @brief displayXMatrix
         * @param values
         * @param currentSentenceId
         * @param nbSentences
         */
        void displayXMatrix(QVector<QVector<double> > *values, int currentSentenceId, int nbSentences);


        /**
         * @brief initPlot
         * @param nbCurves
         * @param sizeDim1Meaning
         * @param sizeDim2Meaning
         * @param name
         */
        void initPlot(int nbCurves, int sizeDim1Meaning, int sizeDim2Meaning, QString name);

        /**
         * @brief cleanResultsDisplay
         */
        void cleanResultsDisplay();

        /**
         * @brief displayLogInfo
         * @param info
         */
        void displayLogInfo(QString info);

        /**
         * @brief displayOutputMatrix
         * @param output
         */
        void displayOutputMatrix(cv::Mat output);

        /**
         * @brief displayTrainInputMatrix
         */
        void displayTrainInputMatrix(cv::Mat trainMeaning, cv::Mat trainSentence);

        /**
         * @brief openCorpus
         */
        void openCorpus();

        /**
         * @brief reloadCorpus
         */
        void reloadCorpus();

        /**
         * @brief loadSettings
         */
        void loadSettings();

        /**
         * @brief updateSettings
         */
        void updateSettings();

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
         * @brief saveTrainingSignal
         */
        void saveTrainingSignal(QString);

        /**
         * @brief loadTrainingSignal
         */
        void loadTrainingSignal(QString);

        /**
         * @brief sendReservoirParametersSignal
         */
        void sendReservoirParametersSignal(ReservoirParameters);

        /**
         * @brief sendLanguageParametersSignal
         */
        void sendLanguageParametersSignal(LanguageParameters);

        /**
         * @brief leaveProgram
         */
        void leaveProgram();


        /**
         * @brief sendMatrixXDisplayParameters
         */
        void sendMatrixXDisplayParameters(bool enabled, bool randomSentence, int nbRandomNeurons, int startIdNeurons, int endIdNeurons);

    private :

        int m_sizeDim1Meaning;
        int m_sizeDim2Meaning;


        QFile m_logFile;

        QString m_absolutePath;

        // widgets & ui
        DisplayImageWidget *m_imageDisplay;
        QVector<QCustomPlot*> m_plotListX;
        QVector<QCustomPlot*> m_plotListOutput;
        QVector<QCustomPlot*> m_plotListTrainSentenceInput;
        QVector<QCustomPlot*> m_plotListTrainMeaningInput;
        QVector<QLabel*> m_plotLabelListOutput;
        QVector<QLabel*> m_plotLabelListTrainSentenceInput;



        QVector<QVector<double> > m_allValuesPlot;
        QVector<double> m_allXPlot;
//        QVBoxLayout *m_plotLayout;

//        QCustomPlot *m_plotDisplay;
        Ui::UI_Reservoir* m_uiInterface;   /**< qt main window */

        // threads & workers
        InterfaceWorker  *m_pWInterface;    /**< viewer worker */
        QThread         m_TInterface;    /**< viewer thread */


        int m_nbMaxNeuronsSentenceDisplayed;
        int m_nbSentencesDisplayed;
        QTime m_timerDisplayNeurons;
        QMutex m_neuronDisplayMutex;

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

        /**
         * \brief destructor of InterfaceWorker
         */
        ~InterfaceWorker();

        /**
         * @brief gridSearch
         * @return
         */
        GridSearchQt *gridSearch() const;

        /**
         * @brief model
         * @return
         */
        ModelQt *model();

        /**
         * @brief languageParameters
         * @return
         */
        LanguageParameters languageParameters() const;

    public slots:

        /**
         * @brief updateCorpus
         * @param corpusPath
         */
        void addCorpus(QString corpusPath);

        /**
         * @brief removeCorpus
         * @param index
         */
        void removeCorpus(int index);

        /**
         * @brief updateReservoirParameters
         */
        void updateReservoirParameters(ReservoirParameters newParams);

        /**
         * @brief updateLanguageParameters
         * @param newParams
         */
        void updateLanguageParameters(LanguageParameters newParams);

        /**
         * @brief start
         */
        void start();

        /**
         * @brief stop
         */
        void stop();

        /**
         * @brief saveLastTraining
         */
        void saveLastTraining(QString pathDirectory);

        /**
         * @brief loadTraining
         */
        void loadTraining(QString pathDirectory);

    signals:

        /**
         * @brief lockInterfaceSignal
         */
        void lockInterfaceSignal(bool);

        /**
         * @brief displayValidityOperation
         */
        void displayValidityOperationSignal(bool, int);


        /**
         * @brief endTrainingSignal
         */
        void endTrainingSignal(bool);

        /**
         * @brief startInitDisplaySignal TODO
         */
        void startInitDisplaySignal();


    private :

        int m_nbOfCorpus;



        ReservoirParameters m_reservoirParameters;
        LanguageParameters m_languageParameters;

        ModelQt m_model;
        GridSearchQt *m_gridSearch;

        QStringList m_corpusList;
};


#endif
