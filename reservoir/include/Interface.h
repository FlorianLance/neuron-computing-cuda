
/**
 * \file Interface.h
 * \brief Defines Interface
 * \author Florian Lance
 * \date 01/12/14
 */

#ifndef _INTERFACE_
#define _INTERFACE_

// Qt
#include <QMainWindow>
#include <QThread>
#include <QtGui>

// Qt customplot
#include "qcustomplot.h"

// Ui
#include "../genUI/UI_Interface.h"

// reservoir
#include "GridSearchQtObject.h"


namespace Ui {
    class UI_Reservoir;
}


class InterfaceWorker;


struct ReplayParameters
{

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
    bool m_useLoadedW;
    bool m_useLoadedWIn;
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
         * @brief Add a corpus in the list
         */
        void addCorpus();

        /**
         * @brief Remove a corpus from the list
         */
        void removeCorpus();

        /**
         * @brief Save the last training done
         */
        void saveTraining();

        /**
         * @brief saveReplay
         */
        void saveReplay();

        /**
         * @brief Load a training by picking it in a dialog window
         */
        void loadTraining();

        /**
         * @brief loadWMatrix
         */
        void loadWMatrix();

        /**
         * @brief loadWInMatrix
         */
        void loadWInMatrix();

        /**
         * @brief loadReplay
         */
        void loadReplay();

        /**
         * @brief updateParamters
         */
        void updateReservoirParameters();

        /**
         * @brief updateReservoirParameters
         * @param value
         */
        void updateReservoirParameters(int value);

        /**
         * @brief updateReservoirParameters
         * @param value
         */
        void updateReservoirParameters(double value);

        /**
         * @brief updateReservoirParameters
         * @param value
         */
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
         * @param colorText
         */
        void displayLogInfo(QString info, QColor colorText);

        /**
         * @brief displayOutputMatrix
         * @param output
         * @param sentences
         */
        void displayOutputMatrix(cv::Mat output, Sentences sentences);

        /**
         * @brief displayTrainInputMatrix
         */
        void displayTrainInputMatrix(cv::Mat trainMeaning, cv::Mat trainSentence, Sentences sentences);

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

        /**
         * @brief openCorpus
         */
        void openCorpus(QModelIndex index);

        /**
         * @brief setXTabFocus
         * @param index
         */
        void setXTabFocus(int index);

        /**
         * @brief disableCustomMatrix
         * @param index
         */
        void disableCustomMatrix(int index);

        /**
         * @brief disableTraining
         * @param index
         */
        void disableTraining(int index);

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
         * @brief saveReplaySignal
         */
        void saveReplaySignal(QString);

        /**
         * @brief loadTrainingSignal
         */
        void loadTrainingSignal(QString);

        /**
         * @brief loadWSignal
         */
        void loadWSignal(QString);

        /**
         * @brief loadWSignal
         */
        void loadWInSignal(QString);

        /**
         * @brief loadReplaySignal
         */
        void loadReplaySignal(QString);

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

        // miscellanous
        QString m_absolutePath;         /**< ... */
        QFile m_logFile;                /**< ... */
        QVector<QColor> m_colorsCCW;    /**< colors used for the CCW display */

        // widgets & ui
        //  input
        QVector<QCustomPlot*> m_plotListTrainSentenceInput;     /**< ... */
        QVector<QCustomPlot*> m_plotListTrainMeaningInput;      /**< ... */
        QVector<QLabel*> m_plotLabelListTrainSentenceInput;     /**< ... */
        QVector<QLabel*> m_labelListInputSentences;             /**< labels of the input sentences displayed in the input panel */
        //  output
        QVector<QCustomPlot*> m_plotListTrainSentenceOutput;    /**< ... */
        QVector<QLabel*> m_labelListRetrievedSentences;             /**< labels of the retrieved sentences displayed in the output panel */
        QVector<QLabel*> m_plotLabelListTrainSentenceOutput;    /**< ... */
        Ui::UI_Reservoir* m_uiInterface;                        /**< qt main window */

        //threads / workers
        InterfaceWorker  *m_pWInterface;    /**< viewer worker */
        QThread         m_TInterface;       /**< viewer thread */

        // test
        QVector<QCustomPlot*> m_replayPlotList;
        QVector<double> m_replayData;

        // old
        QVector<QCustomPlot*> m_plotListX;   // ?
        int m_sizeDim1Meaning; // ?
        int m_sizeDim2Meaning; // ?
        QVector<QVector<double> > m_allValuesPlot; // ?
        int m_nbMaxNeuronsSentenceDisplayed; // ?
        int m_nbSentencesDisplayed; // ?
        QTime m_timerDisplayNeurons; // ?
        QMutex m_neuronDisplayMutex; // ?
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
         * @param absolutePath
         */
        InterfaceWorker(QString absolutePath);

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
         * @param pathDirectory
         */
        void saveLastTraining(QString pathDirectory);

        /**
         * @brief saveLastReplay
         * @param pathDirectory
         */
        void saveLastReplay(QString pathDirectory);


        /**
         * @brief loadTraining
         * @param pathDirectory
         */
        void loadTraining(QString pathDirectory);

        /**
         * @brief loadW
         * @param pathDirectory
         */
        void loadW(QString pathDirectory);

        /**
         * @brief loadWIn
         * @param pathDirectory
         */
        void loadWIn(QString pathDirectory);


        /**
         * @brief setLoadedTrainingParameters
         * @param loadedParams
         */
        void setLoadedTrainingParameters(QStringList loadedParams);

        /**
         * @brief setLoadedWParameters
         * @param loadedParams
         */
        void setLoadedWParameters(QStringList loadedParams);

        /**
         * @brief setLoadedWInParameters
         * @param loadedParams
         */
        void setLoadedWInParameters(QStringList loadedParams);

        /**
         * @brief loadReplay
         * @param pathReplay
         */
        void loadReplay(QString pathReplay);

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
         * @brief sendLogInfo
         */
        void sendLogInfo(QString, QColor);

    private :

        int m_nbOfCorpus;

        QString m_absolutePath;

        QStringList m_parametersTrainingLoaded;
        QStringList m_parametersWLoaded;
        QStringList m_parametersWInLoaded;

        ReservoirParameters m_reservoirParameters;
        LanguageParameters m_languageParameters;
        ReplayParameters m_replayParameters;

        ModelQt m_model;
        GridSearchQt *m_gridSearch;

        QStringList m_corpusList;
};



#endif
