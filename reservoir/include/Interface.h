
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
#include "InterfaceWorker.h"


namespace Ui {
    class UI_Reservoir;
}


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
//        void closeEvent(QCloseEvent *event);

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
         * @brief updateReplayParameters
         */
        void updateReplayParameters();

        /**
         * @brief updateReplayParameters
         * @param value
         */
        void updateReplayParameters(int value);

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
        void displayCurrentParameters(ModelParameters params);

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

//        /**
//         * @brief displayXMatrix
//         * @param values
//         * @param currentSentenceId
//         * @param nbSentences
//         */
//        void displayXMatrix(QVector<QVector<double> > *values, int currentSentenceId, int nbSentences);


//        /**
//         * @brief initPlot
//         * @param nbCurves
//         * @param sizeDim1Meaning
//         * @param sizeDim2Meaning
//         * @param name
//         */
//        void initPlot(int nbCurves, int sizeDim1Meaning, int sizeDim2Meaning, QString name);

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

        /**
         * @brief resetLoadingBar
         */
        void resetLoadingBar();

        /**
         * @brief replayLoaded
         */
        void replayLoaded();

        /**
         * @brief updateDisplayReplay
         * @param data
         * @param neuronsId
         * @param sentencesId
         */
        void updateDisplayReplay(QVector<QVector<double> > data, QVector<int> neuronsId, QVector<int> sentencesId);

        /**
         * @brief updateColorCCW
         * @param params
         */
        void updateColorCCW(LanguageParameters params);

        /**
         * @brief setColorLine
         */
        void setColorLine();

        /**
         * @brief saveXPlot
         */
        void saveXPlot();

        /**
         * @brief saveOutput
         */
        void saveOutput();

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
         * @brief sendReplayParametersSignal
         */
        void sendReplayParametersSignal(ReplayParameters);

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

        // replay
        bool m_replayLoaded;
        QColor m_colorLine;
        QVector<QCustomPlot*> m_plotReplay;

        // old
//        QVector<QCustomPlot*> m_plotListX;   // ?
        int m_sizeDim1Meaning; // ?
        int m_sizeDim2Meaning; // ?
        QVector<QVector<double> > m_allValuesPlot; // ?
        int m_nbMaxNeuronsSentenceDisplayed; // ?
        int m_nbSentencesDisplayed; // ?
        QTime m_timerDisplayNeurons; // ?
        QMutex m_neuronDisplayMutex; // ?
};



#endif
