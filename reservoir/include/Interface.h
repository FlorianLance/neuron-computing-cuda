
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
 * \brief Qt interface for using the Reservoir model.
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
         * @brief openOnlineDocumentation
         */
        void openOnlineDocumentation();

        /**
         * @brief openAboutWindow
         */
        void openAboutWindow();

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
        QString m_absolutePath;         /**< absolute path used for init dialog boxes */
        QFile m_logFile;                /**< log file, will save all the output displayed in the INFOS part */
        QVector<QColor> m_colorsCCW;    /**< colors used for the CCW display */

        // widgets & ui
        //  input
        QVector<QCustomPlot*> m_plotListTrainSentenceInput;     /**< list of the plots for the sentences input */
        QVector<QCustomPlot*> m_plotListTrainMeaningInput;      /**< list of the plots for the meaning input */
        QVector<QLabel*> m_plotLabelListTrainSentenceInput;     /**< labels of the input meaning displayed in the input panel labels of the input sentences displayed in the input panel */
        QVector<QLabel*> m_labelListInputSentences;             /**< labels of the input sentences displayed in the input panel */
        //  output
        QVector<QCustomPlot*> m_plotListTrainSentenceOutput;    /**< list of the plots for the sentences output */
        QVector<QLabel*> m_labelListRetrievedSentences;         /**< labels of the retrieved sentences displayed in the output panel */
        QVector<QLabel*> m_plotLabelListTrainSentenceOutput;    /**< labels of the CCW displayed in the output panel */
        Ui::UI_Reservoir* m_uiInterface;                        /**< qt main window */

        //threads / workers
        InterfaceWorker  *m_pWInterface;    /**< viewer worker */
        QThread         m_TInterface;       /**< viewer thread */

        // replay
        bool m_replayLoaded;                /**< is the replay loaded ? */
        QColor m_colorLine;                 /**< color of the line */
        QVector<QCustomPlot*> m_plotReplay; /**< plots of the replay */
};



#endif
