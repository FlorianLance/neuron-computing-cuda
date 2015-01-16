

/**
 * \file Interface.cpp
 * \brief Defines Interface
 * \author Florian Lance
 * \date 01/12/14
 */

#include "Interface.h"
#include "../moc/moc_Interface.cpp"

#include <time.h>


int main(int argc, char* argv[])
{
    srand(1);
    culaWarmup(1);

    QApplication l_oApp(argc, argv);
    Interface l_oViewerInterface(&l_oApp);
    l_oViewerInterface.move(0,0);
    l_oViewerInterface.show();

    return l_oApp.exec();
}


Interface::Interface(QApplication *parent) : m_uiInterface(new Ui::UI_Reservoir), m_replayLoaded(false), m_colorLine(QColor(Qt::blue))
{    
    // set absolute path
        m_absolutePath = QDir::currentPath() + "/";

    // create folders
        QVector<QDir> l_dirs;
        l_dirs.push_back(QDir(m_absolutePath     + "../data/input/Corpus"));
        l_dirs.push_back(QDir(m_absolutePath     + "../data/input/Settings"));
        l_dirs.push_back(QDir(m_absolutePath     + "../data/input/Matrices/W"));
        l_dirs.push_back(QDir(m_absolutePath     + "../data/input/Matrices/WIn"));
        l_dirs.push_back(QDir(m_absolutePath     + "../data/input/Settings"));
        l_dirs.push_back(QDir(m_absolutePath     + "../data/Results/raw"));
        l_dirs.push_back(QDir(m_absolutePath     + "../data/training"));
        l_dirs.push_back(QDir(m_absolutePath     + "../data/replay"));
        l_dirs.push_back(QDir(m_absolutePath     + "../data/images"));
        l_dirs.push_back(QDir(m_absolutePath     + "../log"));

        for(int ii = 0; ii < l_dirs.size(); ++ii)
        {
            if(!l_dirs[ii].exists())
            {
                l_dirs[ii].mkpath(".");
            }
        }

    // init main widget
        m_uiInterface->setupUi(this);

    // set icon/title
        this->setWindowTitle(QString("RESERVOIR COMPUTING - CUDA"));
        this->setWindowIcon(QIcon(m_absolutePath + "../data/images/iconeN.png"));

    // create log file
        QDateTime l_dateTime;
        l_dateTime = l_dateTime.currentDateTime();
        QDate l_date = l_dateTime.date();
        QTime l_time = QTime::currentTime();
        QString l_nameLogFile = m_absolutePath + "../log/logs_reservoir_interface_" + l_date.toString("dd_MM_yyyy") + "_" +  QString::number(l_time.hour()) + "h" + QString::number(l_time.minute()) + "m" + QString::number(l_time.second()) + "s.txt";
        m_logFile.setFileName(l_nameLogFile);
        if (!m_logFile.open(QIODevice::WriteOnly | QIODevice::Text))
        {
            qWarning() << "Cannot write log file. Start of the path must be ./dist, and ../log must exist. ";
        }

    // init worker
        m_pWInterface = new InterfaceWorker(m_absolutePath);

    // init connections
        GridSearch *l_gridSearchQt = m_pWInterface->gridSearch();
        Model *l_model = m_pWInterface->model();

        QObject::connect(this, SIGNAL(leaveProgram()), parent, SLOT(quit()));
        // listwidgets
        QObject::connect(m_uiInterface->lwCorpus,  SIGNAL(doubleClicked(QModelIndex)), this, SLOT(openCorpus(QModelIndex)));
        // push button
        QObject::connect(m_uiInterface->pbStart,                SIGNAL(clicked()), m_pWInterface,   SLOT(start()));        
        QObject::connect(m_uiInterface->pbStop,                 SIGNAL(clicked()), m_pWInterface,   SLOT(stop()));
        QObject::connect(m_uiInterface->pbStartReplay,          SIGNAL(clicked()), m_pWInterface,   SLOT(startReplay()));                
        QObject::connect(m_uiInterface->pbColor,                SIGNAL(clicked()), this,            SLOT(setColorLine()));
        QObject::connect(m_uiInterface->pbSaveX,                SIGNAL(clicked()), this,            SLOT(saveXPlot()));
        QObject::connect(m_uiInterface->pbSaveOutput,           SIGNAL(clicked()), this,            SLOT(saveOutput()));
        QObject::connect(m_uiInterface->pbStart,                SIGNAL(clicked()), this,            SLOT(resetLoadingBar()));
        QObject::connect(m_uiInterface->pbAddCorpus,            SIGNAL(clicked()), this,            SLOT(addCorpus()));
        QObject::connect(m_uiInterface->pbRemoveCorpus,         SIGNAL(clicked()), this,            SLOT(removeCorpus()));
        QObject::connect(m_uiInterface->pbSaveLastTrainingFile, SIGNAL(clicked()), this,            SLOT(saveTraining()));
        QObject::connect(m_uiInterface->pbSaveReplay,           SIGNAL(clicked()), this,            SLOT(saveReplay()));
        QObject::connect(m_uiInterface->pbClearResults,         SIGNAL(clicked()), this,            SLOT(cleanResultsDisplay()));
        QObject::connect(m_uiInterface->pbLoadTrainingFile,     SIGNAL(clicked()), this,            SLOT(loadTraining()));
        QObject::connect(m_uiInterface->pbLoadW,                SIGNAL(clicked()), this,            SLOT(loadWMatrix()));
        QObject::connect(m_uiInterface->pbLoadWin,              SIGNAL(clicked()), this,            SLOT(loadWInMatrix()));
        QObject::connect(m_uiInterface->pbOpenSelectedCorpus,   SIGNAL(clicked()), this,            SLOT(openCorpus()));
        QObject::connect(m_uiInterface->pbReloadSelectedCorpus, SIGNAL(clicked()), this,            SLOT(reloadCorpus()));
        QObject::connect(m_uiInterface->pbLoadSettings,         SIGNAL(clicked()), this,            SLOT(loadSettings()));
        QObject::connect(m_uiInterface->pbLoadReplay,           SIGNAL(clicked()), this,            SLOT(loadReplay()));
        // radio button
        QObject::connect(m_uiInterface->rbTrain,                SIGNAL(clicked()), SLOT(updateReservoirParameters()));
        QObject::connect(m_uiInterface->rbTest,                 SIGNAL(clicked()), SLOT(updateReservoirParameters()));
        QObject::connect(m_uiInterface->rbBoth,                 SIGNAL(clicked()), SLOT(updateReservoirParameters()));
        QObject::connect(m_uiInterface->rbRangeNeurons,         SIGNAL(clicked()), SLOT(updateReplayParameters()));
        QObject::connect(m_uiInterface->rbRandomNeurons,        SIGNAL(clicked()), SLOT(updateReplayParameters()));
        QObject::connect(m_uiInterface->rbRangeSentences,       SIGNAL(clicked()), SLOT(updateReplayParameters()));
        QObject::connect(m_uiInterface->rbRandomSentences,      SIGNAL(clicked()), SLOT(updateReplayParameters()));
        QObject::connect(m_uiInterface->rbLastTrainingReplay,   SIGNAL(clicked()), SLOT(updateReplayParameters()));
        QObject::connect(m_uiInterface->rbLoadReplay,           SIGNAL(clicked()), SLOT(updateReplayParameters()));
        // tab
        QObject::connect(m_uiInterface->twSettings,  SIGNAL(currentChanged(int)), this , SLOT(setXTabFocus(int)));
        // spinbox
        QObject::connect(m_uiInterface->sbStartNeurons,         SIGNAL(valueChanged(int)),    SLOT(updateReservoirParameters(int)));
        QObject::connect(m_uiInterface->sbStartLeakRate,        SIGNAL(valueChanged(double)), SLOT(updateReservoirParameters(double)));
        QObject::connect(m_uiInterface->sbStartIS,              SIGNAL(valueChanged(double)), SLOT(updateReservoirParameters(double)));
        QObject::connect(m_uiInterface->sbStartSpectralRadius,  SIGNAL(valueChanged(double)), SLOT(updateReservoirParameters(double)));
        QObject::connect(m_uiInterface->sbStartRidge,           SIGNAL(valueChanged(double)), SLOT(updateReservoirParameters(double)));
        QObject::connect(m_uiInterface->sbStartSparcity,        SIGNAL(valueChanged(double)), SLOT(updateReservoirParameters(double)));       
        QObject::connect(m_uiInterface->sbEndNeurons,           SIGNAL(valueChanged(int)),    SLOT(updateReservoirParameters(int)));
        QObject::connect(m_uiInterface->sbEndLeakRate,          SIGNAL(valueChanged(double)), SLOT(updateReservoirParameters(double)));
        QObject::connect(m_uiInterface->sbEndIS,                SIGNAL(valueChanged(double)), SLOT(updateReservoirParameters(double)));
        QObject::connect(m_uiInterface->sbEndSpectralRadius,    SIGNAL(valueChanged(double)), SLOT(updateReservoirParameters(double)));
        QObject::connect(m_uiInterface->sbEndRidge,             SIGNAL(valueChanged(double)), SLOT(updateReservoirParameters(double)));
        QObject::connect(m_uiInterface->sbEndSparcity,          SIGNAL(valueChanged(double)), SLOT(updateReservoirParameters(double)));
        QObject::connect(m_uiInterface->sbNbUseNeurons,         SIGNAL(valueChanged(int)),    SLOT(updateReservoirParameters(int)));
        QObject::connect(m_uiInterface->sbNbUseNeurons,         SIGNAL(valueChanged(int)),    SLOT(updateReservoirParameters(int)));
        QObject::connect(m_uiInterface->sbNbUseLeakRate,        SIGNAL(valueChanged(int)),    SLOT(updateReservoirParameters(int)));
        QObject::connect(m_uiInterface->sbNbUseISS,             SIGNAL(valueChanged(int)),    SLOT(updateReservoirParameters(int)));
        QObject::connect(m_uiInterface->sbNbUseSpectralRadius,  SIGNAL(valueChanged(int)),    SLOT(updateReservoirParameters(int)));
        QObject::connect(m_uiInterface->sbNbUseRidge,           SIGNAL(valueChanged(int)),    SLOT(updateReservoirParameters(int)));
        QObject::connect(m_uiInterface->sbNbUseSparcity,        SIGNAL(valueChanged(int)),    SLOT(updateReservoirParameters(int)));
        QObject::connect(m_uiInterface->sbStartRangeNeuronDisplay,      SIGNAL(valueChanged(int)),    SLOT(updateReplayParameters(int)));
        QObject::connect(m_uiInterface->sbEndRangeNeuronDisplay,        SIGNAL(valueChanged(int)),    SLOT(updateReplayParameters(int)));
        QObject::connect(m_uiInterface->sbNbRandomNeurons,              SIGNAL(valueChanged(int)),    SLOT(updateReplayParameters(int)));
        QObject::connect(m_uiInterface->sbStartRangeSentencesDisplay,   SIGNAL(valueChanged(int)),    SLOT(updateReplayParameters(int)));
        QObject::connect(m_uiInterface->sbEndRangeSentencesDisplay,     SIGNAL(valueChanged(int)),    SLOT(updateReplayParameters(int)));
        QObject::connect(m_uiInterface->sbNbRandomSentences,            SIGNAL(valueChanged(int)),    SLOT(updateReplayParameters(int)));
        // checkbox
        QObject::connect(m_uiInterface->cbNeurons,              SIGNAL(stateChanged(int)), SLOT(updateReservoirParameters(int)));
        QObject::connect(m_uiInterface->cbLeakRate,             SIGNAL(stateChanged(int)), SLOT(updateReservoirParameters(int)));
        QObject::connect(m_uiInterface->cbIS,                   SIGNAL(stateChanged(int)), SLOT(updateReservoirParameters(int)));
        QObject::connect(m_uiInterface->cbSpectralRadius,       SIGNAL(stateChanged(int)), SLOT(updateReservoirParameters(int)));
        QObject::connect(m_uiInterface->cbRidge,                SIGNAL(stateChanged(int)), SLOT(updateReservoirParameters(int)));
        QObject::connect(m_uiInterface->cbSparcity,             SIGNAL(stateChanged(int)), SLOT(updateReservoirParameters(int)));
        QObject::connect(m_uiInterface->cbTrainingFile,         SIGNAL(stateChanged(int)), SLOT(updateReservoirParameters(int)));
        QObject::connect(m_uiInterface->cbTrainingFile,         SIGNAL(stateChanged(int)), SLOT(disableCustomMatrix(int)));
        QObject::connect(m_uiInterface->cbW,                    SIGNAL(stateChanged(int)), SLOT(updateReservoirParameters(int)));
        QObject::connect(m_uiInterface->cbW,                    SIGNAL(stateChanged(int)), SLOT(disableTraining(int)));
        QObject::connect(m_uiInterface->cbWIn,                  SIGNAL(stateChanged(int)), SLOT(updateReservoirParameters(int)));
        QObject::connect(m_uiInterface->cbWIn,                  SIGNAL(stateChanged(int)), SLOT(disableTraining(int)));
        QObject::connect(m_uiInterface->cbOnlyStartValue,       SIGNAL(stateChanged(int)), SLOT(updateReservoirParameters(int)));
        QObject::connect(m_uiInterface->cbEnableGPU,            SIGNAL(stateChanged(int)), SLOT(updateReservoirParameters(int)));
        QObject::connect(m_uiInterface->cbEnableMultiThread,    SIGNAL(toggled(bool)), l_model->reservoir(), SLOT(enableMaxOmpThreadNumber(bool)));
        // lineedit
        QObject::connect(m_uiInterface->leNeuronsOperation,         SIGNAL(editingFinished()), SLOT(updateReservoirParameters()));
        QObject::connect(m_uiInterface->leLeakRateOperation,        SIGNAL(editingFinished()), SLOT(updateReservoirParameters()));
        QObject::connect(m_uiInterface->leISOperation,              SIGNAL(editingFinished()), SLOT(updateReservoirParameters()));
        QObject::connect(m_uiInterface->leSpectralRadiusOperation,  SIGNAL(editingFinished()), SLOT(updateReservoirParameters()));
        QObject::connect(m_uiInterface->leRidgeOperation,           SIGNAL(editingFinished()), SLOT(updateReservoirParameters()));
        QObject::connect(m_uiInterface->leSparcityOperation,        SIGNAL(editingFinished()), SLOT(updateReservoirParameters()));
        QObject::connect(m_uiInterface->leDisplayCCW,               SIGNAL(editingFinished()), SLOT(updateSettings()));
        QObject::connect(m_uiInterface->leDisplayStructure,         SIGNAL(editingFinished()), SLOT(updateSettings()));
        // lock
        QObject::connect(m_pWInterface, SIGNAL(lockInterfaceSignal(bool)), this, SLOT(lockInterface(bool)));
        // this
        QObject::connect(this, SIGNAL(sendMatrixXDisplayParameters(bool,bool,int,int,int)), l_model->reservoir(),   SLOT(updateMatrixXDisplayParameters(bool,bool,int,int,int)));
        QObject::connect(this, SIGNAL(addCorpusSignal(QString)),                            m_pWInterface,          SLOT(addCorpus(QString)));
        QObject::connect(this, SIGNAL(removeCorpusSignal(int)),                             m_pWInterface,          SLOT(removeCorpus(int)));
        QObject::connect(this, SIGNAL(sendReservoirParametersSignal(ReservoirParameters)),  m_pWInterface,          SLOT(updateReservoirParameters(ReservoirParameters)));
        QObject::connect(this, SIGNAL(sendLanguageParametersSignal(LanguageParameters)),    m_pWInterface,          SLOT(updateLanguageParameters(LanguageParameters)));
        QObject::connect(this, SIGNAL(sendReplayParametersSignal(ReplayParameters)),        m_pWInterface,          SLOT(updateReplayParameters(ReplayParameters)));
        QObject::connect(this, SIGNAL(saveTrainingSignal(QString)),                         m_pWInterface,          SLOT(saveLastTraining(QString)));
        QObject::connect(this, SIGNAL(saveReplaySignal(QString)),                           m_pWInterface,          SLOT(saveLastReplay(QString)));
        QObject::connect(this, SIGNAL(loadTrainingSignal(QString)),                         m_pWInterface,          SLOT(loadTraining(QString)));
        QObject::connect(this, SIGNAL(loadWSignal(QString)),                                m_pWInterface,          SLOT(loadW(QString)));
        QObject::connect(this, SIGNAL(loadWInSignal(QString)),                              m_pWInterface,          SLOT(loadWIn(QString)));
        QObject::connect(this, SIGNAL(loadReplaySignal(QString)),                           m_pWInterface,          SLOT(loadReplay(QString)));
        // gridsearch
        QObject::connect(l_gridSearchQt, SIGNAL(sendCurrentParametersSignal(ModelParameters)),          this, SLOT(displayCurrentParameters(ModelParameters)));
        QObject::connect(l_gridSearchQt, SIGNAL(sendResultsReservoirSignal(ResultsDisplayReservoir)),   this, SLOT(displayCurrentResults(ResultsDisplayReservoir)));
        QObject::connect(l_gridSearchQt, SIGNAL(sendLogInfo(QString, QColor)),                          this, SLOT(displayLogInfo(QString, QColor)));
        // model
        QObject::connect(l_model,SIGNAL(sendLogInfo(QString, QColor)),                          this,  SLOT(displayLogInfo(QString, QColor)));
        QObject::connect(l_model,SIGNAL(sendOutputMatrix(cv::Mat, Sentences)),                  this,  SLOT(displayOutputMatrix(cv::Mat, Sentences)));
        QObject::connect(l_model,SIGNAL(sendTrainInputMatrixSignal(cv::Mat,cv::Mat,Sentences)), this,  SLOT(displayTrainInputMatrix(cv::Mat,cv::Mat, Sentences)));
        // reservoir
        QObject::connect(l_model->reservoir(),  SIGNAL(sendLogInfo(QString, QColor)),               this,           SLOT(displayLogInfo(QString, QColor)));
        QObject::connect(l_model->reservoir(),  SIGNAL(sendComputingState(int,int,QString)),        this,           SLOT(updateProgressBar(int, int, QString)));
        QObject::connect(l_model->reservoir(),  SIGNAL(sendLoadedTrainingParameters(QStringList)),  m_pWInterface,  SLOT(setLoadedTrainingParameters(QStringList)));
        QObject::connect(l_model->reservoir(),  SIGNAL(sendLoadedWParameters(QStringList)),         m_pWInterface,  SLOT(setLoadedWParameters(QStringList)));
        QObject::connect(l_model->reservoir(),  SIGNAL(sendLoadedWInParameters(QStringList)),       m_pWInterface,  SLOT(setLoadedWInParameters(QStringList)));
        // worker
        QObject::connect(m_pWInterface, SIGNAL(sendLogInfo(QString, QColor)),               this,                                   SLOT(displayLogInfo(QString, QColor)));
        QObject::connect(m_pWInterface, SIGNAL(displayValidityOperationSignal(bool, int)),  this,                                   SLOT(displayValidityOperation(bool, int)));
        QObject::connect(m_pWInterface, SIGNAL(endTrainingSignal(bool)),                    m_uiInterface->pbSaveLastTrainingFile,  SLOT(setEnabled(bool)));
        QObject::connect(m_pWInterface, SIGNAL(endTrainingSignal(bool)),                    m_uiInterface->pbSaveReplay,            SLOT(setEnabled(bool)));
        QObject::connect(m_pWInterface, SIGNAL(endTrainingSignal(bool)),                    m_uiInterface->rbLastTrainingReplay,    SLOT(setEnabled(bool)));
        QObject::connect(m_pWInterface, SIGNAL(replayLoaded()),                             this,                                   SLOT(replayLoaded()));
        QObject::connect(m_pWInterface, SIGNAL(sendReplayData(QVector<QVector<double> > , QVector<int>, QVector<int>)), this,       SLOT(updateDisplayReplay(QVector<QVector<double> > , QVector<int>, QVector<int>)));

    // init widgets
        // pushbuttons
        m_uiInterface->pbComputing->setRange(0,100);
        m_uiInterface->pbComputing->setValue(0);
        m_uiInterface->pbStop->setVisible(false);
        m_uiInterface->pbAddCorpus->setStyleSheet("* { font-weight: bold }");
        m_uiInterface->pbStart->setStyleSheet("* { font-weight: bold }");
        m_uiInterface->pbLoadSettings->setStyleSheet("* { font-weight: bold }");
        m_uiInterface->pbStartReplay->setStyleSheet("* { font-weight: bold }");
        m_uiInterface->pbLoadReplay->setStyleSheet("* { font-weight: bold }");
        // scroll
        m_uiInterface->scrollAreaPlotX->setWidgetResizable(true);
        m_uiInterface->scrollAreaPlotOutput->setWidgetResizable(true);
        m_uiInterface->scrollAreaPlotInput->setWidgetResizable(true);
        // tab widgets
        m_uiInterface->twDisplay->setTabEnabled(1, false);
        // spin boxes
        m_uiInterface->sbStartSpectralRadius->setRange(0,10000);
        m_uiInterface->sbEndSpectralRadius->setRange(0,10000);
        // interface
        setStyleSheet("QGroupBox { color: blue; } ");        


    // init thread
        m_pWInterface->moveToThread(&m_TInterface);
        m_TInterface.start();

    // update worker parameters with defaults values
        updateReservoirParameters();
        updateReplayParameters();
}


Interface::~Interface()
{
    m_TInterface.quit();
    m_TInterface.wait();

    delete m_pWInterface;
}



void Interface::addCorpus()
{        
    QString l_sPathCorpus = QFileDialog::getOpenFileName(this, "Load corpus file", m_absolutePath + "../data/input/Corpus", "Corpus file (*.txt)");

    if(l_sPathCorpus.size() > 0)
    {        
        m_uiInterface->lwCorpus->addItem(l_sPathCorpus);

        // send item
            emit addCorpusSignal(l_sPathCorpus);

        // unlock ui
            m_uiInterface->pbRemoveCorpus->setEnabled(true);
            m_uiInterface->pbOpenSelectedCorpus->setEnabled(true);
            m_uiInterface->pbReloadSelectedCorpus->setEnabled(true);

        m_uiInterface->lwCorpus->setCurrentRow(m_uiInterface->lwCorpus->count()-1);

        // unlock start
            if(m_uiInterface->leDisplayCCW->text().size() > 0)
            {
                m_uiInterface->pbStart->setEnabled(true);
            }
    }
}

void Interface::removeCorpus()
{
    int l_currentIndex = m_uiInterface->lwCorpus->currentRow();

    if(l_currentIndex >= 0)
    {
        delete m_uiInterface->lwCorpus->takeItem(l_currentIndex);
    }

    // remove item
        emit removeCorpusSignal(l_currentIndex);


    if(m_uiInterface->lwCorpus->count() == 0)
    {
        // lock ui
            m_uiInterface->pbRemoveCorpus->setEnabled(false);
            m_uiInterface->pbOpenSelectedCorpus->setEnabled(false);
            m_uiInterface->pbReloadSelectedCorpus->setEnabled(false);
            m_uiInterface->pbStart->setEnabled(false);
    }
}


void Interface::saveTraining()
{
    QString l_sPathTrainingFile = QFileDialog::getExistingDirectory(this, "Select directory", m_absolutePath + "../data/training");

    if(l_sPathTrainingFile.size() == 0)
    {
        return;
    }

    // send directory path
    emit saveTrainingSignal(l_sPathTrainingFile);
}

void Interface::saveReplay()
{
    QString l_sPathReplay = QFileDialog::getExistingDirectory(this, "Select directory", m_absolutePath + "../data/replay");

    if(l_sPathReplay.size() == 0)
    {
        return;
    }

    // send directory path
    emit saveReplaySignal(l_sPathReplay);
}



void Interface::loadTraining()
{
    QString l_sPathTrainingFile = QFileDialog::getExistingDirectory(this, "Select directory", m_absolutePath + "../data/training");

    if(l_sPathTrainingFile.size() == 0 )
    {
        return;
    }

    QFile l_fileW(l_sPathTrainingFile    + "/w.txt");
    QFile l_fileWin(l_sPathTrainingFile  + "/wIn.txt");
    QFile l_fileWOut(l_sPathTrainingFile + "/wOut.txt");
    QFile l_fileParam(l_sPathTrainingFile + "/param.txt");

    QPalette l_palette;
    if(l_fileW.exists() && l_fileWin.exists() && l_fileWOut.exists() && l_fileParam.exists())
    {
        // send directory path
        l_palette.setColor(QPalette::Text,Qt::black);
        m_uiInterface->leCurrentTrainingFile->setText(l_sPathTrainingFile);
        m_uiInterface->cbTrainingFile->setEnabled(true);

        emit loadTrainingSignal(l_sPathTrainingFile);
    }
    else
    {
        l_palette.setColor(QPalette::Text,Qt::red);

        QString l_message;


        if(!l_fileWin.exists())
        {
            m_uiInterface->leCurrentTrainingFile->setText("Training matrices not found in the directory...");
            l_message = "Training matrices not found, loading not done. \n";
        }
        else
        {
            m_uiInterface->leCurrentTrainingFile->setText("Parameter file not found in the directory...");
            l_message = "Parameter file not found, loading not done. \n";
        }


        std::cerr << l_message.toStdString() << std::endl;
        displayLogInfo(l_message, QColor(Qt::red));
    }

    m_uiInterface->leCurrentTrainingFile->setPalette(l_palette);
}

void Interface::loadWMatrix()
{
    QString l_sPathWFile = QFileDialog::getExistingDirectory(this, "Select directory", m_absolutePath + "../data/input/Matrices/W");

    if(l_sPathWFile.size() == 0 )
    {
        return;
    }

    QFile l_fileW(l_sPathWFile    + "/w.txt");
    QFile l_fileParam(l_sPathWFile + "/param.txt");

    QPalette l_palette;
    if(l_fileW.exists() && l_fileParam.exists())
    {
        // send directory path
        l_palette.setColor(QPalette::Text,Qt::black);
        m_uiInterface->leW->setText(l_sPathWFile);
        m_uiInterface->cbW->setEnabled(true);

        emit loadWSignal(l_sPathWFile);
    }
    else
    {
        l_palette.setColor(QPalette::Text,Qt::red);

        QString l_message;


        if(!l_fileW.exists())
        {
            m_uiInterface->leW->setText("W matrice not found in the directory...");
            l_message = "W matrice not found, loading not done. \n";
        }
        else
        {
            m_uiInterface->leW->setText("Parameter file not found in the directory...");
            l_message = "Parameter file not found, loading not done. \n";
        }

        std::cerr << l_message.toStdString() << std::endl;
        displayLogInfo(l_message, QColor(Qt::red));
    }

    m_uiInterface->leW->setPalette(l_palette);
}


void Interface::loadWInMatrix()
{
    QString l_sPathWInFile = QFileDialog::getExistingDirectory(this, "Select directory", m_absolutePath + "../data/input/Matrices/WIn");

    if(l_sPathWInFile.size() == 0 )
    {
        return;
    }

    QFile l_fileWIn(l_sPathWInFile    + "/wIn.txt");
    QFile l_fileParam(l_sPathWInFile + "/param.txt");

    QPalette l_palette;
    if(l_fileWIn.exists() && l_fileParam.exists())
    {
        // send directory path
        l_palette.setColor(QPalette::Text,Qt::black);
        m_uiInterface->leWIn->setText(l_sPathWInFile);
        m_uiInterface->cbWIn->setEnabled(true);

        emit loadWInSignal(l_sPathWInFile);
    }
    else
    {
        l_palette.setColor(QPalette::Text,Qt::red);

        QString l_message;


        if(!l_fileWIn.exists())
        {
            m_uiInterface->leWIn->setText("WIn matrice not found in the directory...");
            l_message = "WIn matrice not found, loading not done. \n";
        }
        else
        {
            m_uiInterface->leWIn->setText("Parameter file not found in the directory...");
            l_message = "Parameter file not found, loading not done. \n";
        }

        std::cerr << l_message.toStdString() << std::endl;
        displayLogInfo(l_message, QColor(Qt::red));
    }

    m_uiInterface->leWIn->setPalette(l_palette);        
}

void Interface::loadReplay()
{
    QString l_pathReplay = QFileDialog::getExistingDirectory(this, "Select directory", m_absolutePath + "../data/replay");

    if(l_pathReplay.size() == 0 )
    {
        return;
    }

    QFile l_fileReplay(l_pathReplay    + "/xTot.txt");

    QPalette l_palette;
    if(l_fileReplay.exists())
    {
        l_palette.setColor(QPalette::Text,Qt::black);
        m_uiInterface->lePathReplay->setText(l_pathReplay);
        m_uiInterface->twSettings->setEnabled(false);
        m_uiInterface->pbComputing->setValue(0);
        m_uiInterface->laStateComputing->setText("Loading replay...");
        emit loadReplaySignal(l_pathReplay);

        m_replayLoaded = true;
    }
    else
    {
        l_palette.setColor(QPalette::Text,Qt::red);

        m_uiInterface->lePathReplay->setText("xTot matrice not found in the directory...");
        QString l_message = "xTot matrice not found, loading replay not done. \n";

        std::cerr << l_message.toStdString() << std::endl;
        displayLogInfo(l_message, QColor(Qt::red));
    }

    m_uiInterface->lePathReplay->setPalette(l_palette);
    updateReplayParameters();
}

void Interface::updateReservoirParameters(int value)
{
    updateReservoirParameters();
}

void Interface::updateReservoirParameters(double value)
{
    updateReservoirParameters();
}

void Interface::updateReservoirParameters(QString value)
{
    updateReservoirParameters();
}

void Interface::updateReplayParameters(int value)
{
    updateReplayParameters();
}


void Interface::updateReplayParameters()
{
    ReplayParameters l_params;
    l_params.m_useLastTraining      = m_uiInterface->rbLastTrainingReplay->isChecked();
    l_params.m_randomNeurons        = m_uiInterface->rbRandomNeurons->isChecked();
    l_params.m_randomSentence       = m_uiInterface->rbRandomSentences->isChecked();
    l_params.m_rangeNeuronsStart    = m_uiInterface->sbStartRangeNeuronDisplay->value();
    l_params.m_rangeNeuronsEnd      = m_uiInterface->sbEndRangeNeuronDisplay->value();
    l_params.m_rangeSentencesStart  = m_uiInterface->sbStartRangeSentencesDisplay->value();
    l_params.m_rangeSentencesEnd    = m_uiInterface->sbEndRangeSentencesDisplay->value();
    l_params.m_randomNeuronsNumber  = m_uiInterface->sbNbRandomNeurons->value();
    l_params.m_randomSentencesNumber= m_uiInterface->sbNbRandomSentences->value();

    if(m_uiInterface->rbLastTrainingReplay->isChecked())
    {
        m_uiInterface->pbStartReplay->setEnabled(true);
    }

    if(m_uiInterface->rbLoadReplay->isChecked())
    {
        m_uiInterface->pbStartReplay->setEnabled(m_replayLoaded);
    }


    emit sendReplayParametersSignal(l_params);
}

void Interface::updateReservoirParameters()
{
    ReservoirParameters l_params;

    l_params.m_useCuda                  = m_uiInterface->cbEnableGPU->isChecked();
    l_params.m_useLoadedTraining        = m_uiInterface->cbTrainingFile->isChecked();
    l_params.m_useLoadedW               = m_uiInterface->cbW->isChecked();
    l_params.m_useLoadedWIn             = m_uiInterface->cbWIn->isChecked();
    l_params.m_useOnlyStartValue        = m_uiInterface->cbOnlyStartValue->isChecked();

    l_params.m_neuronsStart             = m_uiInterface->sbStartNeurons->value();
    l_params.m_leakRateStart            = m_uiInterface->sbStartLeakRate->value();
    l_params.m_issStart                 = m_uiInterface->sbStartIS->value();
    l_params.m_spectralRadiusStart      = m_uiInterface->sbStartSpectralRadius->value();
    l_params.m_ridgeStart               = m_uiInterface->sbStartRidge->value();
    l_params.m_sparcityStart            = m_uiInterface->sbStartSparcity->value();

    l_params.m_neuronsEnd               = m_uiInterface->sbEndNeurons->value();
    l_params.m_leakRateEnd              = m_uiInterface->sbEndLeakRate->value();
    l_params.m_issEnd                   = m_uiInterface->sbEndIS->value();
    l_params.m_spectralRadiusEnd        = m_uiInterface->sbEndSpectralRadius->value();
    l_params.m_ridgeEnd                 = m_uiInterface->sbEndRidge->value();
    l_params.m_sparcityEnd              = m_uiInterface->sbEndSparcity->value();

    l_params.m_neuronsNbOfUses          = m_uiInterface->sbNbUseNeurons->value();
    l_params.m_leakRateNbOfUses         = m_uiInterface->sbNbUseLeakRate->value();
    l_params.m_issNbOfUses              = m_uiInterface->sbNbUseISS->value();
    l_params.m_spectralRadiusNbOfUses   = m_uiInterface->sbNbUseSpectralRadius->value();
    l_params.m_ridgeNbOfUses            = m_uiInterface->sbNbUseRidge->value();
    l_params.m_sparcityNbOfUses         = m_uiInterface->sbNbUseSparcity->value();

    l_params.m_neuronsEnabled           = m_uiInterface->cbNeurons->isChecked();
    l_params.m_leakRateEnabled          = m_uiInterface->cbLeakRate->isChecked();
    l_params.m_issEnabled               = m_uiInterface->cbIS->isChecked();
    l_params.m_spectralRadiusEnabled    = m_uiInterface->cbSpectralRadius->isChecked();
    l_params.m_ridgeEnabled             = m_uiInterface->cbRidge->isChecked();
    l_params.m_sparcityEnabled          = m_uiInterface->cbSparcity->isChecked();

    l_params.m_neuronsOperation         = m_uiInterface->leNeuronsOperation->text();
    l_params.m_leakRateOperation        = m_uiInterface->leLeakRateOperation->text();
    l_params.m_issOperation             = m_uiInterface->leISOperation->text();
    l_params.m_spectralRadiusOperation  = m_uiInterface->leSpectralRadiusOperation->text();
    l_params.m_ridgeOperation           = m_uiInterface->leRidgeOperation->text();
    l_params.m_sparcityOperation        = m_uiInterface->leSparcityOperation->text();

    // disable parameters interface if a training file is used
        if(m_uiInterface->cbTrainingFile->isChecked() || m_uiInterface->cbW->isChecked() || m_uiInterface->cbWIn->isChecked())
        {
            m_uiInterface->gbReservoirParameters->setEnabled(false);
        }
        else
        {
            m_uiInterface->gbReservoirParameters->setEnabled(true);
        }


    if(m_uiInterface->rbTrain->isChecked())
    {
        l_params.m_action = TRAINING_RES;
    }
    else if(m_uiInterface->rbTest->isChecked())
    {
        l_params.m_action = TEST_RES;
    }
    else
    {
        l_params.m_action = BOTH_RES;
    }

    emit sendReservoirParametersSignal(l_params);
}

void Interface::lockInterface(bool lock)
{
    m_uiInterface->twSettings->setEnabled(!lock);
}

void Interface::displayValidityOperation(bool operationValid, int indexParameter)
{
    QPalette l_palette;

    if(operationValid)
    {
        l_palette.setColor(QPalette::Text,Qt::green);
    }
    else
    {
        l_palette.setColor(QPalette::Text,Qt::red);
    }

    switch(indexParameter)
    {
        case GridSearch::NEURONS_NB :
            m_uiInterface->leNeuronsOperation->setPalette(l_palette);
        break;
        case GridSearch::LEAK_RATE :
            m_uiInterface->leLeakRateOperation->setPalette(l_palette);
        break;
        case GridSearch::SPARCITY :
            m_uiInterface->leSparcityOperation->setPalette(l_palette);
        break;
        case GridSearch::INPUT_SCALING :
            m_uiInterface->leISOperation->setPalette(l_palette);
        break;
        case GridSearch::RIDGE :
            m_uiInterface->leRidgeOperation->setPalette(l_palette);
        break;
        case GridSearch::SPECTRAL_RADIUS :
            m_uiInterface->leSpectralRadiusOperation->setPalette(l_palette);
        break;
    }
}

void Interface::displayCurrentParameters(ModelParameters params)
{
    m_uiInterface->laNeuronsValue->setText(QString::number(params.m_nbNeurons));
    m_uiInterface->laLeakRateValue->setText(QString::number(params.m_leakRate));
    m_uiInterface->laISValue->setText(QString::number(params.m_inputScaling));
    m_uiInterface->laSpectralRadiusValue->setText(QString::number(params.m_spectralRadius));
    m_uiInterface->laRidgeValue->setText(QString::number(params.m_ridge));
    m_uiInterface->laSparcityValue->setText(QString::number(params.m_sparcity));
}

void Interface::displayCurrentResults(ResultsDisplayReservoir results)
{
    int l_widthTb = m_uiInterface->tbResults->width();

    QFont l_font("Calibri", 9);
    QFontMetrics l_fm(l_font);
    int pixelsWide = l_fm.width("W");
    m_uiInterface->tbResults->setFont(l_font);
    int l_numberCharLine = 2*l_widthTb / pixelsWide;

    QString l_display;

    if(results.m_action == TRAINING_RES || results.m_action == BOTH_RES)
    {
        if(results.m_trainResults.size() > 0)
        {
            m_uiInterface->tbResults->setTextColor(QColor(Qt::blue));
            m_uiInterface->tbResults->insertPlainText("[TRAINING]\n");
            m_uiInterface->tbResults->setTextColor(QColor(Qt::black));
        }

        for(int ii = 0; ii < results.m_trainSentences.size(); ++ii)
        {
            m_uiInterface->tbResults->insertPlainText("Corpus sentence      : ");

            l_display = "";
            for(int jj = 0; jj < results.m_trainSentences[ii].size(); ++jj)
            {
                l_display+= QString::fromStdString(results.m_trainSentences[ii][jj]) + " ";
            }
            m_uiInterface->tbResults->insertPlainText(l_display);
            m_uiInterface->tbResults->insertPlainText("\nSentence retrieved  : ");

            l_display = "";
            for(int jj = 0; jj < results.m_trainResults[ii].size(); ++jj)
            {
                l_display+= QString::fromStdString(results.m_trainResults[ii][jj]) + " ";
            }
            m_uiInterface->tbResults->insertPlainText(l_display);
            m_uiInterface->tbResults->insertPlainText("\nResults computed    : ");

            if(results.m_absoluteCCW.size() == results.m_trainSentences.size())
            {
                int l_nbCCW = results.m_continuousCCW[ii];
                int l_nbAll = results.m_continuousAll[ii];

                if(l_nbCCW != 100 && l_nbAll != 100)
                {
                    m_uiInterface->tbResults->setTextColor(QColor(Qt::darkRed));
                }
                else if(l_nbCCW != 100 || l_nbAll != 100)
                {
                    m_uiInterface->tbResults->setTextColor(QColor(Qt::red));
                }
                else
                {
                    m_uiInterface->tbResults->setTextColor(QColor(Qt::green));
                }

                m_uiInterface->tbResults->insertPlainText("CCW : "  + QString::number(l_nbCCW) + " ALL : " +  QString::number(l_nbAll) + "\n");
                m_uiInterface->tbResults->setTextColor(QColor(Qt::black));
            }

            l_display = "";
            for(int jj = 0; jj < l_numberCharLine; ++jj)
            {
                l_display += "-";
            }

            l_display += "\n";
            m_uiInterface->tbResults->insertPlainText(l_display);
        }
    }

    if(results.m_action == TEST_RES || results.m_action == BOTH_RES)
    {
        if(results.m_testResults.size() > 0)
        {
            m_uiInterface->tbResults->setTextColor(QColor(Qt::blue));
            m_uiInterface->tbResults->insertPlainText("[TEST]\n");
            m_uiInterface->tbResults->setTextColor(QColor(Qt::black));

            for(int ii = 0; ii < results.m_testResults.size(); ++ii)
            {
                m_uiInterface->tbResults->insertPlainText("Sentence retrieved : ");

                l_display = "";
                for(int jj = 0; jj < results.m_testResults[ii].size(); ++jj)
                {
                    l_display+= QString::fromStdString(results.m_testResults[ii][jj]) + " ";
                }

                l_display += "\n";

                for(int jj = 0; jj < l_numberCharLine; ++jj)
                {
                    l_display += "-";
                }

                l_display += "\n";
                m_uiInterface->tbResults->insertPlainText(l_display);
            }
        }
    }

    m_uiInterface->tbResults->verticalScrollBar()->setValue(m_uiInterface->tbResults->verticalScrollBar()->maximum());
}

void Interface::updateProgressBar(int currentValue, int valueMax, QString text)
{
    m_uiInterface->pbComputing->setMaximum(valueMax);
    m_uiInterface->pbComputing->setValue(currentValue);
    m_uiInterface->laStateComputing->setText(text);
}



void Interface::cleanResultsDisplay()
{
    m_uiInterface->tbResults->clear();
}

void Interface::displayLogInfo(QString info, QColor colorText)
{
    if(m_logFile.isOpen())
    {
        QTextStream l_stream(&m_logFile);
        l_stream << info;
    }

    m_uiInterface->tbInfos->setTextColor(colorText);
    m_uiInterface->tbInfos->insertPlainText (info);
    m_uiInterface->tbInfos->verticalScrollBar()->setValue(m_uiInterface->tbInfos->verticalScrollBar()->maximum());
}


void Interface::displayOutputMatrix(cv::Mat output, Sentences sentences)
{
    QVector<QVector<QVector<double> > > l_sentences; // dim 1 -> sentences / dim 2 -> CCW / dim 3 -> values

    // delete previous plots and labels
        qDeleteAll(m_plotListTrainSentenceOutput.begin(), m_plotListTrainSentenceOutput.end());
        qDeleteAll(m_plotLabelListTrainSentenceOutput.begin(), m_plotLabelListTrainSentenceOutput.end());
        qDeleteAll(m_labelListRetrievedSentences.begin(), m_labelListRetrievedSentences.end());
        m_plotListTrainSentenceOutput.clear();
        m_plotLabelListTrainSentenceOutput.clear();
        m_labelListRetrievedSentences.clear();

        LanguageParameters l_language = m_pWInterface->languageParameters();
        QStringList l_CCW = l_language.m_CCW.split(' ');
        l_CCW << "X";

    // create x curves values
        QVector<double> l_xValues;
        for(int ii = 0; ii < output.size[1]; ++ii)
        {
            l_xValues << ii;
        }

    // create y curves values
        for(int ii = 0; ii < output.size[0]; ++ii)
        {
            QVector<QVector<double> > l_CCW;

            for(int jj = 0; jj  < output.size[2]; ++jj)
            {
                QVector<double> l_values;

                for(int kk = 0; kk < output.size[1]; ++kk)
                {
                    l_values << static_cast<double>(output.at<float>(ii,kk,jj));
                }
                l_CCW << l_values;

            }
            l_sentences << l_CCW;
        }

    // set curve legend labels
        for(int ii = 0; ii < l_CCW.size(); ++ii)
        {
            QLabel *l_labelCCW = new QLabel();
            QPalette l_palette = l_labelCCW->palette();
            l_palette.setColor(l_labelCCW->foregroundRole(), m_colorsCCW[ii]);
            l_palette.setColor(l_labelCCW->backgroundRole(), Qt::white);
            l_labelCCW->setAutoFillBackground (true);
            l_labelCCW->setPalette(l_palette);
//            l_labelCCW->setText(l_CCW[ii].toUpper());

            QString labelText = "<P><b>";
            labelText .append(l_CCW[ii].toUpper());
            labelText .append("</b></P>");
            l_labelCCW->setText(labelText);

            l_labelCCW->setAlignment(Qt::AlignCenter);

            m_plotLabelListTrainSentenceOutput.push_back(l_labelCCW);
            m_uiInterface->hlLabelsOutputPlot->addWidget(m_plotLabelListTrainSentenceOutput.back());
        }

    // create the plots
        for(int ii = 0; ii < l_sentences.size(); ++ii)
        {
            QCustomPlot *l_plotDisplay = new QCustomPlot(this);
            l_plotDisplay->setFixedHeight(150);
            l_plotDisplay->setFixedWidth(output.size[2]*35);

            QCPItemText *l_textLabelSentence = new QCPItemText(l_plotDisplay);
            l_plotDisplay->addItem(l_textLabelSentence);
            l_textLabelSentence->setPositionAlignment(Qt::AlignTop|Qt::AlignHCenter);
            l_textLabelSentence->position->setType(QCPItemPosition::ptAxisRectRatio);
            l_textLabelSentence->position->setCoords(0.5, 0); // place position at center/top of axis rect

            Sentence l_senenteRetrieved = sentences[ii];
            QString l_text;
            for(int jj = 0; jj < l_senenteRetrieved.size(); ++jj)
            {
                QString l_part = QString::fromStdString(l_senenteRetrieved[jj]);

                bool l_isWordCCW = false;
                for(int kk = 0; kk < l_CCW.size(); ++kk)
                {
                    if(l_part == l_CCW[kk])
                    {
                        l_text += "<font color=" + m_colorsCCW[kk].name() + ">";
                        l_isWordCCW = true;
                        break;
                    }
                }
                if(!l_isWordCCW)
                {
                    l_text += "<font color=" + m_colorsCCW.back().name() + ">";
                }

                l_text += QString::fromStdString(l_senenteRetrieved[jj]).toUpper() + " </font>";
            }

            l_textLabelSentence->setFont(QFont(font().family(), 8)); // make font a bit larger
            l_textLabelSentence->setText("");

            m_plotListTrainSentenceOutput.push_back(l_plotDisplay);

            QLabel *l_sentenceText = new QLabel;
            l_sentenceText->setText("<P><b>Sentence " + QString::number(ii+1) + " :   " + l_text + "</b></P>");
            m_labelListRetrievedSentences.push_back(l_sentenceText);


            m_uiInterface->vlOutputPlot->addWidget(m_labelListRetrievedSentences.back());
            m_uiInterface->vlOutputPlot->addWidget(m_plotListTrainSentenceOutput.back());

            double l_maxY = DBL_MIN;
            double l_minY = DBL_MAX;

            for(int jj = 0; jj < l_sentences[0].size(); ++jj)
            {
                m_plotListTrainSentenceOutput.back()->addGraph();
                QPen l_pen(m_colorsCCW[jj]);
                l_pen.setWidthF(3);
                m_plotListTrainSentenceOutput[ii]->graph(jj)->setPen(l_pen);
                m_plotListTrainSentenceOutput[ii]->graph(jj)->setData(l_xValues, l_sentences[ii][jj]);

                for(QVector<double>::iterator it = l_sentences[ii][jj].begin(); it != l_sentences[ii][jj].end(); ++it)
                {
                    if((*it) > l_maxY)
                    {
                        l_maxY = (*it);
                    }
                    if((*it) < l_minY)
                    {
                        l_minY = (*it);
                    }
                }

            }

            m_plotListTrainSentenceOutput.back()->xAxis->setRange(0, output.size[1]);
            m_plotListTrainSentenceOutput.back()->yAxis->setRange(l_minY,l_maxY);
            m_plotListTrainSentenceOutput.back()->replot();
        }

    m_uiInterface->pbSaveOutput->setEnabled(true);
}

void Interface::displayTrainInputMatrix(cv::Mat trainMeaning, cv::Mat trainSentence, Sentences sentences)
{
    QVector<QVector<QVector<double> > > l_sentences; // dim 1 -> sentences / dim 2 -> CCW / dim 3 -> values
    QVector<QVector<QVector<double> > > l_meaning; // dim 1 -> sentences / dim 2 -> Structure / dim 3 -> values

    // delete previous plots and labels
        qDeleteAll(m_plotListTrainSentenceInput.begin(), m_plotListTrainSentenceInput.end());
        qDeleteAll(m_plotListTrainMeaningInput.begin(), m_plotListTrainMeaningInput.end());
        qDeleteAll(m_plotLabelListTrainSentenceInput.begin(), m_plotLabelListTrainSentenceInput.end());
        qDeleteAll(m_labelListInputSentences.begin(), m_labelListInputSentences.end());
        m_plotListTrainSentenceInput.clear();
        m_plotListTrainMeaningInput.clear();
        m_plotLabelListTrainSentenceInput.clear();
        m_labelListInputSentences.clear();

        LanguageParameters l_language = m_pWInterface->languageParameters();

        QStringList l_CCW = l_language.m_CCW.split(' ');
        l_CCW << "X";

    // create y curves values for sentences
        for(int ii = 0; ii < trainSentence.size[0]; ++ii)
        {
            QVector<QVector<double> > l_CCW;

            for(int jj = 0; jj  < trainSentence.size[2]; ++jj)
            {
                QVector<double> l_values;

                for(int kk = 0; kk < trainSentence.size[1]; ++kk)
                {
                    l_values << static_cast<double>(trainSentence.at<float>(ii,kk,jj));
                }
                l_CCW << l_values;

            }
            l_sentences << l_CCW;
        }
    // create y curves values for meaning
        for(int ii = 0; ii < trainMeaning.size[0]; ++ii)
        {
            QVector<QVector<double> > l_Structure;

            for(int jj = 0; jj  < trainMeaning.size[2]; ++jj)
            {
                QVector<double> l_values;

                for(int kk = 0; kk < trainMeaning.size[1]; ++kk)
                {
                    l_values << static_cast<double>(trainMeaning.at<float>(ii,kk,jj));
                }
                l_Structure << l_values;

            }
            l_meaning << l_Structure;
        }

    // create x curves values
        QVector<double> l_xValues;
        for(int ii = 0; ii < trainSentence.size[1]; ++ii)
        {
            l_xValues << ii;
        }

    // set curve legend labels
        for(int ii = 0; ii < l_CCW.size(); ++ii)
        {
            QLabel *l_labelCCW = new QLabel();
            QPalette l_palette = l_labelCCW->palette();
            l_palette.setColor(l_labelCCW->foregroundRole(), m_colorsCCW[ii]);
            l_palette.setColor(l_labelCCW->backgroundRole(), Qt::white);
            l_labelCCW->setAutoFillBackground (true);
            l_labelCCW->setPalette(l_palette);

            QString labelText = "<P><b>";
            labelText .append(l_CCW[ii].toUpper());
            labelText .append("</b></P>");
            l_labelCCW->setText(labelText);

            l_labelCCW->setAlignment(Qt::AlignCenter);
            m_plotLabelListTrainSentenceInput.push_back(l_labelCCW);
            m_uiInterface->hlLabelsInputPlot->addWidget(m_plotLabelListTrainSentenceInput.back());
        }

    // create the plots
        for(int ii = 0; ii < l_sentences.size(); ++ii)
        {
            // add sentences
                QCustomPlot *l_plotDisplaySentence = new QCustomPlot(this);
                l_plotDisplaySentence->setFixedHeight(150);
                l_plotDisplaySentence->setFixedWidth(trainSentence.size[2]*35);

                m_plotListTrainSentenceInput.push_back(l_plotDisplaySentence);


            // add sentence label
                Sentence l_senenteRetrieved = sentences[ii];
                QString l_text;
                for(int jj = 0; jj < l_senenteRetrieved.size(); ++jj)
                {
                    QString l_part = QString::fromStdString(l_senenteRetrieved[jj]);

                    bool l_isWordCCW = false;
                    for(int kk = 0; kk < l_CCW.size(); ++kk)
                    {
                        if(l_part == l_CCW[kk])
                        {
                            l_text += "<font color=" + m_colorsCCW[kk].name() + ">";
                            l_isWordCCW = true;
                            break;
                        }
                    }
                    if(!l_isWordCCW)
                    {
                        l_text += "<font color=" + m_colorsCCW.back().name() + ">";
                    }

                    l_text += QString::fromStdString(l_senenteRetrieved[jj]).toUpper() + " </font>";
                }


                QCPItemText *l_textLabelSentence = new QCPItemText(l_plotDisplaySentence);
                l_plotDisplaySentence->addItem(l_textLabelSentence);
                l_textLabelSentence->setPositionAlignment(Qt::AlignTop|Qt::AlignHCenter);
                l_textLabelSentence->position->setType(QCPItemPosition::ptAxisRectRatio);
                l_textLabelSentence->position->setCoords(0.5, 0); // place position at center/top of axis rect
                l_textLabelSentence->setText("");
                l_textLabelSentence->setFont(QFont(font().family(), 8)); // make font a bit larger

                QLabel *l_sentenceText = new QLabel;
                l_sentenceText->setText("<P><b>Sentence " + QString::number(ii+1) + " :   " + l_text + "</b></P>");
                m_labelListInputSentences.push_back(l_sentenceText);

                double l_maxY = DBL_MIN;
                double l_minY = DBL_MAX;

                for(int jj = 0; jj < l_sentences[0].size(); ++jj)
                {
                    m_plotListTrainSentenceInput.back()->addGraph();
                    QPen l_pen(m_colorsCCW[jj]);
                    l_pen.setWidthF(3);
                    m_plotListTrainSentenceInput[ii]->graph(jj)->setPen(l_pen);
                    m_plotListTrainSentenceInput[ii]->graph(jj)->setData(l_xValues, l_sentences[ii][jj]);

                    for(QVector<double>::iterator it = l_sentences[ii][jj].begin(); it != l_sentences[ii][jj].end(); ++it)
                    {
                        if((*it) > l_maxY)
                        {
                            l_maxY = (*it);
                        }
                        if((*it) < l_minY)
                        {
                            l_minY = (*it);
                        }
                    }
                }

                m_plotListTrainSentenceInput.back()->xAxis->setRange(0, trainSentence.size[1]);
                m_plotListTrainSentenceInput.back()->yAxis->setRange(l_minY,l_maxY);
                m_plotListTrainSentenceInput.back()->replot();

            // add meaning
                QCustomPlot *l_plotDisplayMeaning = new QCustomPlot(this);
                l_plotDisplayMeaning->setFixedHeight(100);
                l_plotDisplayMeaning->setFixedWidth(trainMeaning.size[2]*10);
                m_plotListTrainMeaningInput.push_back(l_plotDisplayMeaning);
                m_uiInterface->vlInputPlot->addWidget(m_labelListInputSentences.back());
                m_uiInterface->vlInputPlot->addWidget(m_plotListTrainSentenceInput.back());
//                m_uiInterface->vlInputPlot->addWidget(m_plotListTrainMeaningInput.back());


                QCPColorMap *l_colorMap = new QCPColorMap(l_plotDisplayMeaning->xAxis, l_plotDisplayMeaning->yAxis);
                l_plotDisplayMeaning->addPlottable(l_colorMap);
                l_colorMap->data()->setSize(trainMeaning.size[2], trainMeaning.size[1]);

                double z;
                for(int jj = 0; jj < trainMeaning.size[1]; ++jj)
                {
                    for(int kk = 0; kk < trainMeaning.size[2]; ++kk)
                    {
                        z = static_cast<double>(trainMeaning.at<float>(ii,jj,kk));
                        l_colorMap->data()->setCell(kk, jj, z);
                    }
                }

                // add a color scale:
                QCPColorScale *colorScale = new QCPColorScale(l_plotDisplayMeaning);
//                l_plotDisplayMeaning->plotLayout()->addElement(0, 1, colorScale); // add it to the right of the main axis rect
                colorScale->setType(QCPAxis::atRight); // scale shall be vertical bar with tick/axis labels right (actually atRight is already the default)
                l_colorMap->setColorScale(colorScale); // associate the color map with the color scale

                l_colorMap->setGradient(QCPColorGradient::gpPolar);
                l_colorMap->rescaleDataRange(true);
                l_plotDisplayMeaning->rescaleAxes();
                colorScale->axis()->setLabel("Meaning");

                m_plotListTrainMeaningInput.back()->xAxis->setAutoTickStep(false);
                m_plotListTrainMeaningInput.back()->xAxis->setAutoTickLabels(false);
                m_plotListTrainMeaningInput.back()->replot();
        }
}

void Interface::openCorpus()
{
    int l_currentIndex = m_uiInterface->lwCorpus->currentRow();

    if(l_currentIndex >= 0)
    {
        QDesktopServices::openUrl(QUrl("file:///" + m_uiInterface->lwCorpus->currentItem()->text()));
        m_uiInterface->lwCorpus->currentItem()->setTextColor(QColor(255,0,0));
    }
}

void Interface::reloadCorpus()
{
    int l_currentIndex2Reload = m_uiInterface->lwCorpus->currentRow();

    if(l_currentIndex2Reload >= 0)
    {
        QString l_pathCorpus2Reload = m_uiInterface->lwCorpus->currentItem()->text();

        // remove item
            delete m_uiInterface->lwCorpus->takeItem(l_currentIndex2Reload);
            emit removeCorpusSignal(l_currentIndex2Reload);

        // wait
            QTime l_oDieTime = QTime::currentTime().addMSecs(10);
            while( QTime::currentTime() < l_oDieTime)
            {
                QCoreApplication::processEvents(QEventLoop::AllEvents, 10);
            }

        // send item
            m_uiInterface->lwCorpus->addItem(l_pathCorpus2Reload);
            emit addCorpusSignal(l_pathCorpus2Reload);

        m_uiInterface->lwCorpus->setCurrentRow(m_uiInterface->lwCorpus->count()-1);
        m_uiInterface->lwCorpus->currentItem()->setTextColor(QColor(0,0,0));
    }
}


void Interface::updateColorCCW(LanguageParameters params)
{
    // create color for each CCW
    int l_nbLoop = 0;
    m_colorsCCW.clear();
    while(m_colorsCCW.size() < params.m_CCW.size()+1)
    {
        ++l_nbLoop;;
        bool l_addColor = true;

        int l_r = rand()%240;
        int l_g = rand()%255;
        int l_b = rand()%255;

        if(l_nbLoop < 500)
        {
            for(int ii = 0; ii < m_colorsCCW.size(); ++ii)
            {
                int l_diffRed = m_colorsCCW[ii].red()-l_r;
                if(l_diffRed < 0)
                {
                    l_diffRed = -l_diffRed;
                }
                int l_diffGreen = m_colorsCCW[ii].green()-l_g;
                if(l_diffGreen < 0)
                {
                    l_diffGreen = - l_diffGreen;
                }
                int l_diffBlue = m_colorsCCW[ii].blue()-l_b;
                if(l_diffBlue < 0)
                {
                    l_diffBlue = - l_diffBlue;
                }

                if(l_diffRed + l_diffGreen + l_diffBlue < 150)
                {
                    l_addColor = false;
                    break;
                }
            }
        }

        if(l_addColor)
        {
            m_colorsCCW << QColor(l_r,l_g,l_b);
        }
    }
}

void Interface::setColorLine()
{
   m_colorLine = QColorDialog::getColor(Qt::blue, this);
   QRgb l_colorRGB = m_colorLine.rgb();
   m_uiInterface->pbColor->setStyleSheet("background-color: rgb("+ QString::number(m_colorLine.red()) +
                                         ", " + QString::number(m_colorLine.green()) + ", " + QString::number(m_colorLine.blue()) +");");
}

void Interface::loadSettings()
{
    QString l_sPathSettings = QFileDialog::getOpenFileName(this, "Load settings file", m_absolutePath + "../data/input/Settings", "Corpus file (*.txt)");

    if(l_sPathSettings.size() > 0)
    {
        QStringList l_CCW, l_structure;
        extractDataFromSettingFile(l_sPathSettings,l_CCW, l_structure);

        m_uiInterface->leSettingFile->setText(l_sPathSettings);
        m_uiInterface->leDisplayCCW->setText(l_CCW.join(" "));
        m_uiInterface->leDisplayStructure->setText(l_structure.join(" "));

        LanguageParameters l_params;
        l_params.m_CCW = l_CCW.join(" ");
        l_params.m_structure = l_structure.join(" ");

        updateColorCCW(l_params);

        emit sendLanguageParametersSignal(l_params);

        // unlock start
        if(m_uiInterface->lwCorpus->count() > 0)
        {
            m_uiInterface->pbStart->setEnabled(true);
        }
    }
}

void Interface::updateSettings()
{
    LanguageParameters l_params;
    l_params.m_CCW = m_uiInterface->leDisplayCCW->text();
    updateColorCCW(l_params);

    l_params.m_structure = m_uiInterface->leDisplayStructure->text();

    if(l_params.m_CCW.size() > 0 && l_params.m_structure.size() > 0)
    {
        emit sendLanguageParametersSignal(l_params);

        // unlock start
        if(m_uiInterface->lwCorpus->count() > 0)
        {
            m_uiInterface->pbStart->setEnabled(true);
        }
    }
    else
    {
        m_uiInterface->pbStart->setEnabled(false);
    }
}

void Interface::openCorpus(QModelIndex index)
{
    int l_clickedRow = index.row();

    if(l_clickedRow >= 0)
    {
        QDesktopServices::openUrl(QUrl("file:///" + m_uiInterface->lwCorpus->currentItem()->text()));
        m_uiInterface->lwCorpus->currentItem()->setTextColor(QColor(255,0,0));
    }
}

void Interface::setXTabFocus(int index)
{
    if(index == 1)
    {
        m_uiInterface->twDisplay->setTabEnabled(1, true);
        m_uiInterface->twDisplay->setCurrentIndex(1);
    }
    else
    {
        m_uiInterface->twDisplay->setTabEnabled(1, false);
        m_uiInterface->twDisplay->setCurrentIndex(0);
    }
}

void Interface::disableCustomMatrix(int index)
{
    if(index !=0)
    {
        m_uiInterface->cbW->setChecked(false);
        m_uiInterface->cbWIn->setChecked(false);
    }
}

void Interface::disableTraining(int index)
{
    if(index !=0)
    {
        m_uiInterface->cbTrainingFile->setChecked(false);
    }
}

void Interface::resetLoadingBar()
{
    m_uiInterface->pbComputing->setValue(0);
    m_uiInterface->laStateComputing->setText("Started...");
}

void Interface::replayLoaded()
{
    m_uiInterface->laStateComputing->setText("Replay loaded...");
    m_uiInterface->pbComputing->setValue(100);
    m_uiInterface->twSettings->setEnabled(true);
}

void Interface::updateDisplayReplay(QVector<QVector<double> > data, QVector<int> neuronsId, QVector<int> sentencesId)
{
    // define size of the sentences :
    double l_sentenceWidth;
    if(m_uiInterface->rbFixedWidth->isChecked())
    {
        l_sentenceWidth = m_uiInterface->sbWidthSentence->value();
    }
    else
    {
        l_sentenceWidth = (m_uiInterface->twDisplay->size().width()-90)*1.0/sentencesId.size();
    }
    double l_sentenceHeight;
    if(m_uiInterface->rbFixedHeight->isChecked())
    {
        l_sentenceHeight = m_uiInterface->sbHeightSentence->value();
    }
    else
    {
        l_sentenceHeight = (m_uiInterface->twDisplay->size().height()-200)*1.0/neuronsId.size();
    }


    m_uiInterface->twSettings->setEnabled(false);
    m_uiInterface->pbComputing->setValue(0);
    m_uiInterface->laStateComputing->setText("Plotting...");

    QTime l_oDieTime = QTime::currentTime().addMSecs(10);
    while( QTime::currentTime() < l_oDieTime)
    {
        QCoreApplication::processEvents(QEventLoop::AllEvents, 10);
    }


    // reset plots
    qDeleteAll(m_plotReplay.begin(), m_plotReplay.end());
    m_plotReplay.clear();

    QVector<double> l_xValues;
    for(int ii = 0; ii < data[0].size(); ++ii)
    {
        l_xValues.push_back(ii);
    }

    QVector<QString> l_labelsX;
    for(int ii = 0; ii < sentencesId.size(); ++ii)
    {
        if(ii == 0)
        {
            l_labelsX << "Sentence " + QString::number(sentencesId[ii]);
        }
        else
        {
            l_labelsX << "S_" + QString::number(sentencesId[ii]);
        }
    }

    for(int ii = 0; ii < data.size(); ++ii)
    {
        QCustomPlot *l_plot = new QCustomPlot(this);
        l_plot->setFixedHeight(l_sentenceHeight);
        l_plot->setFixedWidth(sentencesId.size()*l_sentenceWidth);
        m_plotReplay << l_plot;
        m_uiInterface->vlXPlot->addWidget(m_plotReplay.back());
        m_plotReplay.back()->addGraph();

        m_plotReplay.back()->xAxis->setLabel("x");
        m_plotReplay.back()->xAxis->setRange(0, data[0].size());

        m_plotReplay.back()->xAxis->setAutoTickStep(false);
        m_plotReplay.back()->xAxis->setAutoTickLabels(false);
        m_plotReplay.back()->xAxis->setTickVectorLabels(l_labelsX);
        m_plotReplay.back()->xAxis->setTickStep(data[0].size()/sentencesId.size());
        m_plotReplay.back()->xAxis->setLabel("");

        m_plotReplay.back()->yAxis->setLabel("");
        m_plotReplay.back()->yAxis->setRange(-1, 1);
        m_plotReplay.back()->yAxis->setAutoTickStep(false);
        m_plotReplay.back()->yAxis->setAutoTickLabels(false);
        m_plotReplay.back()->yAxis->setTickStep(1.0);

        if(ii == 0)
        {
            m_plotReplay.back()->yAxis->setLabel("Neuron " + QString::number(neuronsId[ii]));
        }
        else
        {
            m_plotReplay.back()->yAxis->setLabel("N_" + QString::number(neuronsId[ii]));
        }

        QPen l_pen(m_colorLine);
        l_pen.setWidthF(m_uiInterface->sbLineWidth->value());
        m_plotReplay.back()->graph(0)->setPen(l_pen);
        m_plotReplay.back()->graph(0)->addData(l_xValues, data[ii]);
        m_plotReplay.back()->replot();

        m_uiInterface->pbComputing->setValue(100.0 * ii / data.size());

        l_oDieTime = QTime::currentTime().addMSecs(10);
        while( QTime::currentTime() < l_oDieTime)
        {
            QCoreApplication::processEvents(QEventLoop::AllEvents, 10);
        }
    }

    m_uiInterface->twSettings->setEnabled(true);
    m_uiInterface->pbComputing->setValue(100);
    m_uiInterface->laStateComputing->setText("End plotting");

    m_uiInterface->pbSaveX->setEnabled(true);

}


void Interface::saveXPlot()
{
    if(m_plotReplay.size() > 0)
    {
        QString l_pathPlot = QFileDialog::getSaveFileName(this, "Save plot image", m_absolutePath + "../data/images/plots", "Plot image (*.png)");

        QSize l_sizePlot = m_plotReplay[0]->size();
        QSize l_sizeImage = l_sizePlot;
        l_sizeImage.setHeight(l_sizeImage.height()*m_plotReplay.size());

        QPixmap l_plotImage(l_sizeImage);
        l_plotImage.fill();

        QPainter l_painter(&l_plotImage);

        for(int ii = 0; ii< m_plotReplay.size(); ++ii)
        {
            l_painter.drawPixmap(QRectF(QPointF(0,ii*l_sizePlot.height()),QPointF(l_sizePlot.width(),(ii+1)*l_sizePlot.height())),m_plotReplay[ii]->toPixmap(),
                               QRectF(QPointF(0,0),                     QPointF(l_sizePlot.width(),       l_sizePlot.height())));
        }
        l_plotImage.save(l_pathPlot);

        displayLogInfo(QString("Plot : " + l_pathPlot + " saved. "), QColor(Qt::blue));
    }
    else
    {
        displayLogInfo(QString("The plot must be done before the saving. "), QColor(Qt::red));
    }
}

void Interface::saveOutput()
{
    if(m_plotListTrainSentenceOutput.size() > 0)
    {
        QString l_pathPlot = QFileDialog::getSaveFileName(this, "Save output plot image", m_absolutePath + "../data/images/plots", "Plot image (*.png)");

        QSize l_sizePlot = m_plotListTrainSentenceOutput[0]->size();
        QSize l_sizeImage = l_sizePlot;
        l_sizeImage.setHeight(l_sizeImage.height()*m_plotListTrainSentenceOutput.size());

        QPixmap l_plotImage(l_sizeImage);
        l_plotImage.fill();

        QPainter l_painter(&l_plotImage);

        for(int ii = 0; ii< m_plotListTrainSentenceOutput.size(); ++ii)
        {
            l_painter.drawPixmap(QRectF(QPointF(0,ii*l_sizePlot.height()),QPointF(l_sizePlot.width(),(ii+1)*l_sizePlot.height())),m_plotListTrainSentenceOutput[ii]->toPixmap(),
                               QRectF(QPointF(0,0),                     QPointF(l_sizePlot.width(),       l_sizePlot.height())));

//            l_painter.drawText(QRectF(QPointF(0,ii*l_sizePlot.height()),QPointF(l_sizePlot.width(),(ii+1)*l_sizePlot.height())), m_labelListInputSentences[ii]->text());
        }
        l_plotImage.save(l_pathPlot);

        displayLogInfo(QString("Output plot : " + l_pathPlot + " saved. "), QColor(Qt::blue));
    }
    else
    {
        displayLogInfo(QString("The computing must be done before the saving. "), QColor(Qt::red));
    }
}
