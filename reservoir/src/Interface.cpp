

/**
 * \file Interface.cpp
 * \brief Defines SWViewerInterface
 * \author Florian Lance
 * \date 01/12/14
 */

#include "Interface.h"
#include "../moc/moc_Interface.cpp"

#include <QCheckBox>
#include <time.h>


#include <QDateTime>

int main(int argc, char* argv[])
{
    srand(1);
    culaWarmup(1);

    QApplication l_oApp(argc, argv);
    Interface l_oViewerInterface(&l_oApp);
    l_oViewerInterface.move(50,50);
    l_oViewerInterface.show();

    return l_oApp.exec();
}


Interface::Interface(QApplication *parent) : m_uiInterface(new Ui::UI_Reservoir)
{
    // init main widget
    m_uiInterface->setupUi(this);
    this->setWindowTitle(QString("Reservoir - cuda"));

    // init CCW / structure (TEMP)
        QStringList l_listCCW, l_listStructure;
        l_listCCW << "and s of the to . -ed -ing -s by it that was did , from";
        l_listCCW << "-ga -ni -wo -yotte -o -to sore";
        m_uiInterface->cbCCW->addItems(l_listCCW);
        l_listStructure << "P0 A1 O2 R3";
        l_listStructure << "P0 A1 O2 R3 Q0";
        m_uiInterface->cbStructure->addItems(l_listStructure);

    // create log file
        QDateTime l_dateTime;
        l_dateTime = l_dateTime.currentDateTime();
        QDate l_date = l_dateTime.date();
        QTime l_time = QTime::currentTime();
        QString l_nameLogFile = "../log/logs_reservoir_interface_" + l_date.toString("dd_MM_yyyy") + "_" +  QString::number(l_time.hour()) + "h" + QString::number(l_time.minute()) + "m" + QString::number(l_time.second()) + "s.txt";
        m_logFile.setFileName(l_nameLogFile);
        if (!m_logFile.open(QIODevice::WriteOnly | QIODevice::Text))
        {
           qWarning() << "Cannot write log file. Start of the path must be ./dist, and ../log must exist. ";
        }

    // init worker
        m_pWInterface = new InterfaceWorker();

    // init widgets
        m_uiInterface->scrollAreaPlotX->setWidgetResizable(true);
        m_uiInterface->scrollAreaPlotOutput->setWidgetResizable(true);
        m_uiInterface->scrollAreaPlotInput->setWidgetResizable(true);
        m_uiInterface->pbStop->setVisible(false);

    // init connections
        QObject::connect(this, SIGNAL(leaveProgram()), parent, SLOT(quit()));

        // push button
        QObject::connect(m_uiInterface->pbStart,        SIGNAL(clicked()), m_pWInterface, SLOT(start()));
        QObject::connect(m_uiInterface->pbStop,         SIGNAL(clicked()), m_pWInterface, SLOT(stop()));
        QObject::connect(m_uiInterface->pbAddCorpus,    SIGNAL(clicked()), this, SLOT(addCorpus()));
        QObject::connect(m_uiInterface->pbRemoveCorpus, SIGNAL(clicked()), this, SLOT(removeCorpus()));
        QObject::connect(m_uiInterface->pbSaveLastTrainingFile, SIGNAL(clicked()), this, SLOT(saveTraining()));
        QObject::connect(m_uiInterface->pbClearResults, SIGNAL(clicked()), this, SLOT(cleanResultsDisplay()));

        // radio button
        QObject::connect(m_uiInterface->rbTrain,    SIGNAL(clicked()), SLOT(updateReservoirParameters()));
        QObject::connect(m_uiInterface->rbTest,     SIGNAL(clicked()), SLOT(updateReservoirParameters()));
        QObject::connect(m_uiInterface->rbBoth,     SIGNAL(clicked()), SLOT(updateReservoirParameters()));

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
        QObject::connect(m_uiInterface->sbMaxSentencesDisplayed,SIGNAL(valueChanged(int)),    SLOT(updateReservoirParameters(int)));
        QObject::connect(m_uiInterface->sbStartRangeNeuronDisplay,SIGNAL(valueChanged(int)),    SLOT(updateReservoirParameters(int)));
        QObject::connect(m_uiInterface->sbEndRangeNeuronDisplay,SIGNAL(valueChanged(int)),    SLOT(updateReservoirParameters(int)));
        QObject::connect(m_uiInterface->sbNbRandomNeurons,      SIGNAL(valueChanged(int)),    SLOT(updateReservoirParameters(int)));

        // checkbox
        QObject::connect(m_uiInterface->cbNeurons,              SIGNAL(stateChanged(int)), SLOT(updateReservoirParameters(int)));
        QObject::connect(m_uiInterface->cbLeakRate,             SIGNAL(stateChanged(int)), SLOT(updateReservoirParameters(int)));
        QObject::connect(m_uiInterface->cbIS,                   SIGNAL(stateChanged(int)), SLOT(updateReservoirParameters(int)));
        QObject::connect(m_uiInterface->cbSpectralRadius,       SIGNAL(stateChanged(int)), SLOT(updateReservoirParameters(int)));
        QObject::connect(m_uiInterface->cbRidge,                SIGNAL(stateChanged(int)), SLOT(updateReservoirParameters(int)));
        QObject::connect(m_uiInterface->cbSparcity,             SIGNAL(stateChanged(int)), SLOT(updateReservoirParameters(int)));
        QObject::connect(m_uiInterface->cbTrainingFile,         SIGNAL(stateChanged(int)), SLOT(updateReservoirParameters(int)));
        QObject::connect(m_uiInterface->cbOnlyStartValue,       SIGNAL(stateChanged(int)), SLOT(updateReservoirParameters(int)));
        QObject::connect(m_uiInterface->cbEnableGPU,            SIGNAL(stateChanged(int)), SLOT(updateReservoirParameters(int)));
        QObject::connect(m_uiInterface->cbEnableDisplay,        SIGNAL(stateChanged(int)), SLOT(updateReservoirParameters(int)));
        QObject::connect(m_uiInterface->cbSelectRandomNeurons,  SIGNAL(stateChanged(int)), SLOT(updateReservoirParameters(int)));

        // lineedit
        QObject::connect(m_uiInterface->leNeuronsOperation,         SIGNAL(editingFinished()), SLOT(updateReservoirParameters()));
        QObject::connect(m_uiInterface->leLeakRateOperation,        SIGNAL(editingFinished()), SLOT(updateReservoirParameters()));
        QObject::connect(m_uiInterface->leISOperation,              SIGNAL(editingFinished()), SLOT(updateReservoirParameters()));
        QObject::connect(m_uiInterface->leSpectralRadiusOperation,  SIGNAL(editingFinished()), SLOT(updateReservoirParameters()));
        QObject::connect(m_uiInterface->leRidgeOperation,           SIGNAL(editingFinished()), SLOT(updateReservoirParameters()));
        QObject::connect(m_uiInterface->leSparcityOperation,        SIGNAL(editingFinished()), SLOT(updateReservoirParameters()));

        // combobox
        QObject::connect(m_uiInterface->cbStructure,    SIGNAL(currentIndexChanged(int)), SLOT(updateLanguageParameters(int)));
        QObject::connect(m_uiInterface->cbStructure,    SIGNAL(editTextChanged(QString)), SLOT(updateLanguageParameters(QString)));
        QObject::connect(m_uiInterface->cbCCW,          SIGNAL(currentIndexChanged(int)), SLOT(updateLanguageParameters(int)));
        QObject::connect(m_uiInterface->cbCCW,          SIGNAL(editTextChanged(QString)), SLOT(updateLanguageParameters(QString)));


        // lock
        QObject::connect(m_pWInterface, SIGNAL(lockInterfaceSignal(bool)), this, SLOT(lockInterface(bool)));

        // worker
        QObject::connect(this, SIGNAL(addCorpusSignal(QString)), m_pWInterface, SLOT(addCorpus(QString)));
        QObject::connect(this, SIGNAL(removeCorpusSignal(int)), m_pWInterface, SLOT(removeCorpus(int)));
        QObject::connect(this, SIGNAL(sendReservoirParametersSignal(ReservoirParameters)), m_pWInterface, SLOT(updateReservoirParameters(ReservoirParameters)));
        QObject::connect(this, SIGNAL(sendLanguageParametersSignal(LanguageParameters)), m_pWInterface, SLOT(updateLanguageParameters(LanguageParameters)));
        QObject::connect(m_pWInterface, SIGNAL(displayValidityOperationSignal(bool, int)), this, SLOT(displayValidityOperation(bool, int)));
        QObject::connect(m_pWInterface, SIGNAL(endTrainingSignal(bool)), m_uiInterface->pbSaveLastTrainingFile, SLOT(setEnabled(bool)));
        QObject::connect(this, SIGNAL(saveTrainingSignal(QString)), m_pWInterface, SLOT(saveLastTraining(QString)));
        QObject::connect(this, SIGNAL(loadTrainingSignal(QString)), m_pWInterface, SLOT(loadTraining(QString)));

        // gridsearch
        GridSearchQt *l_gridSearchQt = m_pWInterface->gridSearch();
        QObject::connect(l_gridSearchQt, SIGNAL(sendCurrentParametersSignal(ModelParametersQt)), this, SLOT(displayCurrentParameters(ModelParametersQt)));
        QObject::connect(l_gridSearchQt, SIGNAL(sendResultsReservoirSignal(ResultsDisplayReservoir)), this, SLOT(displayCurrentResults(ResultsDisplayReservoir)));

        // model
        ModelQt *l_model = m_pWInterface->model();
        QObject::connect(l_model->reservoir(), SIGNAL(sendComputingState(int,int,QString)), this, SLOT(updateProgressBar(int, int, QString)));
        QObject::connect(m_uiInterface->cbEnableMultiThread, SIGNAL(toggled(bool)), l_model->reservoir(), SLOT(enableMaxOmpThreadNumber(bool)));
        QObject::connect(m_uiInterface->cbEnableDisplay, SIGNAL(clicked(bool)), l_model->reservoir(), SLOT(enableDisplay(bool)));
        QObject::connect(l_model->reservoir(), SIGNAL(sendXMatriceData(QVector<QVector<double> >*, int, int )), this, SLOT(displayXMatrix(QVector<QVector<double> >*, int, int)));

        QObject::connect(l_model->reservoir(), SIGNAL(sendInfoPlot(int,int,int,QString)), this, SLOT(initPlot(int,int,int,QString)));
        QObject::connect(l_model->reservoir(), SIGNAL(sendLogInfo(QString)), this, SLOT(displayLogInfo(QString)));
        QObject::connect(l_model->reservoir(), SIGNAL(sendOutputMatrix(cv::Mat)), this, SLOT(displayOutputMatrix(cv::Mat)));
        QObject::connect(l_model, SIGNAL(sendTrainInputMatrix(cv::Mat,cv::Mat)), this, SLOT(displayTrainInputMatrix(cv::Mat,cv::Mat)));
        QObject::connect(this,                 SIGNAL(sendMatrixXDisplayParameters(bool,bool,int,int,int)), l_model->reservoir(), SLOT(updateMatrixXDisplayParameters(bool,bool,int,int,int)));

    // interface setting
        m_uiInterface->pbComputing->setRange(0,100);
        m_uiInterface->pbComputing->setValue(0);

    // init thread
        m_pWInterface->moveToThread(&m_TInterface);
        m_TInterface.start();

    // update worker parameters with defaults values
        updateReservoirParameters();
        updateLanguageParameters();
}

Interface::~Interface()
{
    m_TInterface.quit();
    m_TInterface.wait();

    delete m_pWInterface;
}

void Interface::closeEvent(QCloseEvent *event)
{
//    emit stopLoop();

    QTime l_oDieTime = QTime::currentTime().addMSecs(200);
    while( QTime::currentTime() < l_oDieTime)
    {
        QCoreApplication::processEvents(QEventLoop::AllEvents, 100);
    }
}

void Interface::addCorpus()
{

    QString l_sPathCorpus = QFileDialog::getOpenFileName(this, "Load corpus file", "../data/input/Corpus", "Corpus file (*.txt)");

    if(l_sPathCorpus.size() > 0)
    {        
        m_uiInterface->lwCorpus->addItem(l_sPathCorpus);

        // send item
        emit addCorpusSignal(l_sPathCorpus);
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
}

void Interface::saveTraining()
{
    QString l_sPathTrainingFile = QFileDialog::getExistingDirectory(this, "Select directory", "../data/training");

    if(l_sPathTrainingFile.size() == 0)
    {
        return;
    }

    // send directory path
    emit saveTrainingSignal(l_sPathTrainingFile);
}

void Interface::loadTraining()
{
    QString l_sPathTrainingFile = QFileDialog::getExistingDirectory(this, "Select directory", "../data/training");

    if(l_sPathTrainingFile.size() == 0 )
    {
        return;
    }

    QFile l_fileW(l_sPathTrainingFile + "/w.txt");
    QFile l_fileWin(l_sPathTrainingFile + "/wIn.txt");
    QFile l_fileWOut(l_sPathTrainingFile + "/wOut.txt");

    QPalette l_palette;
    if(l_fileW.exists() && l_fileWin.exists() && l_fileWOut.exists())
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
        m_uiInterface->leCurrentTrainingFile->setText("Training matrices not found in the directory...");
        std::cerr << "Training matrices not found, loading not done. " << std::endl;
    }

    m_uiInterface->leCurrentTrainingFile->setPalette(l_palette);
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

void Interface::updateLanguageParameters()
{
    LanguageParameters l_params;
    l_params.m_CCW = m_uiInterface->cbCCW->currentText();
    l_params.m_structure = m_uiInterface->cbStructure->currentText();

    emit sendLanguageParametersSignal(l_params);
}

void Interface::updateLanguageParameters(int value)
{
    updateLanguageParameters();
}

void Interface::updateLanguageParameters(QString value)
{
    updateLanguageParameters();
}

void Interface::updateReservoirParameters()
{
    ReservoirParameters l_params;

    l_params.m_useCuda                  = m_uiInterface->cbEnableGPU->isChecked();
    l_params.m_useLoadedTraining        = m_uiInterface->cbTrainingFile->isChecked();
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


    // display neurons activities parameters
    m_nbMaxNeuronsSentenceDisplayed = m_uiInterface->sbMaxSentencesDisplayed->value();
    emit sendMatrixXDisplayParameters(m_uiInterface->cbEnableDisplay->isChecked(), m_uiInterface->cbSelectRandomNeurons->isChecked(),
                                      m_uiInterface->sbNbRandomNeurons->value(), m_uiInterface->sbStartRangeNeuronDisplay->value(), m_uiInterface->sbEndRangeNeuronDisplay->value());
}

void Interface::lockInterface(bool lock)
{
    m_uiInterface->sbStartNeurons->setDisabled(lock);
    m_uiInterface->sbStartLeakRate->setDisabled(lock);
    m_uiInterface->sbStartIS->setDisabled(lock);
    m_uiInterface->sbStartSpectralRadius->setDisabled(lock);
    m_uiInterface->sbStartRidge->setDisabled(lock);
    m_uiInterface->sbStartSparcity->setDisabled(lock);

    m_uiInterface->sbEndNeurons->setDisabled(lock);
    m_uiInterface->sbEndLeakRate->setDisabled(lock);
    m_uiInterface->sbEndIS->setDisabled(lock);
    m_uiInterface->sbEndSpectralRadius->setDisabled(lock);
    m_uiInterface->sbEndRidge->setDisabled(lock);
    m_uiInterface->sbEndSparcity->setDisabled(lock);

    m_uiInterface->sbNbUseNeurons->setDisabled(lock);
    m_uiInterface->sbNbUseLeakRate->setDisabled(lock);
    m_uiInterface->sbNbUseISS->setDisabled(lock);
    m_uiInterface->sbNbUseSpectralRadius->setDisabled(lock);
    m_uiInterface->sbNbUseRidge->setDisabled(lock);
    m_uiInterface->sbNbUseSparcity->setDisabled(lock);

    m_uiInterface->cbNeurons->setDisabled(lock);
    m_uiInterface->cbLeakRate->setDisabled(lock);
    m_uiInterface->cbIS->setDisabled(lock);
    m_uiInterface->cbSpectralRadius->setDisabled(lock);
    m_uiInterface->cbRidge->setDisabled(lock);
    m_uiInterface->cbSparcity->setDisabled(lock);

    m_uiInterface->leNeuronsOperation->setDisabled(lock);
    m_uiInterface->leLeakRateOperation->setDisabled(lock);
    m_uiInterface->leISOperation->setDisabled(lock);
    m_uiInterface->leSpectralRadiusOperation->setDisabled(lock);
    m_uiInterface->leRidgeOperation->setDisabled(lock);
    m_uiInterface->leSparcityOperation->setDisabled(lock);

    m_uiInterface->cbCCW->setDisabled(lock);
    m_uiInterface->cbStructure->setDisabled(lock);

    m_uiInterface->pbAddCorpus->setDisabled(lock);
    m_uiInterface->pbRemoveCorpus->setDisabled(lock);
    m_uiInterface->pbClearResults->setDisabled(lock);

    m_uiInterface->pbStart->setDisabled(lock);
    m_uiInterface->pbLoadTrainingFile->setDisabled(lock);
    m_uiInterface->pbStop->setDisabled(!lock);

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
        case GridSearchQt::NEURONS_NB :
            m_uiInterface->leNeuronsOperation->setPalette(l_palette);
        break;
        case GridSearchQt::LEAK_RATE :
            m_uiInterface->leLeakRateOperation->setPalette(l_palette);
        break;
        case GridSearchQt::SPARCITY :
            m_uiInterface->leSparcityOperation->setPalette(l_palette);
        break;
        case GridSearchQt::INPUT_SCALING :
            m_uiInterface->leISOperation->setPalette(l_palette);
        break;
        case GridSearchQt::RIDGE :
            m_uiInterface->leRidgeOperation->setPalette(l_palette);
        break;
        case GridSearchQt::SPECTRAL_RADIUS :
            m_uiInterface->leSpectralRadiusOperation->setPalette(l_palette);
        break;
    }
}

void Interface::displayCurrentParameters(ModelParametersQt params)
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

    QString l_display = m_uiInterface->tbResults->toPlainText();

    if(results.m_trainResults.size() > 0)
    {
        l_display += "##################### TRAINING :\n";
    }

    for(int ii = 0; ii < results.m_trainSentences.size(); ++ii)
    {
        l_display += "Corpus sentence     : ";

        for(int jj = 0; jj < results.m_trainSentences[ii].size(); ++jj)
        {
            l_display+= QString::fromStdString(results.m_trainSentences[ii][jj]) + " ";
        }
        l_display += "\nSentence retrieved : ";

        for(int jj = 0; jj < results.m_trainResults[ii].size(); ++jj)
        {
            l_display+= QString::fromStdString(results.m_trainResults[ii][jj]) + " ";
        }
        l_display += "\nResults computed    : ";

        if(results.m_absoluteCCW.size() == results.m_trainSentences.size())
        {
            l_display += "CCW : "  + QString::number(results.m_absoluteCCW[ii]) + " ALL : " +  QString::number(results.m_absoluteAll[ii]) + "\n";
        }

        for(int jj = 0; jj < l_numberCharLine; ++jj)
        {
            l_display += "-";
        }

        l_display += "\n";
    }

    if(results.m_testResults.size() > 0)
    {
        l_display += "\n##################### TEST :\n";

        for(int ii = 0; ii < results.m_testResults.size(); ++ii)
        {
            l_display += "Sentence retrieved : ";

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
        }
    }

    m_uiInterface->tbResults->setPlainText(l_display);
    m_uiInterface->tbResults->verticalScrollBar()->setValue(m_uiInterface->tbResults->verticalScrollBar()->maximum());
}

void Interface::updateProgressBar(int currentValue, int valueMax, QString text)
{
    m_uiInterface->pbComputing->setMaximum(valueMax);
    m_uiInterface->pbComputing->setValue(currentValue);
    m_uiInterface->laStateComputing->setText(text);
}

void Interface::initPlot(int nbCurves, int sizeDim1Meaning, int sizeDim2Meaning, QString name)
{
    qDeleteAll(m_plotListX.begin(), m_plotListX.end());
    m_plotListX.clear();
    m_allValuesPlot.clear();
    m_nbSentencesDisplayed = 0;

    m_sizeDim1Meaning = sizeDim1Meaning;
    m_sizeDim2Meaning = sizeDim2Meaning;

    QVector<QString> l_labelsX;

    for(int ii = 0; ii < m_sizeDim1Meaning; ++ii)
    {
        l_labelsX << "S" + QString::number(ii+1);
    }

    QVector<QString> l_labelsY;
    l_labelsY << "-1" << "0" << "1";


    LanguageParameters l_language  = m_pWInterface->languageParameters();
    int l_nbCCW = l_language.m_CCW.split(" ").size();

    for(int ii = 0; ii < nbCurves; ++ii)
    {
        QCustomPlot *l_plotDisplay = new QCustomPlot(this);

        l_plotDisplay->setFixedHeight(100);
//        l_plotDisplay->setFixedWidth(sizeDim1Meaning*l_nbCCW*35);
        l_plotDisplay->setFixedWidth(sizeDim1Meaning*l_nbCCW*5);
//        l_plotDisplay->setFixedWidth(m_nbMaxNeuronsSentenceDisplayed*l_nbCCW*10);
        m_plotListX.push_back(l_plotDisplay);
        m_uiInterface->vlXPlot->addWidget(m_plotListX.back());
        m_plotListX.back()->addGraph();

        m_plotListX.back()->xAxis->setLabel("x");
        m_plotListX.back()->xAxis->setRange(0, sizeDim1Meaning);
//        m_plotListX.back()->xAxis->setRange(0, m_nbMaxNeuronsSentenceDisplayed);

        m_plotListX.back()->xAxis->setAutoTickStep(false);
        m_plotListX.back()->xAxis->setAutoTickLabels(false);
        m_plotListX.back()->xAxis->setTickVectorLabels(l_labelsX);
        m_plotListX.back()->xAxis->setTickStep(1);
        m_plotListX.back()->xAxis->setLabel("Sentences");


        m_plotListX.back()->yAxis->setLabel("y");
        m_plotListX.back()->yAxis->setRange(-1, 1);
        m_plotListX.back()->yAxis->setAutoTickStep(false);
        m_plotListX.back()->yAxis->setAutoTickLabels(false);
        m_plotListX.back()->yAxis->setTickVectorLabels(l_labelsY);
        m_plotListX.back()->yAxis->setTickStep(1.0);
        m_plotListX.back()->yAxis->setLabel("Neuron value");


//        m_plotListX.back()->graph(0)->setLineStyle(QCPGraph::lsLine);
//        m_plotListX.back()->graph(0)->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssDisc, 2));
    }
}


//void Interface::displayNeuronsActivities(QVector<QVector<double> > valuesNeurons)
//{
//    qDeleteAll(m_plotListX.begin(), m_plotListX.end());
//    m_plotListX.clear();
//    m_allValuesPlot.clear();
//}

void Interface::displayXMatrix(QVector<QVector<double> > *values, int currentSentenceId, int nbSentences)
{
    // wait (necessary to get the events)
//    QTime l_oDieTime = QTime::currentTime().addMSecs(10);
//    while( QTime::currentTime() < l_oDieTime)
//    {
//        QCoreApplication::processEvents(QEventLoop::AllEvents, 10);
//    }

    if(m_allValuesPlot.size() == 0)
    {
        m_timerDisplayNeurons.start();

        for(int ii = 0; ii < values->size(); ++ii)
        {
            QVector<double> l_init;
            m_allValuesPlot.push_back(l_init);
        }
    }

    for(int ii = 0; ii < values->size(); ++ii)
    {
        for(int jj = 0; jj < (*values)[ii].size(); ++jj)
        {
            m_allValuesPlot[ii].push_back((*values)[ii][jj]);
//            m_allXPlot.push_back();
        }
    }


//    int l_sizeS = (*values)[0].size();
//    int l_lenghtX = l_sizeS * m_nbMaxNeuronsSentenceDisplayed;

//    double l_startKey, l_endKey;
//    if(currentSentenceId <= m_nbMaxNeuronsSentenceDisplayed)
//    {
//        l_startKey = 0.0;
//        l_endKey = currentSentenceId;
//    }
//    else
//    {
//        l_startKey = currentSentenceId - m_nbMaxNeuronsSentenceDisplayed;
//        l_endKey = currentSentenceId;
//    }

//    qDebug() << "info : " << l_startKey << " " << l_endKey << " " << m_nbMaxNeuronsSentenceDisplayed << " " << currentSentenceId << " " << l_sizeS << " " << l_lenghtX << " " << m_allValuesPlot.size() << " " << m_allValuesPlot[0].size();


    if(m_timerDisplayNeurons.elapsed() > 250)// || ++m_nbSentencesDisplayed == nbSentences)
    {
        m_timerDisplayNeurons.restart();
        QVector<double> l_xValues;
        for(int ii = 0; ii < m_allValuesPlot[0].size(); ++ii)
        {
            l_xValues.push_back(ii/static_cast<double>(m_sizeDim2Meaning));
        }
//        for(int ii = 0; ii < l_lenghtX; ++ii)
//        {
//            l_xValues.push_back(ii/static_cast<double>(l_lenghtX));
//        }

        for(int ii = 0; ii < m_plotListX.size(); ++ii)
        {

//            QVector<double> l_currentPart;
//            for(int jj = l_startKey; jj < l_startKey + l_lenghtX; ++jj)
//            {
//                l_currentPart <<  m_allValuesPlot[ii][jj];
//            }

            QPen l_pen(Qt::blue);
            l_pen.setWidthF(4);
//            m_plotListX[ii]->xAxis->setRange(l_startKey, l_endKey);
            m_plotListX[ii]->graph(0)->setPen(l_pen);
//            m_plotListX[ii]->graph(0)->addData();
            m_plotListX[ii]->graph(0)->setData(l_xValues, m_allValuesPlot[ii]);
//            m_plotListX[ii]->graph(0)->setData(l_xValues,l_currentPart);
//            m_plotListX[ii]->graph(0)->removeDataBefore(l_startKey);
//            m_plotListX[ii]->graph(0)->removeDataAfter(l_endKey);
            m_plotListX[ii]->replot();

        }
    }

    m_neuronDisplayMutex.unlock();

    delete values;
}


void Interface::cleanResultsDisplay()
{
    m_uiInterface->tbResults->clear();
}

void Interface::displayLogInfo(QString info)
{
    if(m_logFile.isOpen())
    {
        QTextStream l_stream(&m_logFile);
        l_stream << info;
    }

    m_uiInterface->tbInfos->setPlainText(m_uiInterface->tbInfos->toPlainText() + info);
    m_uiInterface->tbInfos->verticalScrollBar()->setValue(m_uiInterface->tbInfos->verticalScrollBar()->maximum());
}

void Interface::displayOutputMatrix(cv::Mat output)
{
    QVector<QVector<QVector<double> > > l_sentences; // dim 1 -> sentences / dim 2 -> CCW / dim 3 -> values

    // delete previous plots and labels
        qDeleteAll(m_plotListOutput.begin(), m_plotListOutput.end());
        qDeleteAll(m_plotLabelListOutput.begin(), m_plotLabelListOutput.end());
        m_plotListOutput.clear();
        m_plotLabelListOutput.clear();

        LanguageParameters l_language = m_pWInterface->languageParameters();
        QStringList l_CCW = l_language.m_CCW.split(' ');
        l_CCW << "X";


    // create color for each CCW
        QVector<QColor> l_colors;
        int l_nbLoop = 0;
        while(l_colors.size() < l_CCW.size())
        {
            ++l_nbLoop;;
            bool l_addColor = true;

            int l_r = rand()%240;
            int l_g = rand()%255;
            int l_b = rand()%255;

            if(l_nbLoop < 500)
            {
                for(int ii = 0; ii < l_colors.size(); ++ii)
                {
                    int l_diffRed = l_colors[ii].red()-l_r;
                    if(l_diffRed < 0)
                    {
                        l_diffRed = -l_diffRed;
                    }
                    int l_diffGreen = l_colors[ii].green()-l_r;
                    if(l_diffGreen < 0)
                    {
                        l_diffGreen = - l_diffGreen;
                    }
                    int l_diffBlue = l_colors[ii].blue()-l_r;
                    if(l_diffBlue < 0)
                    {
                        l_diffBlue = - l_diffBlue;
                    }

                    if(l_diffRed + l_diffGreen + l_diffBlue < 120)
                    {
                        l_addColor = false;
                        break;
                    }
                }
            }

            if(l_addColor)
            {
                l_colors << QColor(l_r,l_g,l_b);
            }

        }

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
            l_palette.setColor(l_labelCCW->foregroundRole(), l_colors[ii]);
            l_palette.setColor(l_labelCCW->backgroundRole(), Qt::white);
            l_labelCCW->setAutoFillBackground (true);
            l_labelCCW->setPalette(l_palette);
            l_labelCCW->setText(l_CCW[ii]);
            l_labelCCW->setAlignment(Qt::AlignCenter);
            m_plotLabelListOutput.push_back(l_labelCCW);
            m_uiInterface->hlLabelsOutputPlot->addWidget(m_plotLabelListOutput.back());
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
            l_textLabelSentence->setText("Sentence " + QString::number(ii+1));
            l_textLabelSentence->setFont(QFont(font().family(), 8)); // make font a bit larger


            m_plotListOutput.push_back(l_plotDisplay);
            m_uiInterface->vlOutputPlot->addWidget(m_plotListOutput.back());

            double l_maxY = DBL_MIN;
            double l_minY = DBL_MAX;

            for(int jj = 0; jj < l_sentences[0].size(); ++jj)
            {
                m_plotListOutput.back()->addGraph();
                QPen l_pen(l_colors[jj]);
                l_pen.setWidthF(3);
                m_plotListOutput[ii]->graph(jj)->setPen(l_pen);
                m_plotListOutput[ii]->graph(jj)->setData(l_xValues, l_sentences[ii][jj]);

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

            m_plotListOutput.back()->xAxis->setRange(0, output.size[1]);
            m_plotListOutput.back()->yAxis->setRange(l_minY,l_maxY);
            m_plotListOutput.back()->replot();
        }
}

void Interface::displayTrainInputMatrix(cv::Mat trainMeaning, cv::Mat trainSentence)
{
    QVector<QVector<QVector<double> > > l_sentences; // dim 1 -> sentences / dim 2 -> CCW / dim 3 -> values
    QVector<QVector<QVector<double> > > l_meaning; // dim 1 -> sentences / dim 2 -> Structure / dim 3 -> values

    // delete previous plots and labels
        qDeleteAll(m_plotListTrainSentenceInput.begin(), m_plotListTrainSentenceInput.end());
        qDeleteAll(m_plotListTrainMeaningInput.begin(), m_plotListTrainMeaningInput.end());
        qDeleteAll(m_plotLabelListTrainSentenceInput.begin(), m_plotLabelListTrainSentenceInput.end());
        m_plotListTrainSentenceInput.clear();
        m_plotListTrainMeaningInput.clear();
        m_plotLabelListTrainSentenceInput.clear();

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

    // create color for each CCW
        QVector<QColor> l_colors;
        int l_nbLoop = 0;
        while(l_colors.size() < l_CCW.size())
        {
            ++l_nbLoop;;
            bool l_addColor = true;

            int l_r = rand()%255;
            int l_g = rand()%255;
            int l_b = rand()%205;

            if(l_nbLoop < 500)
            {
                for(int ii = 0; ii < l_colors.size(); ++ii)
                {
                    int l_diffRed = l_colors[ii].red()-l_r;
                    if(l_diffRed < 0)
                    {
                        l_diffRed = - l_diffRed;
                    }
                    int l_diffGreen = l_colors[ii].green()-l_r;
                    if(l_diffGreen < 0)
                    {
                        l_diffGreen = - l_diffGreen;
                    }
                    int l_diffBlue = l_colors[ii].blue()-l_r;
                    if(l_diffBlue < 0)
                    {
                        l_diffBlue = - l_diffBlue;
                    }

                    if(l_diffRed + l_diffGreen + l_diffBlue < 80)
                    {
                        l_addColor = false;
                        break;
                    }
                }
            }

            if(l_addColor)
            {
                l_colors << QColor(l_r,l_g,l_b);
            }
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
            l_palette.setColor(l_labelCCW->foregroundRole(), l_colors[ii]);
            l_palette.setColor(l_labelCCW->backgroundRole(), Qt::white);
            l_labelCCW->setAutoFillBackground (true);
            l_labelCCW->setPalette(l_palette);
            l_labelCCW->setText(l_CCW[ii]);
            l_labelCCW->setAlignment(Qt::AlignCenter);
            m_plotLabelListTrainSentenceInput.push_back(l_labelCCW);
            m_uiInterface->hlLabelsInputPlot->addWidget(m_plotLabelListTrainSentenceInput.back());
        }

    // create the plots

    qDebug() << l_meaning.size() << " " << l_meaning[0].size() << " " << l_meaning[0][0].size() ;
        for(int ii = 0; ii < l_sentences.size(); ++ii)
        {
            // add sentences
                QCustomPlot *l_plotDisplaySentence = new QCustomPlot(this);
                l_plotDisplaySentence->setFixedHeight(150);
                l_plotDisplaySentence->setFixedWidth(trainSentence.size[2]*35);

                m_plotListTrainSentenceInput.push_back(l_plotDisplaySentence);
                m_uiInterface->vlInputPlot->addWidget(m_plotListTrainSentenceInput.back());

                QCPItemText *l_textLabelSentence = new QCPItemText(l_plotDisplaySentence);
                l_plotDisplaySentence->addItem(l_textLabelSentence);
                l_textLabelSentence->setPositionAlignment(Qt::AlignTop|Qt::AlignHCenter);
                l_textLabelSentence->position->setType(QCPItemPosition::ptAxisRectRatio);
                l_textLabelSentence->position->setCoords(0.5, 0); // place position at center/top of axis rect
                l_textLabelSentence->setText("Sentence " + QString::number(ii+1));
                l_textLabelSentence->setFont(QFont(font().family(), 8)); // make font a bit larger


                double l_maxY = DBL_MIN;
                double l_minY = DBL_MAX;

                for(int jj = 0; jj < l_sentences[0].size(); ++jj)
                {
                    m_plotListTrainSentenceInput.back()->addGraph();
                    QPen l_pen(l_colors[jj]);
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
                m_uiInterface->vlInputPlot->addWidget(m_plotListTrainMeaningInput.back());

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
                l_plotDisplayMeaning->plotLayout()->addElement(0, 1, colorScale); // add it to the right of the main axis rect
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

InterfaceWorker::InterfaceWorker() : m_gridSearch(new GridSearchQt(m_model)), m_nbOfCorpus(0)
{
    qRegisterMetaType<ReservoirParameters>("ReservoirParameters");
    qRegisterMetaType<LanguageParameters>("LanguageParameters");
    qRegisterMetaType<ModelParametersQt>("ModelParametersQt");
    qRegisterMetaType<ResultsDisplayReservoir>("ResultsDisplayReservoir");
    qRegisterMetaType<QVector<QVector<double> > >("QVector<QVector<double> >");
    qRegisterMetaType<cv::Mat>("cv::Mat");
}

InterfaceWorker::~InterfaceWorker()
{
    delete m_gridSearch;
}

GridSearchQt *InterfaceWorker::gridSearch() const
{
    return m_gridSearch;
}

ModelQt *InterfaceWorker::model()
{
    return &m_model;
}

LanguageParameters InterfaceWorker::languageParameters() const
{
    return m_languageParameters;
}

void InterfaceWorker::addCorpus(QString corpusPath)
{
    m_corpusList.push_back(corpusPath);

    std::vector<std::string> l_stringListCorpus;

    for(int ii = 0; ii < m_corpusList.size(); ++ii)
    {
        l_stringListCorpus.push_back(m_corpusList[ii].toStdString());
    }

    m_gridSearch->setCorpusList(l_stringListCorpus);
    ++m_nbOfCorpus;
}

void InterfaceWorker::removeCorpus(int index)
{
    if(index == -1)
    {
        return;
    }

    m_corpusList.removeAt(index);

    std::vector<std::string> l_stringListCorpus;

    for(int ii = 0; ii < m_corpusList.size(); ++ii)
    {
        l_stringListCorpus.push_back(m_corpusList[ii].toStdString());
    }

    m_gridSearch->setCorpusList(l_stringListCorpus);
    --m_nbOfCorpus;
}

void InterfaceWorker::updateReservoirParameters(ReservoirParameters newParams)
{
    m_reservoirParameters = newParams;
}

void InterfaceWorker::updateLanguageParameters(LanguageParameters newParams)
{
    m_languageParameters = newParams;
}

void InterfaceWorker::start()
{
    if(m_nbOfCorpus <= 0)
    {
        std::cerr << "Cannot start, no corpus is defined. " << std::endl;
        return;
    }

    emit startInitDisplaySignal();

    // define language parameters
        Sentence l_CCW, l_structure;
        QStringList l_CCWList = m_languageParameters.m_CCW.split(" ");
        for(int ii = 0; ii < l_CCWList.size(); ++ii)
        {
            l_CCW.push_back(l_CCWList[ii].toStdString());
        }
        QStringList l_structureList = m_languageParameters.m_structure.split(" ");
        for(int ii = 0; ii < l_structureList.size(); ++ii)
        {
            l_structure.push_back(l_structureList[ii].toStdString());
        }

        m_model.setCCWAndStructure(l_CCW, l_structure);

    // define all grid search parameters
        m_gridSearch->deleteParameterValues();
        m_gridSearch->setCudaParameters(m_reservoirParameters.m_useCuda, m_reservoirParameters.m_useCuda);

        bool l_operationValid;
        int l_OperationInvalid = 0;


        bool l_onlyStartValue = m_reservoirParameters.m_useOnlyStartValue;

        if(m_reservoirParameters.m_neuronsEnabled)
        {
            l_operationValid = m_gridSearch->setParameterValues(GridSearchQt::NEURONS_NB, m_reservoirParameters.m_neuronsStart, m_reservoirParameters.m_neuronsEnd,
                                                                m_reservoirParameters.m_neuronsOperation.toStdString(), l_onlyStartValue, m_reservoirParameters.m_neuronsNbOfUses);
            emit displayValidityOperationSignal(l_operationValid, GridSearchQt::NEURONS_NB);
            if(!l_operationValid)
            {
                ++l_OperationInvalid;
            }
        }
        if(m_reservoirParameters.m_leakRateEnabled)
        {
            l_operationValid = m_gridSearch->setParameterValues(GridSearchQt::LEAK_RATE,m_reservoirParameters.m_leakRateStart, m_reservoirParameters.m_leakRateEnd,
                                                                m_reservoirParameters.m_leakRateOperation.toStdString(), l_onlyStartValue, m_reservoirParameters.m_leakRateNbOfUses);
            emit displayValidityOperationSignal(l_operationValid, GridSearchQt::LEAK_RATE);
            if(!l_operationValid)
            {
                ++l_OperationInvalid;
            }
        }
        if(m_reservoirParameters.m_issEnabled)
        {
            l_operationValid = m_gridSearch->setParameterValues(GridSearchQt::INPUT_SCALING,m_reservoirParameters.m_issStart, m_reservoirParameters.m_issEnd,
                                                                m_reservoirParameters.m_issOperation.toStdString(), l_onlyStartValue, m_reservoirParameters.m_issNbOfUses);
            emit displayValidityOperationSignal(l_operationValid, GridSearchQt::INPUT_SCALING);
            if(!l_operationValid)
            {
                ++l_OperationInvalid;
            }
        }
        if(m_reservoirParameters.m_spectralRadiusEnabled)
        {
            l_operationValid = m_gridSearch->setParameterValues(GridSearchQt::SPECTRAL_RADIUS,    m_reservoirParameters.m_spectralRadiusStart,  m_reservoirParameters.m_spectralRadiusEnd,
                                                                m_reservoirParameters.m_spectralRadiusOperation.toStdString(), l_onlyStartValue, m_reservoirParameters.m_spectralRadiusNbOfUses);
            emit displayValidityOperationSignal(l_operationValid, GridSearchQt::SPECTRAL_RADIUS);
            if(!l_operationValid)
            {
                ++l_OperationInvalid;
            }
        }
        if(m_reservoirParameters.m_ridgeEnabled)
        {
            l_operationValid = m_gridSearch->setParameterValues(GridSearchQt::RIDGE,m_reservoirParameters.m_ridgeStart, m_reservoirParameters.m_ridgeEnd,
                                                                m_reservoirParameters.m_ridgeOperation.toStdString(), l_onlyStartValue, m_reservoirParameters.m_ridgeNbOfUses);
            emit displayValidityOperationSignal(l_operationValid, GridSearchQt::RIDGE);
            if(!l_operationValid)
            {
                ++l_OperationInvalid;
            }
        }
        if(m_reservoirParameters.m_sparcityEnabled)
        {
            l_operationValid = m_gridSearch->setParameterValues(GridSearchQt::SPARCITY, m_reservoirParameters.m_sparcityStart,  m_reservoirParameters.m_sparcityEnd,
                                                                m_reservoirParameters.m_sparcityOperation.toStdString(), l_onlyStartValue, m_reservoirParameters.m_sparcityNbOfUses);
            emit displayValidityOperationSignal(l_operationValid, GridSearchQt::SPARCITY);
            if(!l_operationValid)
            {
                ++l_OperationInvalid;
            }
        }

        if(l_OperationInvalid != 0)
        {
            std::cerr << "Cannot start, " << l_OperationInvalid << " operation are invalid. (Displayed in red)" << std::endl;
            return;
        }

        bool l_doTrain, l_doTest;

        switch(m_reservoirParameters.m_action)
        {
            case TRAINING_RES :
                l_doTrain = true;
                l_doTest = false;
            break;
            case TEST_RES :
                l_doTrain = false;
                l_doTest = true;
            break;
            case BOTH_RES :
                l_doTrain = true;
                l_doTest = true;
            break;
        }

    // launch reservoir computing
    lockInterfaceSignal(true);
        m_gridSearch->launchTrainWithAllParameters("../data/Results/interfacetest.txt", "../data/Results/interfacetest_raw.txt", l_doTrain, l_doTest, m_reservoirParameters.m_useLoadedTraining);
    lockInterfaceSignal(false);

    emit endTrainingSignal(true);
}

void InterfaceWorker::stop()
{
    lockInterfaceSignal(false);
}

void InterfaceWorker::saveLastTraining(QString pathDirectory)
{
    if(pathDirectory.size() > 0)
    {
        m_model.saveTraining(pathDirectory.toStdString());
        qDebug() << "Training saved in the directory : " << pathDirectory;
    }
}

void InterfaceWorker::loadTraining(QString pathDirectory)
{
    if(pathDirectory.size() > 0)
    {
        m_model.loadTraining(pathDirectory.toStdString());
        qDebug() << "Training loaded in the directory : " << pathDirectory;
    }
}
