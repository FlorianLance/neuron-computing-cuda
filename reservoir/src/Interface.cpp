

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

    // init gramar / structure (TEMP)
        QStringList l_listGrammar, l_listStructure;
        l_listGrammar << "and s of the to . -ed -ing -s by it that was did , from";
        l_listGrammar << "-ga -ni -wo -yotte -o -to sore";
        m_uiInterface->cbGrammar->addItems(l_listGrammar);
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

        // checkbox
        QObject::connect(m_uiInterface->cbNeurons,              SIGNAL(stateChanged(int)), SLOT(updateReservoirParameters(int)));
        QObject::connect(m_uiInterface->cbLeakRate,             SIGNAL(stateChanged(int)), SLOT(updateReservoirParameters(int)));
        QObject::connect(m_uiInterface->cbIS,                   SIGNAL(stateChanged(int)), SLOT(updateReservoirParameters(int)));
        QObject::connect(m_uiInterface->cbSpectralRadius,       SIGNAL(stateChanged(int)), SLOT(updateReservoirParameters(int)));
        QObject::connect(m_uiInterface->cbRidge,                SIGNAL(stateChanged(int)), SLOT(updateReservoirParameters(int)));
        QObject::connect(m_uiInterface->cbSparcity,             SIGNAL(stateChanged(int)), SLOT(updateReservoirParameters(int)));
        QObject::connect(m_uiInterface->cbTrainingFile,         SIGNAL(stateChanged(int)), SLOT(updateReservoirParameters(int)));
        QObject::connect(m_uiInterface->cbOnlyStartValue,       SIGNAL(stateChanged(int)), SLOT(updateReservoirParameters(int)));

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
        QObject::connect(m_uiInterface->cbGrammar,      SIGNAL(currentIndexChanged(int)), SLOT(updateLanguageParameters(int)));
        QObject::connect(m_uiInterface->cbGrammar,      SIGNAL(editTextChanged(QString)), SLOT(updateLanguageParameters(QString)));        

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
        QObject::connect(m_uiInterface->cbEnableDisplay, SIGNAL(clicked(bool)), l_model->reservoir(), SLOT(disableMaxOmpThreadNumber(bool)));
        QObject::connect(m_uiInterface->cbEnableDisplay, SIGNAL(clicked(bool)), l_model->reservoir(), SLOT(enableDisplay(bool)));
//        QObject::connect(l_model->reservoir(), SIGNAL(sendMatriceImage2Display(QImage)), m_imageDisplay, SLOT(refreshDisplay(QImage)));
        QObject::connect(l_model->reservoir(), SIGNAL(sendMatriceData(QVector<QVector<double> >)), this, SLOT(plotData(QVector<QVector<double> >)));
        QObject::connect(l_model->reservoir(), SIGNAL(sendInfoPlot(int,int,int,QString)), this, SLOT(initPlot(int,int,int,QString)));
        QObject::connect(l_model->reservoir(), SIGNAL(sendLogInfo(QString)), this, SLOT(displayLogInfo(QString)));
        QObject::connect(l_model->reservoir(), SIGNAL(sendOutputMatrix(cv::Mat)), this, SLOT(displayOutputMatrix(cv::Mat)));

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
    l_params.m_grammar = m_uiInterface->cbGrammar->currentText();
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

    m_uiInterface->cbGrammar->setDisabled(lock);
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


void Interface::plotData(QVector<QVector<double> > values)
{
    if(m_allValuesPlot.size() == 0)
    {
        for(int ii = 0; ii < values.size(); ++ii)
        {
            QVector<double> l_init;
            m_allValuesPlot.push_back(l_init);
        }
    }

    for(int ii = 0; ii < values.size(); ++ii)
    {
        for(int jj = 0; jj < values[ii].size(); ++jj)
        {
            m_allValuesPlot[ii].push_back(values[ii][jj]);
        }
    }


//    for(int ii = 0; ii < )

//    qDebug() << "plot :" << values.size() << " " << values[0].size();
//    qDebug() << "start";
    QVector<double> l_xValues;

    for(int ii = 0; ii < m_allValuesPlot[0].size(); ++ii)
    {
        l_xValues.push_back(ii/static_cast<double>(m_sizeDim2Meaning));
    }

    for(int ii = 0; ii < m_plotListX.size(); ++ii)
    {
        m_plotListX[ii]->graph(0)->setPen(QPen(Qt::blue));
        m_plotListX[ii]->graph(0)->setData(l_xValues, m_allValuesPlot[ii]);
        m_plotListX[ii]->replot();
    }
}

void Interface::initPlot(int nbCurves, int sizeDim1Meaning, int sizeDim2Meaning, QString name)
{
    qDeleteAll(m_plotListX.begin(), m_plotListX.end());
    m_plotListX.clear();
    m_allValuesPlot.clear();

    m_sizeDim1Meaning = sizeDim1Meaning;
    m_sizeDim2Meaning = sizeDim2Meaning;

    QVector<QString> l_labelsX;

    for(int ii = 0; ii < m_sizeDim1Meaning; ++ii)
    {
        l_labelsX << "S" + QString::number(ii+1);
    }

    QVector<QString> l_labelsY;
    l_labelsY << "-1" << "0" << "1";

    for(int ii = 0; ii < nbCurves; ++ii)
    {
        QCustomPlot *l_plotDisplay = new QCustomPlot(this);

        l_plotDisplay->setFixedHeight(100);
        l_plotDisplay->setFixedWidth(sizeDim1Meaning*60);
        m_plotListX.push_back(l_plotDisplay);
        m_uiInterface->vlXPlot->addWidget(m_plotListX.back());
        m_plotListX.back()->addGraph();

        m_plotListX.back()->xAxis->setLabel("x");
        m_plotListX.back()->xAxis->setRange(0, sizeDim1Meaning);

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
    }
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
    qDeleteAll(m_plotListOutput.begin(), m_plotListOutput.end());
    m_plotListOutput.clear();

    QVector<QVector<QVector<double> > > l_sentences; // dim 1 -> sentences / dim 2 -> grammar / dim 3 -> values
    QVector<QPen> l_pens;
    QVector<double> l_xValues;
    for(int ii = 0; ii < output.size[1]; ++ii)
    {
        l_xValues << ii;
    }

    for(int ii = 0; ii < output.size[0]; ++ii)
    {
        QVector<QVector<double> > l_grammars;

        for(int jj = 0; jj  < output.size[2]; ++jj)
        {
            QVector<double> l_values;

            for(int kk = 0; kk < output.size[1]; ++kk)
            {
                l_values << static_cast<double>(output.at<float>(ii,kk,jj));
            }
            l_grammars << l_values;


            l_pens << QPen(QColor(rand()%255,rand()%255,rand()%255));
        }
        l_sentences << l_grammars;
    }

    qDebug() << l_sentences.size() << " " << l_sentences[0].size() << " " << l_sentences[0][0].size();

//    QVector<QString> l_labelsX;

//    for(int ii = 0; ii < m_sizeDim1Meaning; ++ii)
//    {
//        l_labelsX << "S" + QString::number(ii+1);
//    }

//    QVector<QString> l_labelsY;
//    l_labelsY << "-1" << "0" << "1";


    QVector<QString> l_labelsY;
    l_labelsY << "-0.2" << "0.4" << "1.0" << "1.6" << "2.2";

    for(int ii = 0; ii < l_sentences.size(); ++ii)
    {
        QCustomPlot *l_plotDisplay = new QCustomPlot(this);
        l_plotDisplay->setFixedHeight(150);
        l_plotDisplay->setFixedWidth(output.size[2]*100);
        m_plotListOutput.push_back(l_plotDisplay);
        m_uiInterface->vlOutputPlot->addWidget(m_plotListOutput.back());

        for(int jj = 0; jj < l_sentences[0].size(); ++jj)
        {
            m_plotListOutput.back()->addGraph();
            m_plotListOutput[ii]->graph(jj)->setPen(l_pens[jj]);
            m_plotListOutput[ii]->graph(jj)->setData(l_xValues, l_sentences[ii][jj]);

        }

//        m_plotListOutput.back()->xAxis->setTickStep(1);
        m_plotListOutput.back()->xAxis->setRange(0, output.size[1]);

        m_plotListOutput.back()->yAxis->setAutoTickStep(false);
        m_plotListOutput.back()->yAxis->setAutoTickLabels(false);
        m_plotListOutput.back()->yAxis->setRange(-0.2, 2.2);
        m_plotListOutput.back()->yAxis->setTickStep(0.6);
        m_plotListOutput.back()->yAxis->setTickVectorLabels(l_labelsY);
        m_plotListOutput.back()->replot();

//        m_plotListX.back()->addGraph();

//        m_plotListX.back()->xAxis->setLabel("x");
//        m_plotListX.back()->xAxis->setRange(0, sizeDim1Meaning);

//        m_plotListX.back()->xAxis->setAutoTickStep(false);
//        m_plotListX.back()->xAxis->setAutoTickLabels(false);
//        m_plotListX.back()->xAxis->setTickVectorLabels(l_labelsX);
//        m_plotListX.back()->xAxis->setTickStep(1);
//        m_plotListX.back()->xAxis->setLabel("Sentences");


//        m_plotListX.back()->yAxis->setLabel("y");
//        m_plotListX.back()->yAxis->setRange(-1, 1);
//        m_plotListX.back()->yAxis->setAutoTickStep(false);
//        m_plotListX.back()->yAxis->setAutoTickLabels(false);
//        m_plotListX.back()->yAxis->setTickVectorLabels(l_labelsY);
//        m_plotListX.back()->yAxis->setTickStep(1.0);
//        m_plotListX.back()->yAxis->setLabel("Neuron value");
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
        Sentence l_grammar, l_structure;
        QStringList l_grammarList = m_languageParameters.m_grammar.split(" ");
        for(int ii = 0; ii < l_grammarList.size(); ++ii)
        {
            l_grammar.push_back(l_grammarList[ii].toStdString());
        }
        QStringList l_structureList = m_languageParameters.m_structure.split(" ");
        for(int ii = 0; ii < l_structureList.size(); ++ii)
        {
            l_structure.push_back(l_structureList[ii].toStdString());
        }

        m_model.setGrammar(l_grammar, l_structure);

    // define all grid search parameters
        m_gridSearch->deleteParameterValues();
        m_gridSearch->setCudaParameters(true, true);


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
        qDebug() << "Trianing saved in the directory : " << pathDirectory;
    }
}

void InterfaceWorker::loadTraining(QString pathDirectory)
{
    if(pathDirectory.size() > 0)
    {
        m_model.loadTraining(pathDirectory.toStdString());
        qDebug() << "Trianing loaded in the directory : " << pathDirectory;
    }
}
