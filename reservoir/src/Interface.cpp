

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





Interface::Interface() : m_uiInterface(new Ui::UI_Reservoir)
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

    // init worker
        m_pWInterface = new InterfaceWorker();

    // init connections
        // push button
        QObject::connect(m_uiInterface->pbStart,        SIGNAL(clicked()), m_pWInterface, SLOT(start()));
        QObject::connect(m_uiInterface->pbStop,         SIGNAL(clicked()), m_pWInterface, SLOT(stop()));
        QObject::connect(m_uiInterface->pbAddCorpus,    SIGNAL(clicked()), this, SLOT(addCorpus()));
        QObject::connect(m_uiInterface->pbRemoveCorpus, SIGNAL(clicked()), this, SLOT(removeCorpus()));
        QObject::connect(m_uiInterface->pbSaveLastTrainingFile, SIGNAL(clicked()), this, SLOT(saveTraining()));

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

        // checkbox
        QObject::connect(m_uiInterface->cbNeurons,              SIGNAL(stateChanged(int)), SLOT(updateReservoirParameters(int)));
        QObject::connect(m_uiInterface->cbLeakRate,             SIGNAL(stateChanged(int)), SLOT(updateReservoirParameters(int)));
        QObject::connect(m_uiInterface->cbIS,                   SIGNAL(stateChanged(int)), SLOT(updateReservoirParameters(int)));
        QObject::connect(m_uiInterface->cbSpectralRadius,       SIGNAL(stateChanged(int)), SLOT(updateReservoirParameters(int)));
        QObject::connect(m_uiInterface->cbRidge,                SIGNAL(stateChanged(int)), SLOT(updateReservoirParameters(int)));
        QObject::connect(m_uiInterface->cbSparcity,             SIGNAL(stateChanged(int)), SLOT(updateReservoirParameters(int)));

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

//            QObject::connect(this,  SIGNAL(stopLoop()), m_pWViewer, SLOT(stopLoop()));
//            QObject::connect(this, SIGNAL(setModFilePath(bool,int,QString)), m_pWViewer, SLOT(setModFile(bool,int,QString)));
//            QObject::connect(this, SIGNAL(setSeqFilePath(bool,int,QString)), m_pWViewer, SLOT(setSeqFile(bool,int,QString)));
//            QObject::connect(this, SIGNAL(setCorrFilePath(bool,int,QString)), m_pWViewer, SLOT(setCorrFilePath(bool,int,QString)));
//            QObject::connect(m_pWViewer, SIGNAL(sendAnimationPathFile(QString,QString,QString)), this, SLOT(updateAnimationPathFileDisplay(QString,QString,QString)));
//            QObject::connect(this, SIGNAL(deleteAnimation(bool,int)), m_pWViewer, SLOT(deleteAnimation(bool,int)));
//            QObject::connect(this, SIGNAL(addAnimation(bool)), m_pWViewer, SLOT(addAnimation(bool)));
//            QObject::connect(m_pWViewer, SIGNAL(sendOffsetAnimation(SWAnimationSendDataPtr)),m_pGLMultiObject, SLOT(setAnimationOffset(SWAnimationSendDataPtr)),Qt::DirectConnection);
////            QObject::connect(m_pWViewer, SIGNAL(sendOffsetAnimation(SWAnimationSendDataPtr)),m_pGLMultiObject, SLOT(setAnimationOffset(SWAnimationSendDataPtr)));
//            QObject::connect(m_pWViewer, SIGNAL(startAnimation(bool,int)), m_pGLMultiObject, SLOT(beginAnimation(bool,int)));

//            QObject::connect(m_pWViewer, SIGNAL(drawSceneSignal()), m_pGLMultiObject, SLOT(updateGL()));

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
    QString l_sPathCorpus = QFileDialog::getOpenFileName(this, "Load corpus file", QString(), "Corpus file (*.txt)");
    m_uiInterface->lwCorpus->addItem(l_sPathCorpus);

    // send item
    emit addCorpusSignal(l_sPathCorpus);
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

    // send directory path
    emit saveTrainingSignal(l_sPathTrainingFile);
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

    m_uiInterface->pbStart->setDisabled(lock);
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



InterfaceWorker::InterfaceWorker() : m_gridSearch(m_model), m_nbOfCorpus(0)
{
    qRegisterMetaType<ReservoirParameters>("ReservoirParameters");
    qRegisterMetaType<LanguageParameters>("LanguageParameters");
}

void InterfaceWorker::addCorpus(QString corpusPath)
{
    m_corpusList.push_back(corpusPath);

    std::vector<std::string> l_stringListCorpus;

    for(int ii = 0; ii < m_corpusList.size(); ++ii)
    {
        l_stringListCorpus.push_back(m_corpusList[ii].toStdString());
    }

    m_gridSearch.setCorpusList(l_stringListCorpus);
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

    m_gridSearch.setCorpusList(l_stringListCorpus);
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
    qDebug() << "start";

    if(m_nbOfCorpus <= 0)
    {
        std::cerr << "Cannot start, no corpus is defined. " << std::endl;
        return;
    }

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
        m_gridSearch.deleteParameterValues();
        m_gridSearch.setCudaParameters(true, true);

        bool l_operationValid;
        int l_OperationInvalid = 0;

        if(m_reservoirParameters.m_neuronsEnabled)
        {
            l_operationValid = m_gridSearch.setParameterValues(GridSearch::NEURONS_NB,         m_reservoirParameters.m_neuronsStart,         m_reservoirParameters.m_neuronsEnd, m_reservoirParameters.m_neuronsOperation.toStdString());
            emit displayValidityOperationSignal(l_operationValid, GridSearch::NEURONS_NB);
            if(!l_operationValid)
            {
                ++l_OperationInvalid;
            }
        }
        if(m_reservoirParameters.m_leakRateEnabled)
        {
            l_operationValid = m_gridSearch.setParameterValues(GridSearch::LEAK_RATE,          m_reservoirParameters.m_leakRateStart,        m_reservoirParameters.m_leakRateEnd, m_reservoirParameters.m_leakRateOperation.toStdString());
            emit displayValidityOperationSignal(l_operationValid, GridSearch::LEAK_RATE);
            if(!l_operationValid)
            {
                ++l_OperationInvalid;
            }
        }
        if(m_reservoirParameters.m_issEnabled)
        {
            l_operationValid = m_gridSearch.setParameterValues(GridSearch::INPUT_SCALING,      m_reservoirParameters.m_issStart,             m_reservoirParameters.m_issEnd, m_reservoirParameters.m_issOperation.toStdString());
            emit displayValidityOperationSignal(l_operationValid, GridSearch::INPUT_SCALING);
            if(!l_operationValid)
            {
                ++l_OperationInvalid;
            }
        }
        if(m_reservoirParameters.m_spectralRadiusEnabled)
        {
            l_operationValid = m_gridSearch.setParameterValues(GridSearch::SPECTRAL_RADIUS,    m_reservoirParameters.m_spectralRadiusStart,  m_reservoirParameters.m_spectralRadiusEnd, m_reservoirParameters.m_spectralRadiusOperation.toStdString());
            emit displayValidityOperationSignal(l_operationValid, GridSearch::SPECTRAL_RADIUS);
            if(!l_operationValid)
            {
                ++l_OperationInvalid;
            }
        }
        if(m_reservoirParameters.m_ridgeEnabled)
        {
            l_operationValid = m_gridSearch.setParameterValues(GridSearch::RIDGE,              m_reservoirParameters.m_ridgeStart,           m_reservoirParameters.m_ridgeEnd, m_reservoirParameters.m_ridgeOperation.toStdString());
            emit displayValidityOperationSignal(l_operationValid, GridSearch::RIDGE);
            if(!l_operationValid)
            {
                ++l_OperationInvalid;
            }
        }
        if(m_reservoirParameters.m_sparcityEnabled)
        {
            l_operationValid = m_gridSearch.setParameterValues(GridSearch::SPARCITY,           m_reservoirParameters.m_sparcityStart,        m_reservoirParameters.m_sparcityEnd, m_reservoirParameters.m_sparcityOperation.toStdString());
            emit displayValidityOperationSignal(l_operationValid, GridSearch::SPARCITY);
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

    // launch reservoir computing
    lockInterfaceSignal(true);
        m_gridSearch.launchTrainWithAllParameters("../data/Results/interfacetest.txt", "../data/Results/random_res/interfacetest_raw.txt");
    lockInterfaceSignal(false);
}

void InterfaceWorker::stop()
{
    lockInterfaceSignal(false);

    qDebug() << "unlock interface" << m_nbOfCorpus;
}