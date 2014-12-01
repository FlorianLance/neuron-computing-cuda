

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
//    this->setWindowIcon(QIcon(QString("../data/images/logos/icon_swooz_viewer.png")));

    // middle container
//        QHBoxLayout *l_pGLContainerLayout = new QHBoxLayout();
//        m_pGLContainer = new QWidget();
//        QGLFormat l_glFormat;
//        l_glFormat.setVersion( 4, 3 );
//        l_glFormat.setProfile(  QGLFormat::CompatibilityProfile);
//        l_glFormat.setSampleBuffers( true );
//        QGLContext *l_glContext = new QGLContext(l_glFormat);
//        m_pGLMultiObject = new SWGLMultiObjectWidget(l_glContext, m_pGLContainer);
//        l_pGLContainerLayout->addWidget(m_pGLMultiObject);
//        l_pGLContainerLayout->layout()->setContentsMargins(0,0,0,0);
//        m_pGLContainer->setLayout(l_pGLContainerLayout);
//        m_uiViewer->glScene->addWidget(m_pGLContainer);

    // init worker
        m_pWInterface = new InterfaceWorker();

    // init connections
        QObject::connect(m_uiInterface->pbAddCorpus, SIGNAL(clicked()), this, SLOT(addCorpus()));
        QObject::connect(m_uiInterface->pbRemoveCorpus, SIGNAL(clicked()), this, SLOT(removeCorpus()));

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

        QObject::connect(m_uiInterface->cbNeurons,              SIGNAL(stateChanged(int)), SLOT(updateReservoirParameters(int)));
        QObject::connect(m_uiInterface->cbLeakRate,             SIGNAL(stateChanged(int)), SLOT(updateReservoirParameters(int)));
        QObject::connect(m_uiInterface->cbIS,                   SIGNAL(stateChanged(int)), SLOT(updateReservoirParameters(int)));
        QObject::connect(m_uiInterface->cbSpectralRadius,       SIGNAL(stateChanged(int)), SLOT(updateReservoirParameters(int)));
        QObject::connect(m_uiInterface->cbRidge,                SIGNAL(stateChanged(int)), SLOT(updateReservoirParameters(int)));
        QObject::connect(m_uiInterface->cbSparcity,             SIGNAL(stateChanged(int)), SLOT(updateReservoirParameters(int)));

        QObject::connect(m_uiInterface->leNeuronsOperation,         SIGNAL(editingFinished()), SLOT(updateReservoirParameters()));
        QObject::connect(m_uiInterface->leLeakRateOperation,        SIGNAL(editingFinished()), SLOT(updateReservoirParameters()));
        QObject::connect(m_uiInterface->leISOperation,              SIGNAL(editingFinished()), SLOT(updateReservoirParameters()));
        QObject::connect(m_uiInterface->leSpectralRadiusOperation,  SIGNAL(editingFinished()), SLOT(updateReservoirParameters()));
        QObject::connect(m_uiInterface->leRidgeOperation,           SIGNAL(editingFinished()), SLOT(updateReservoirParameters()));
        QObject::connect(m_uiInterface->leSparcityOperation,        SIGNAL(editingFinished()), SLOT(updateReservoirParameters()));

//        l_params.m_neuronsEnabled           = m_uiInterface->cbNeurons->isChecked();
//        l_params.m_leakRateEnabled          = m_uiInterface->cbLeakRate->isChecked();
//        l_params.m_issEnabled               = m_uiInterface->cbIS->isChecked();
//        l_params.m_spectralRadiusEnabled    = m_uiInterface->cbSpectralRadius->isChecked();
//        l_params.m_ridgeEnabled             = m_uiInterface->cbRidge->isChecked();
//        l_params.m_sparcityEnabled          = m_uiInterface->cbSparcity->isChecked();

//        l_params.m_neuronsOperation         = m_uiInterface->leNeuronsOperation->text();
//        l_params.m_leakRateOperation        = m_uiInterface->leLeakRateOperation->text();
//        l_params.m_issOperation             = m_uiInterface->leISOperation->text();
//        l_params.m_spectralRadiusOperation  = m_uiInterface->leSpectralRadiusOperation->text();
//        l_params.m_ridgeOperation           = m_uiInterface->leRidgeOperation->text();
//        l_params.m_sparcityOperation        = m_uiInterface->leSparcityOperation->text();


//        QObject::connect(m_uiViewer->pbLoadCloud, SIGNAL(clicked()), this, SLOT(loadCloud()));
//        QObject::connect(m_uiViewer->pbLoadMesh, SIGNAL(clicked()), this, SLOT(loadMesh()));
//        QObject::connect(m_uiViewer->pbDeleteCloud, SIGNAL(clicked()), this, SLOT(deleteCloud()));
//        QObject::connect(m_uiViewer->pbDeleteMesh, SIGNAL(clicked()), this, SLOT(deleteMesh()));
//        QObject::connect(m_uiViewer->pbSetTexture, SIGNAL(clicked()), this, SLOT(setTexture()));

//        QObject::connect(m_uiViewer->lwClouds, SIGNAL(currentRowChanged(int)), this, SLOT(updateCloudInterfaceParameters()));
//        QObject::connect(m_uiViewer->lwMeshes, SIGNAL(currentRowChanged(int)), this, SLOT(updateMeshInterfaceParameters()));
//        QObject::connect(m_uiViewer->lwClouds, SIGNAL(itemClicked(QListWidgetItem*)), this, SLOT(updateCloudInterfaceParameters(QListWidgetItem*)));
//        QObject::connect(m_uiViewer->lwMeshes, SIGNAL(itemClicked(QListWidgetItem*)), this, SLOT(updateMeshInterfaceParameters(QListWidgetItem*)));

//        QObject::connect(m_uiViewer->lwClouds, SIGNAL(currentRowChanged(int)), m_pWViewer, SLOT(updateCloudAnimationPath(int)));
//        QObject::connect(m_uiViewer->lwMeshes, SIGNAL(currentRowChanged(int)), m_pWViewer, SLOT(updateMeshAnimationPath(int)));
//        QObject::connect(this, SIGNAL(cloudCurrentRowChanged(int)), m_pWViewer, SLOT(updateCloudAnimationPath(int)));
//        QObject::connect(this, SIGNAL(meshCurrentRowChanged(int)), m_pWViewer, SLOT(updateMeshAnimationPath(int)));
//        QObject::connect(m_uiViewer->lwClouds, SIGNAL(itemClicked(QListWidgetItem*)), this, SLOT(updateCloudAnimationPath(QListWidgetItem*)));
//        QObject::connect(m_uiViewer->lwMeshes, SIGNAL(itemClicked(QListWidgetItem*)), this, SLOT(updateMeshAnimationPath(QListWidgetItem*)));


//        // update interface
//            QObject::connect(m_uiViewer->dsbRX, SIGNAL(valueChanged(double)), this, SLOT(updateParameters(double)));
//            QObject::connect(m_uiViewer->dsbRY, SIGNAL(valueChanged(double)), this, SLOT(updateParameters(double)));
//            QObject::connect(m_uiViewer->dsbRZ, SIGNAL(valueChanged(double)), this, SLOT(updateParameters(double)));
//            QObject::connect(m_uiViewer->dsbTrX, SIGNAL(valueChanged(double)), this, SLOT(updateParameters(double)));
//            QObject::connect(m_uiViewer->dsbTrY, SIGNAL(valueChanged(double)), this, SLOT(updateParameters(double)));
//            QObject::connect(m_uiViewer->dsbTrZ, SIGNAL(valueChanged(double)), this, SLOT(updateParameters(double)));
//            QObject::connect(m_uiViewer->dsbScaling, SIGNAL(valueChanged(double)), this, SLOT(updateParameters(double)));

//            QObject::connect(m_uiViewer->cbDisplayLines, SIGNAL(clicked()), this, SLOT(updateParameters()));
//            QObject::connect(m_uiViewer->cbVisible, SIGNAL(clicked()), this, SLOT(updateParameters()));
//            QObject::connect(m_uiViewer->rbDisplayOriginalColor, SIGNAL(clicked()), this, SLOT(updateParameters()));
//            QObject::connect(m_uiViewer->rbDisplayTexture, SIGNAL(clicked()), this, SLOT(updateParameters()));
//            QObject::connect(m_uiViewer->rbDisplayUnicolor, SIGNAL(clicked()), this, SLOT(updateParameters()));

//            QObject::connect(m_uiViewer->sbColorB, SIGNAL(valueChanged(int)), this, SLOT(updateParameters(int)));
//            QObject::connect(m_uiViewer->sbColorG, SIGNAL(valueChanged(int)), this, SLOT(updateParameters(int)));
//            QObject::connect(m_uiViewer->sbColorR, SIGNAL(valueChanged(int)), this, SLOT(updateParameters(int)));

//            QObject::connect(m_uiViewer->dsbLightX, SIGNAL(valueChanged(double)), this, SLOT(updateParameters(double)));
//            QObject::connect(m_uiViewer->dsbLightY, SIGNAL(valueChanged(double)), this, SLOT(updateParameters(double)));
//            QObject::connect(m_uiViewer->dsbLightZ, SIGNAL(valueChanged(double)), this, SLOT(updateParameters(double)));

//            QObject::connect(m_uiViewer->leTexturePath, SIGNAL(textChanged(QString)), this, SLOT(updateParameters(QString)));

//            QObject::connect(m_uiViewer->dsbAmbiantLight1, SIGNAL(valueChanged(double)), this, SLOT(updateParameters(double)));
//            QObject::connect(m_uiViewer->dsbAmbiantLight2, SIGNAL(valueChanged(double)), this, SLOT(updateParameters(double)));
//            QObject::connect(m_uiViewer->dsbAmbiantLight3, SIGNAL(valueChanged(double)), this, SLOT(updateParameters(double)));
//            QObject::connect(m_uiViewer->dsbDiffusLight1, SIGNAL(valueChanged(double)), this, SLOT(updateParameters(double)));
//            QObject::connect(m_uiViewer->dsbDiffusLight2, SIGNAL(valueChanged(double)), this, SLOT(updateParameters(double)));
//            QObject::connect(m_uiViewer->dsbDiffusLight3, SIGNAL(valueChanged(double)), this, SLOT(updateParameters(double)));
//            QObject::connect(m_uiViewer->dsbSpecularLight1, SIGNAL(valueChanged(double)), this, SLOT(updateParameters(double)));
//            QObject::connect(m_uiViewer->dsbSpecularLight2, SIGNAL(valueChanged(double)), this, SLOT(updateParameters(double)));
//            QObject::connect(m_uiViewer->dsbSpecularLight3, SIGNAL(valueChanged(double)), this, SLOT(updateParameters(double)));
//            QObject::connect(m_uiViewer->dsbAmbiantK, SIGNAL(valueChanged(double)), this, SLOT(updateParameters(double)));
//            QObject::connect(m_uiViewer->dsbDiffusK, SIGNAL(valueChanged(double)), this, SLOT(updateParameters(double)));
//            QObject::connect(m_uiViewer->dsbSpecularK, SIGNAL(valueChanged(double)), this, SLOT(updateParameters(double)));
//            QObject::connect(m_uiViewer->dsbSpecularP, SIGNAL(valueChanged(double)), this, SLOT(updateParameters(double)));
//        // push buttons
//            QObject::connect(m_uiViewer->pbSetCamera, SIGNAL(clicked()), this, SLOT(setCameraToCurrentItem()));
//            QObject::connect(m_uiViewer->pbResetCamera, SIGNAL(clicked()), m_pGLMultiObject, SLOT(resetCamera()));
//            QObject::connect(m_uiViewer->pbLaunchAllAnim, SIGNAL(clicked()), m_pWViewer, SLOT(startLoop()));
//            QObject::connect(m_uiViewer->pbSetModFile, SIGNAL(clicked()), this, SLOT(loadModFile()));
//            QObject::connect(m_uiViewer->pbSetSeqFile, SIGNAL(clicked()), this, SLOT(loadSeqFile()));
//            QObject::connect(m_uiViewer->pbSetMeshCorr, SIGNAL(clicked()), this, SLOT(loadMeshCorrFile()));

//        // fullscreen
//            QObject::connect(m_pGLMultiObject, SIGNAL(enableFullScreen()), this, SLOT(enableGLFullScreen()));
//            QObject::connect(m_pGLMultiObject, SIGNAL(disableFullScreen()), this, SLOT(disableGLFullScreen()));

//        // worker
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

//    // init thread
        m_pWInterface->moveToThread(&m_TInterface);
        m_TInterface.start();
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

void Interface::updateReservoirParameters(bool value)
{
    updateReservoirParameters();
}

void Interface::updateReservoirParameters()
{
    ReservoirParameter l_params;

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




InterfaceWorker::InterfaceWorker()
{
    qRegisterMetaType<ReservoirParameter>("ReservoirParameters");
}

