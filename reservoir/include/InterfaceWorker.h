
/**
 * \file InterfaceWorker.h
 * \brief Defines InterfaceWorker
 * \author Florian Lance
 * \date 01/12/14
 */

#ifndef _INTERFACEWORKER_
#define _INTERFACEWORKER_

// reservoir
#include "GridSearch.h"

/**
 * @brief The ReplayParameters struct
 */
struct ReplayParameters
{
    bool m_useLastTraining; /**< ... */

    bool m_randomNeurons;   /**< ... */
    bool m_randomSentence;  /**< ... */

    int m_rangeNeuronsStart;    /**< ... */
    int m_rangeNeuronsEnd;      /**< ... */
    int m_randomNeuronsNumber;  /**< ... */

    int m_rangeSentencesStart;      /**< ... */
    int m_rangeSentencesEnd;        /**< ... */
    int m_randomSentencesNumber;    /**< ... */
};

/**
 * @brief The LanguageParameters struct
 */
struct LanguageParameters
{
    QString m_structure;    /**< ... */
    QString m_CCW;          /**< ... */
};

/**
 * @brief The ReservoirParameters struct
 */
struct ReservoirParameters
{
    bool m_useCuda; /**< ... */

    bool m_useLoadedTraining;   /**< ... */
    bool m_useLoadedW;          /**< ... */
    bool m_useLoadedWIn;        /**< ... */
    bool m_useOnlyStartValue;   /**< ... */

    ActionToDo m_action;        /**< ... */

    int m_neuronsStart;             /**< ... */
    double m_leakRateStart;         /**< ... */
    double m_issStart;              /**< ... */
    double m_spectralRadiusStart;   /**< ... */
    double m_ridgeStart;            /**< ... */
    double m_sparcityStart;         /**< ... */

    int m_neuronsEnd;           /**< ... */
    double m_leakRateEnd;       /**< ... */
    double m_issEnd;            /**< ... */
    double m_spectralRadiusEnd; /**< ... */
    double m_ridgeEnd;          /**< ... */
    double m_sparcityEnd;       /**< ... */

    int m_neuronsNbOfUses;          /**< ... */
    int m_leakRateNbOfUses;         /**< ... */
    int m_issNbOfUses;              /**< ... */
    int m_spectralRadiusNbOfUses;   /**< ... */
    int m_ridgeNbOfUses;            /**< ... */
    int m_sparcityNbOfUses;         /**< ... */

    bool m_neuronsEnabled;          /**< ... */
    bool m_leakRateEnabled;         /**< ... */
    bool m_issEnabled;              /**< ... */
    bool m_spectralRadiusEnabled;   /**< ... */
    bool m_ridgeEnabled;            /**< ... */
    bool m_sparcityEnabled;         /**< ... */

    QString m_neuronsOperation;         /**< ... */
    QString m_leakRateOperation;        /**< ... */
    QString m_issOperation;             /**< ... */
    QString m_spectralRadiusOperation;  /**< ... */
    QString m_ridgeOperation;           /**< ... */
    QString m_sparcityOperation;        /**< ... */
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
        GridSearch *gridSearch() const;

        /**
         * @brief model
         * @return
         */
        Model *model();

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
         * @brief updateReplayParameters
         * @param newParams
         */
        void updateReplayParameters(ReplayParameters newParams);

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

        /**
         * @brief startReplay
         */
        void startReplay();

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

        /**
         * @brief replayLoaded
         */
        void replayLoaded();

        /**
         * @brief sendReplayData
         */
        void sendReplayData(QVector<QVector<double> >, QVector<int>, QVector<int>);

    private :

        cv::Mat m_xTot; /**< ... */

        int m_nbOfCorpus;   /**< ... */

        QString m_absolutePath; /**< ... */

        QStringList m_parametersTrainingLoaded; /**< ... */
        QStringList m_parametersWLoaded;        /**< ... */
        QStringList m_parametersWInLoaded;      /**< ... */

        ReservoirParameters m_reservoirParameters;  /**< ... */
        LanguageParameters m_languageParameters;    /**< ... */
        ReplayParameters m_replayParameters;        /**< ... */

        Model m_model;                /**< ... */
        GridSearch *m_gridSearch;     /**< ... */

        QStringList m_corpusList;   /**< ... */
};



#endif
