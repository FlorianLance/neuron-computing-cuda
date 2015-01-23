
/**
 * \file Generalization.h
 * \brief defines Generalization
 * \author Florian Lance
 * \date 01/10/14
 */

#ifndef GENERALIZATION_H
#define GENERALIZATION_H

#include <Model.h>

/**
 * @brief The Generalization class
 * Will do a generatilization test of a corpus.
 */
class Generalization
{
    public :

        /**
         * @brief The RandomPart enum
         */
        enum RandomPart{MEANING,INFOS,SENTENCES};


        /**
         * @brief Generalization constructor.
         * @param [in] model : model to be used for the generation.
         */
        Generalization(Model &model);


        /**
         * @brief Build an id random list sentences.
         */
        void retrieveRandomSentenceList(cint sizeCorpus, cint nbSentence, std::vector<int> &randomSentenceList);

        /**
         * @brief retrieveSubMeaningCorpusRandomized
         * @param randomSentenceList
         * @param subMeaning
         * @param subInfo
         * @param subSentence
         */
        void retrieveSubSentenceCorpusRandomized(std::vector<int> &randomSentenceList, QVector<QStringList> &subMeaning, QVector<QStringList> &subInfo, QVector<QStringList> &subSentence);


        /**
         * @brief retrieveSubInfoRandomized
         * @param randomSentenceList
         * @param subMeaning
         * @param subInfo
         * @param subSentence
         */
        void retrieveSubInfoRandomized(std::vector<int> &randomSentenceList, QVector<QStringList> &subMeaning, QVector<QStringList> &subInfo, QVector<QStringList> &subSentence);


        /**
         * @brief retrieveSubMeaningCorpusRandomized
         * @param randomSentenceList
         * @param subMeaning
         * @param subInfo
         * @param subSentence
         */
        void retrieveSubMeaningCorpusRandomized(std::vector<int> &randomSentenceList, QVector<QStringList> &subMeaning, QVector<QStringList> &subInfo, QVector<QStringList> &subSentence);

        /**
         * @brief randomChangeCorpusGeneralization
         * @param numberRandomSentences
         * @param pathRandomCorpus
         * @param randomPart
         */
        void randomChangeCorpusGeneralization(cint numberRandomSentences, const QString pathRandomCorpus, const RandomPart randomPart = SENTENCES);

        /**
         * @brief Start the cross verification and save the results in the input paths.
         * @param [in] xCheckTrainPath         : parameters and results of all the training with S - 1 number of sentences (S -> size of the corpus)
         * @param [in] xCheckTestPath          : parameters and results of all the testing with 1 sentence (the one remove from the training)
         * @param [in] xCheckTestSentencesPath : final sentences retrieved from the testing with the goal sentences
         */
        void startXVerification(const std::string &xCheckTrainPath         = "../data/Results/xCheckTrain.txt",
                                const std::string &xCheckTestPath          = "../data/Results/xCheckTest.txt",
                                const std::string &xCheckTestSentencesPath = "../data/Results/xCheckTestSentences.txt");



    private :

        Model *m_model;  /**< pointer to the model */

        QVector<QStringList> m_trainMeaning;    /**< train meaning data */
        QVector<QStringList> m_trainInfo;       /**< train info data */
        QVector<QStringList> m_trainSentence;   /**< train sentence data */
};

#endif
