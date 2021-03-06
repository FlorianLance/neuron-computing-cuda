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
 * \file GridSearch.h
 * \brief defines GridSearch
 * \author Florian Lance
 * \date 04/12/14
 */

#ifndef GRIDSEARCH_H
#define GRIDSEARCH_H

#include <Model.h>


template <typename T>
/**
 * @brief Display a templated vector with std::cout.
 * @param [in] vec : template vector to be displayed
 */
static void display(std::vector<T> vec)
{
    for(int ii = 0; ii < vec.size(); ++ii)
    {
        std::cout << vec[ii] << " ";
    }
    std::cout << std::endl;
}

/**
 * @brief actions that can be done : TRAINING_RES -> only training / TEST_RES -> only doing the tests / BOTH_RES -> the both
 */
enum ActionToDo
{
    TRAINING_RES,TEST_RES,BOTH_RES
};

/**
 * @brief Results from the reservoir
 */
struct ResultsDisplayReservoir
{
    std::vector<double> m_absoluteCCW;      /**< absolute pairwise CCW results */
    std::vector<double> m_absoluteAll;      /**< absolute pairwise all results  */
    std::vector<double> m_continuousCCW;    /**< continuous pairwise CCW results */
    std::vector<double> m_continuousAll;    /**< continuous pairwise all results */

    Sentences m_trainSentences;             /**< train sentences */
    Sentences m_trainResults;               /**< train sentences results */
    Sentences m_testResults;                /**< test sentences results */

    ActionToDo m_action;                    /**< action to do */
};


/**
 * @brief Class for doing planifications with custom parameters of resrvoir training and testing
 */
class GridSearch : public QObject
{
    Q_OBJECT

    public :

        /**
         * @brief Enum of the parameters used by the grid search.
         */
        enum ReservoirParameter
        {
            NEURONS_NB,LEAK_RATE,SPARCITY,INPUT_SCALING,RIDGE,SPECTRAL_RADIUS
        };

        /**
         * @brief GridSearchQt constructor.
         * @param [in] model : model to be used for the generation.
         */
        GridSearch(Model &model);

        /**
         * @brief Set the cuda parameters.
         * @param [in] useCudaInversion      : use the cuda inversion instead of opencv
         * @param [in] useCudaMultiplication : use the cuda multiplication instead of opencv
         */
        void setCudaParameters(cbool useCudaInversion, cbool useCudaMultiplication);

        /**
         * @brief setNumberGeneratorParameters
         * @param randomSeed
         * @param seed
         */
        void setNumberGeneratorParameters(cbool randomSeed, cint seed);

        /**
         * @brief Start the training with all the parameters defined.
         * @param [in] resultsFilePath      : result file path (readable data)
         * @param [in] resultsRawFilePath   : result file path (raw data, easy to read with gnuplot)
         * @param [in] doTraining           :
         * @param [in] doTest               :
         * @param [in] loadTraining         :
         * @param [in] loadW                :
         * @param [in] loadWIn              :
         */
        void launchTrainWithAllParameters(const std::string resultsFilePath, const std::string resultsRawFilePath, cbool doTraining = true, cbool doTest = false, cbool loadTraining = false
                , cbool loadW = false, cbool loadWIn = false);

        /**
         * @brief Define the range of values for a parameter to be used.
         * @param [in] parameterId  : id of the parameter corresponding to the ReservoirParameter enum
         * @param [in] startValue   : the starting value of the parameter
         * @param [in] endValue     : the ending value of the parameter
         * @param [in] operation    : operator to apply to the value for defining the range : ex "+2" "*5.1" -"2.3"
         * @param [in] useOnlyStartValue      : uses only the start value
         * @param [in] nbOfTimesForEachValues : the number of times a value will be used
         */
        bool setParameterValues(const ReservoirParameter parameterId, cdouble startValue, cdouble endValue, const std::string operation = "*2", cbool useOnlyStartValue = false, cint nbOfTimesForEachValues = 1);

        /**
         * @brief deleteParameterValues
         */
        void deleteParameterValues();

        /**
         * @brief Define the corpus file to be used.
         * @param [in] corpusList : path of the corpus file
         */
        void setCorpusList(const std::vector<std::string> &corpusList);

    signals :

        /**
         * @brief sendCurrentParametersSignal
         */
        void sendCurrentParametersSignal(ModelParameters);

        /**
         * @brief sendResultsReservoirSignal
         */
        void sendResultsReservoirSignal(ResultsDisplayReservoir);

        /**
         * @brief sendLogInfo
         */
        void sendLogInfo(QString, QColor);

    private :

        /**
         * @brief addResultsInStream
         */
        void addResultsInStream(std::ofstream *streamReadableData, std::ofstream *streamRawData, const std::vector<double> &results, cint numCorpus, const double time, const ModelParameters parameters, int *nbCharParams);


        template<typename T>
        /**
         * @brief Apply the operation corresponding to the string operation to the value
         * @param [in] value : value
         * @param operation  : operation to be applied on the value
         * @return the result
         */
        T applyOperation(const T value, const std::string operation)
        {
            std::string l_number = operation;
            l_number.erase(0,1);

            T l_operationNb;
            std::istringstream(l_number) >> l_operationNb;

            if(operation[0] == '*')
            {
                return value * l_operationNb;
            }
            else if(operation[0] == '+')
            {
                return value + l_operationNb;
            }
            else if(operation[0] == '/')
            {
                return value / l_operationNb;
            }
            else if(operation[0] == '-')
            {
                return value - l_operationNb;
            }
            else
            {
                return value;
            }
        }

        bool m_randomSeed;
        bool m_useCudaInv;                          /**< uses cuda inversion ? */
        bool m_useCudaMult;                         /**< uses cuda multiplication ? */

        int m_seed;

        std::vector<int> m_nbNeuronsValues;         /**< neurons nb grid search values */
        std::vector<double> m_leakRateValues;       /**< leakrate grid search values */
        std::vector<double> m_sparcityValues;       /**< sparcity grid search values */
        std::vector<double> m_inputScalingValues;   /**< input scaling grid search values */
        std::vector<double> m_ridgeValues;          /**< ridge grid search values */
        std::vector<double> m_spectralRadiusValues; /**< spectral radius grid search values */
        std::vector<std::string> m_corpusList;      /**< corpus list to be used in the grid search */

        Model *m_model;                             /**< pointer to the model */
};


#endif

