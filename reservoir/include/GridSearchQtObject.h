
/**
 * \file GridSearchQtObject.h
 * \brief defines GridSearchQtObject
 * \author Florian Lance
 * \date 04/12/14
 */

#ifndef GRIDSEARCHQT_H
#define GRIDSEARCHQT_H

//#include <Model.h>
#include <ModelQtObject.h>

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



struct ResultsDisplayReservoir
{
    std::vector<double> m_absoluteCCW;
    std::vector<double> m_absoluteAll;

    Sentences m_trainSentences;
    Sentences m_trainResults;
    Sentences m_testResults;
};


/**
 * @brief The GridSearchQt class
 */
class GridSearchQt : public QObject
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
        GridSearchQt(ModelQt &model);

        /**
         * @brief Set the cuda parameters.
         * @param [in] useCudaInversion      : use the cuda inversion instead of opencv
         * @param [in] useCudaMultiplication : use the cuda multiplication instead of opencv
         */
        void setCudaParameters(cbool useCudaInversion, cbool useCudaMultiplication);


        /**
         * @brief Start the training with all the parameters defined.
         * @param [in] resultsFilePath      : result file path (readable data)
         * @param [in] resultsRawFilePath   : result file path (raw data, easy to read with gnuplot)
         * @param doTraining
         * @param doTest
         * @param loadTraining
         */
        void launchTrainWithAllParameters(const std::string resultsFilePath, const std::string resultsRawFilePath, cbool doTraining = true, cbool doTest = false, cbool loadTraining = false);

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
        void sendCurrentParametersSignal(ModelParametersQt);

        /**
         * @brief sendResultsReservoirSignal
         */
        void sendResultsReservoirSignal(ResultsDisplayReservoir);

    private :

        /**
         * @brief addResultsInStream
         * @param streamReadableData
         * @param streamRawData
         * @param results
         * @param numCorpus
         * @param time
         * @param parameters
         * @param nbCharParams
         */
        void addResultsInStream(std::ofstream *streamReadableData, std::ofstream *streamRawData, const std::vector<double> &results, cint numCorpus, const double time, const ModelParametersQt parameters, int *nbCharParams);


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

        bool m_useCudaInv;                          /**< uses cuda inversion ? */
        bool m_useCudaMult;                         /**< uses cuda multiplication ? */

        std::vector<int> m_nbNeuronsValues;         /**< neurons nb grid search values */
        std::vector<double> m_leakRateValues;       /**< leakrate grid search values */
        std::vector<double> m_sparcityValues;       /**< sparcity grid search values */
        std::vector<double> m_inputScalingValues;   /**< input scaling grid search values */
        std::vector<double> m_ridgeValues;          /**< ridge grid search values */
        std::vector<double> m_spectralRadiusValues; /**< spectral radius grid search values */
        std::vector<std::string> m_corpusList;      /**< corpus list to be used in the grid search */

        ModelQt *m_model;                             /**< pointer to the model */
};


#endif
