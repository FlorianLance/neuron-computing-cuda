
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
 * \file GridSearch.cpp
 * \brief defines GridSearch
 * \author Florian Lance
 * \date 04/12/14
 */

#include "GridSearch.h"


#include "../moc/moc_GridSearch.cpp"

GridSearch::GridSearch(Model &model) : m_model(&model), m_useCudaInv(true), m_useCudaMult(false)
{}

void GridSearch::setCudaParameters(cbool useCudaInversion, cbool useCudaMultiplication)
{
    m_useCudaInv    = useCudaInversion;
    m_useCudaMult   = useCudaMultiplication;
}

void GridSearch::setNumberGeneratorParameters(cbool randomSeed, cint seed)
{
    m_seed = seed;
    m_randomSeed = randomSeed;
}

void GridSearch::launchTrainWithAllParameters(const std::string resultsFilePath, const std::string resultsRawFilePath, cbool doTraining, cbool doTest, cbool loadTraining, cbool loadW, cbool loadWIn)
{
    if(m_corpusList.size() == 0)
    {
        std::cerr << "-ERROR : at least one corpus must be defined. Grid Search aborted. " << std::endl;
        emit sendLogInfo("-ERROR : at least one corpus must be defined. Grid Search aborted.  \n", QColor(Qt::red));
        return;
    }
    if(m_nbNeuronsValues.size() == 0)
    {
        std::cout << "No neuron nb value found, 1 default value set. " << std::endl;
        emit sendLogInfo("No neuron nb value found, 1 default value set.  \n", QColor(Qt::blue));
        m_nbNeuronsValues.push_back(800);
    }
    if(m_leakRateValues.size() == 0)
    {
        std::cout << "No leak rate value found, 1 default value set. " << std::endl;
        emit sendLogInfo("No leak rate value found, 1 default value set.  \n", QColor(Qt::blue));
        m_leakRateValues.push_back(0.1);
    }
    if(m_sparcityValues.size() == 0)
    {
        std::cout << "No sparcity value found, automatic value set. " << std::endl;
        emit sendLogInfo("No sparcity value found, automatic value set.  \n", QColor(Qt::blue));
        m_sparcityValues.push_back(-1);
    }
    if(m_inputScalingValues.size() == 0)
    {
        std::cout << "No input scaling value found, 1 default value set. " << std::endl;
        emit sendLogInfo("No input scaling value found, 1 default value set.  \n", QColor(Qt::blue));
        m_inputScalingValues.push_back(0.1);
    }
    if(m_spectralRadiusValues.size() == 0)
    {
        std::cout << "No spectral radius value found, 1 default value set. " << std::endl;
        emit sendLogInfo("No spectral radius value found, 1 default value set.  \n", QColor(Qt::blue));
        m_spectralRadiusValues.push_back(3);
    }
    if(m_ridgeValues.size() == 0)
    {
        std::cout << "No ridge value found, 1 default value set. " << std::endl;
        emit sendLogInfo("No ridge value found, 1 default value set.  \n", QColor(Qt::blue));
        m_ridgeValues.push_back(1e-5);
    }

    int l_nbTrain = static_cast<int>(m_corpusList.size()*m_nbNeuronsValues.size()*m_leakRateValues.size()*m_sparcityValues.size()*
            m_inputScalingValues.size()*m_spectralRadiusValues.size()*m_ridgeValues.size());

    std::cout << "#################################" << std::endl;
    std::cout << "Start Grid search for training : " << std::endl;
    std::cout << "Number of trains/tests to be done : " << l_nbTrain << std::endl;
    std::cout << "#################################\n" << std::endl;

    emit sendLogInfo("Start Grid search for training :  \nNumber of trains/tests to be done : " + QString::number(l_nbTrain) +"\n\n", QColor(Qt::blue));

    std::ofstream l_flowResFileReadableData(resultsFilePath), l_flowResFileRawData(resultsRawFilePath);

    if(!l_flowResFileReadableData)
    {
        std::cerr << "-ERROR : can not write the results in the file " << resultsFilePath << std::endl;
        emit sendLogInfo("-ERROR : can not write the results in the file. \n", QColor(Qt::red));
        return;
    }
    if(!l_flowResFileRawData)
    {
        std::cerr << "-ERROR : can not write the results in the file " << resultsRawFilePath << std::endl;
        emit sendLogInfo("-ERROR : can not write the results in the file.  \n", QColor(Qt::red));
        return;
    }

    l_flowResFileReadableData << "### Grid search results ###\n";
    l_flowResFileReadableData << "Corpus :\n";

    // write corpus names
    for(int ii = 0; ii < m_corpusList.size(); ++ii)
    {
        l_flowResFileReadableData << m_corpusList[ii] << " " << ii << "\n";
    }

    l_flowResFileReadableData << "\nRES 1 : CCW pairwise absolute (0% or 100%) -> ex : goal : the , the  that -s -ed it | res : that, the the -s -ed it\n";
    l_flowResFileReadableData << "RES 2 : CCW pairwise continuous (between 0% and 100%) \n";
    l_flowResFileReadableData << "RES 3 : ALL pairwise absolute (0% or 100%) -> ex : goal : the X , the X that X -s X -ed it | res : the X X X, the the that X -s X X -ed it \n";
    l_flowResFileReadableData << "RES 4 : ALL pairwise continuous (between 0% and 100%) \n";
    l_flowResFileReadableData << "\n CORPUS ID | NEURONS | LEAK RATE | SPARCITY | INPUT SCALING |  RIDGE  | SPECTRAL RADIUS |   TIME   |   RES 1   |   RES 2   |   RES 3   |   RES 4   |\n";

    int l_nbCharParams[] = {11,9,11,10,15,9,17,10,11,11,11,11};

    int l_currentTrain = 1;
    int l_currentTest = 1;

    for(int aa = 0; aa < m_corpusList.size(); ++aa)
    {
        for(int ii = 0; ii < m_nbNeuronsValues.size(); ++ii)
        {
            for(int jj = 0; jj < m_leakRateValues.size(); ++jj)
            {
                for(int kk = 0; kk < m_sparcityValues.size(); ++kk)
                {
                    for(int ll = 0; ll < m_inputScalingValues.size(); ++ll)
                    {
                        for(int mm = 0; mm < m_ridgeValues.size(); ++mm)
                        {
                            for(int nn = 0; nn < m_spectralRadiusValues.size(); ++nn)
                            {
                                double l_sparcity;

                                if(m_sparcityValues[kk] == -1)
                                {
                                    l_sparcity = 10.0 / m_nbNeuronsValues[ii];
                                }
                                else
                                {
                                    l_sparcity = m_sparcityValues[kk];
                                }

                                ModelParameters l_currentParameters;
                                l_currentParameters.m_corpusFilePath    = m_corpusList[aa];
                                l_currentParameters.m_nbNeurons         = m_nbNeuronsValues[ii];
                                l_currentParameters.m_leakRate          = m_leakRateValues[jj];
                                l_currentParameters.m_sparcity          = l_sparcity;
                                l_currentParameters.m_inputScaling      = m_inputScalingValues[ll];
                                l_currentParameters.m_ridge             = m_ridgeValues[mm];
                                l_currentParameters.m_spectralRadius    = m_spectralRadiusValues[nn];
                                l_currentParameters.m_useCudaInv        = m_useCudaInv;
                                l_currentParameters.m_useCudaMult       = m_useCudaMult;

                                l_currentParameters.m_useLoadedTraining = loadTraining;
                                l_currentParameters.m_useLoadedW        = loadW;
                                l_currentParameters.m_useLoadedWIn      = loadWIn;

                                l_currentParameters.m_randomSeedNumberGenerator = m_randomSeed;
                                l_currentParameters.m_seedNumberGenerator = m_seed;

                                emit sendCurrentParametersSignal(l_currentParameters);

                                std::cout << "############################################################## " << std::endl;

                                m_model->resetModelParameters(l_currentParameters, false);

                                clock_t l_timeTraining = clock();
                                std::vector<double> l_diffSizeOCW, l_absoluteCCW, l_continuousCCW, l_absoluteAll, l_continuousAll;
                                double l_meanDiffSizeOCW, l_meanContinuousCCW, l_meanAbsoluteCCW, l_meanContinuousAll, l_meanAbsoluteAll;

                                // launch the training part
                                if(doTraining && !loadTraining)
                                {
                                    emit sendLogInfo("# Start the training number : " +  QString::number(l_currentTrain) + " / " + QString::number(l_nbTrain) + " \n", QColor(Qt::blue));
                                    std::cout << "########## Start the training number : " << l_currentTrain++ << " / " << l_nbTrain << std::endl << std::endl;

                                    if(!m_model->launchTraining())
                                    {
                                        emit sendLogInfo("Abort gridsearch. \n", QColor(Qt::red));
                                        return;
                                    }

                                    double l_time = static_cast<double>((clock() - l_timeTraining)) / CLOCKS_PER_SEC;

                                    m_model->displayResults(true,false);

                                    m_model->computeResultsData(true, l_diffSizeOCW,
                                                                l_absoluteCCW, l_continuousCCW,
                                                                l_absoluteAll, l_continuousAll,
                                                                l_meanDiffSizeOCW,
                                                                l_meanAbsoluteCCW, l_meanContinuousCCW,
                                                                l_meanAbsoluteAll, l_meanContinuousAll
                                                                );

                                    std::vector<double> l_results;
                                    l_results.push_back(l_meanAbsoluteCCW);
                                    l_results.push_back(l_meanContinuousCCW);
                                    l_results.push_back(l_meanAbsoluteAll);
                                    l_results.push_back(l_meanContinuousAll);

                                    addResultsInStream(&l_flowResFileReadableData, &l_flowResFileRawData, l_results, aa, l_time, l_currentParameters, l_nbCharParams);
                                }

                                // launch the test part
                                if(doTest && (doTraining || loadTraining))
                                {
                                    emit sendLogInfo("# Start the test number : " +  QString::number(l_currentTest) + " / " + QString::number(l_nbTrain) + " \n", QColor(Qt::blue));
                                    std::cout << "########## Start the test number : " << l_currentTest++ << " / " << l_nbTrain << std::endl << std::endl;                                    

                                    if(m_model->launchTests())
                                    {                                        
                                        m_model->displayResults(false,true);
                                    }
                                }

                                // send results to be displayed in the ui
                                    ResultsDisplayReservoir l_resultsToDisplay;
                                    m_model->sentences(l_resultsToDisplay.m_trainSentences, l_resultsToDisplay.m_trainResults, l_resultsToDisplay.m_testResults);
                                    l_resultsToDisplay.m_absoluteCCW = l_absoluteCCW;
                                    l_resultsToDisplay.m_absoluteAll = l_absoluteAll;
                                    l_resultsToDisplay.m_continuousAll = l_continuousAll;
                                    l_resultsToDisplay.m_continuousCCW = l_continuousCCW;

                                    if(doTraining && doTest)
                                    {
                                        l_resultsToDisplay.m_action = BOTH_RES;
                                    }
                                    else if(doTraining)
                                    {
                                        l_resultsToDisplay.m_action = TRAINING_RES;
                                    }
                                    else
                                    {
                                        l_resultsToDisplay.m_action = TEST_RES;
                                    }


                                    emit sendResultsReservoirSignal(l_resultsToDisplay);

                                if(!doTest && !doTraining)
                                {
                                    std::cerr << "Training and test deactivated, nothing to done. " << std::endl;
                                    emit sendLogInfo("Training and test deactivated, nothing to done. \n", QColor(Qt::red));
                                }

                                if(doTest && !loadTraining && !doTraining)
                                {
                                    std::cerr << "Test can't be done, training is deactivated and not loaded. " << std::endl;
                                    emit sendLogInfo("Test can't be done, training is deactivated and not loaded. \n", QColor(Qt::red));
                                }


                                std::cout << "############################################################## " << std::endl << std::endl;
                            }
                        }
                    }
                }
            }
        }

        l_flowResFileRawData << "\n" << std::endl;
    }
}

void GridSearch::deleteParameterValues()
{
    m_nbNeuronsValues.clear();
    m_leakRateValues.clear();
    m_sparcityValues.clear();
    m_inputScalingValues.clear();
    m_ridgeValues.clear();
    m_spectralRadiusValues.clear();
}

bool GridSearch::setParameterValues(const GridSearch::ReservoirParameter parameterId, cdouble startValue, cdouble endValue, const std::string operation, cbool useOnlyStartValue, cint nbOfTimesForEachValues)
{
    bool l_operationValid = true;

    std::vector<int> l_valuesI;
    std::vector<double> l_valuesD;

    switch(parameterId)
    {
        case NEURONS_NB :
            m_nbNeuronsValues.clear();
        break;
        case LEAK_RATE :
            m_leakRateValues.clear();
        break;
        case SPARCITY :
            m_sparcityValues.clear();
        break;
        case INPUT_SCALING :
            m_inputScalingValues.clear();
        break;
        case RIDGE :
            m_ridgeValues.clear();
        break;
        case SPECTRAL_RADIUS :
            m_spectralRadiusValues.clear();
        break;
    }

    double l_value = static_cast<double>(startValue);

    int l_loopStop = 0;
    if(startValue <= endValue && !useOnlyStartValue)
    {
        while(l_value <= endValue)
        {
            if(parameterId != NEURONS_NB)
            {
                for(int ii = 0; ii < nbOfTimesForEachValues; ++ii)
                    l_valuesD.push_back(l_value);
            }
            else
            {
                for(int ii = 0; ii < nbOfTimesForEachValues; ++ii)
                    l_valuesI.push_back(static_cast<int>(l_value));
            }

            l_value = applyOperation(l_value, operation);

            if(++l_loopStop > 500)
            {
                std::cerr << "Bad operation for parameter : " << parameterId << std::endl;
                emit sendLogInfo("Bad operation for parameter : " +  QString::number(parameterId) + " \n", QColor(Qt::red));
                l_operationValid = false;
                break;
            }
        }
    }
    else if(!useOnlyStartValue)
    {
        while(l_value >= endValue)
        {
            if(parameterId != NEURONS_NB)
            {
                for(int ii = 0; ii < nbOfTimesForEachValues; ++ii)
                    l_valuesD.push_back(l_value);
            }
            else
            {
                for(int ii = 0; ii < nbOfTimesForEachValues; ++ii)
                    l_valuesI.push_back(static_cast<int>(l_value));
            }

            l_value = applyOperation(l_value, operation);

            if(++l_loopStop > 500)
            {
                std::cerr << "Bad operation for parameter : " << parameterId << std::endl;
                emit sendLogInfo("Bad operation for parameter : " +  QString::number(parameterId) + " \n", QColor(Qt::red));
                l_operationValid = false;
                break;
            }
        }
    }
    else
    {
        if(parameterId != NEURONS_NB)
        {
            for(int ii = 0; ii < nbOfTimesForEachValues; ++ii)
                l_valuesD.push_back(l_value);
        }
        else
        {
            for(int ii = 0; ii < nbOfTimesForEachValues; ++ii)
                l_valuesI.push_back(static_cast<int>(l_value));
        }
    }

    switch(parameterId)
    {
        case NEURONS_NB :
            m_nbNeuronsValues       = l_valuesI;
        break;
        case LEAK_RATE :
            m_leakRateValues        = l_valuesD;
        break;
        case SPARCITY :
            m_sparcityValues        = l_valuesD;
        break;
        case INPUT_SCALING :
            m_inputScalingValues    = l_valuesD;
        break;
        case RIDGE :
            m_ridgeValues           = l_valuesD;
        break;
        case SPECTRAL_RADIUS :
            m_spectralRadiusValues  = l_valuesD;
        break;
    }

    return l_operationValid;
}

void GridSearch::setCorpusList(const std::vector<std::string> &corpusList)
{
    m_corpusList = corpusList;
}


void GridSearch::addResultsInStream(std::ofstream *streamReadableData, std::ofstream *streamRawData, const std::vector<double> &results, cint numCorpus, const double time, const ModelParameters parameters, int *nbCharParams)
{
    // retrieve string values from parameters
        std::ostringstream l_os1,l_os2,l_os3,l_os4,l_os5,l_os6,l_os7,l_os8,l_os9,l_os10,l_os11, l_os12;
        l_os4.precision(4);l_os8.precision(6),l_os9.precision(3); l_os10.precision(3); l_os11.precision(3),l_os12.precision(3);
        l_os1 << numCorpus; l_os2 << parameters.m_nbNeurons; l_os3 <<  parameters.m_leakRate;
        l_os4 << parameters.m_sparcity; l_os5 << parameters.m_inputScaling; l_os6 << parameters.m_ridge;
        l_os7 << parameters.m_spectralRadius; l_os8 << time; l_os9 << results[0]; l_os10 << results[1]; l_os11 << results[2];
        l_os12 << results[3];

        std::vector<std::string> l_parameters;
        l_parameters.push_back(l_os1.str()); l_parameters.push_back(l_os2.str()); l_parameters.push_back(l_os3.str()); l_parameters.push_back(l_os4.str());
        l_parameters.push_back(l_os5.str()); l_parameters.push_back(l_os6.str()); l_parameters.push_back(l_os7.str()); l_parameters.push_back(l_os8.str());
        l_parameters.push_back(l_os9.str()); l_parameters.push_back(l_os10.str()); l_parameters.push_back(l_os11.str());
        l_parameters.push_back(l_os12.str());
    // read raw data
        for(int oo = 0; oo < l_parameters.size(); ++oo)
        {
            (*streamRawData) << l_parameters[oo] << " ";
        }
        (*streamRawData) << std::endl;

    // read readable data
        int l_nbSpaces,l_nbDivSpaces1,l_nbDivSpaces2;
        std::string l_spaces;

        for(int oo = 0; oo < l_parameters.size(); ++oo)
        {
            l_nbSpaces = nbCharParams[oo] - static_cast<int>(l_parameters[oo].size());
            l_nbDivSpaces1 = l_nbSpaces/2;
            l_nbDivSpaces2 = l_nbSpaces/2 + l_nbSpaces%2;
            l_spaces.append(l_nbDivSpaces1, ' ');
            (*streamReadableData) << l_spaces; l_spaces.clear();
            (*streamReadableData) << l_parameters[oo];
            l_spaces.append(l_nbDivSpaces2, ' ');
            (*streamReadableData) << l_spaces << "|"; l_spaces.clear();
        }

        (*streamReadableData) << std::endl;
}

