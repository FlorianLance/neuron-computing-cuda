
/**
 * \file GridSearch.cpp
 * \brief defines Generalization
 * \author Florian Lance
 * \date 01/10/14
 */

#include "GridSearch.h"

GridSearch::GridSearch(Model &model) : m_model(&model), m_useCudaInv(true), m_useCudaMult(false)
{}

void GridSearch::setCudaParameters(cbool useCudaInversion, cbool useCudaMultiplication)
{
    m_useCudaInv    = useCudaInversion;
    m_useCudaMult   = useCudaMultiplication;
}

void GridSearch::launchTrainWithAllParameters(const std::string resultsFilePath, const std::string resultsRawFilePath)
{
    if(m_corpusList.size() == 0)
    {
        std::cerr << "-ERROR : at least one corpus must be defined. Grid Search aborted. " << std::endl;
        return;
    }
    if(m_nbNeuronsValues.size() == 0)
    {
        std::cout << "No neuron nb value found, 1 default value set. " << std::endl;
        m_nbNeuronsValues.push_back(800);
    }
    if(m_leakRateValues.size() == 0)
    {
        std::cout << "No leak rate value found, 1 default value set. " << std::endl;
        m_leakRateValues.push_back(0.1);
    }
    if(m_sparcityValues.size() == 0)
    {
        std::cout << "No sparcity value found, automatic value set. " << std::endl;
        m_sparcityValues.push_back(-1);
    }
    if(m_inputScalingValues.size() == 0)
    {
        std::cout << "No input scaling value found, 1 default value set. " << std::endl;
        m_inputScalingValues.push_back(0.1);
    }
    if(m_spectralRadiusValues.size() == 0)
    {
        std::cout << "No spectral radius value found, 1 default value set. " << std::endl;
        m_spectralRadiusValues.push_back(3);
    }
    if(m_ridgeValues.size() == 0)
    {
        std::cout << "No ridge value found, 1 default value set. " << std::endl;
        m_ridgeValues.push_back(1e-5);
    }

    int l_nbTrain = static_cast<int>(m_corpusList.size()*m_nbNeuronsValues.size()*m_leakRateValues.size()*m_sparcityValues.size()*
            m_inputScalingValues.size()*m_spectralRadiusValues.size()*m_ridgeValues.size());

    std::cout << "#################################" << std::endl;
    std::cout << "Start Grid search for training : " << std::endl;
    std::cout << "Number of trains to be done : " << l_nbTrain << std::endl;
    std::cout << "#################################\n" << std::endl;

    std::ofstream l_flowResFileReadableData(resultsFilePath), l_flowResFileRawData(resultsRawFilePath);

    if(!l_flowResFileReadableData)
    {
        std::cerr << "-ERROR : can not write the results in the file " << resultsFilePath << std::endl;
        return;
    }
    if(!l_flowResFileRawData)
    {
        std::cerr << "-ERROR : can not write the results in the file " << resultsRawFilePath << std::endl;
        return;
    }

    l_flowResFileReadableData << "### Grid search results ###\n";
    l_flowResFileReadableData << "Corpus :\n";
//    l_flowResFileRawData << m_corpusList.size() << "\n";

    // write corpus names
    for(int ii = 0; ii < m_corpusList.size(); ++ii)
    {
        l_flowResFileReadableData << m_corpusList[ii] << " " << ii << "\n";
//        l_flowResFileRawData << m_corpusList[ii] << " " << ii << "\n";
    }

    l_flowResFileReadableData << "\nRES 1 : average for all sentences of : correct position and word percentage (between 0% and 100%)\n";
    l_flowResFileReadableData << "RES 2 : average for all sentences of : sentence right and absolute position (0% or 100%) \n";
    l_flowResFileReadableData << "RES 3 : average for all sentences of : correct position and word percentage \n";
    l_flowResFileReadableData << "RES 4 : average for all sentences of : correct position and word percentage (between 0% and 100%), CCW only \n";
    l_flowResFileReadableData << "RES 5 : average for all sentences of : sentence right and absolute position (0% or 100%) CCW only \n\n";

    l_flowResFileReadableData << "\n CORPUS ID | NEURONS | LEAK RATE | SPARCITY | INPUT SCALING |  RIDGE  | SPECTRAL RADIUS |   TIME   |   RES 1   |   RES 2   |   RES 3   |   RES 4   |   RES 5   |\n";
    int l_nbCharParams[] = {11,9,11,10,15,9,17,10,11,11,11,11,11};

    int l_currentTrain = 1;

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

                                std::cout << "######## Start the training number : " << l_currentTrain++ << " / " << l_nbTrain << std::endl;
//                                l_currentParameters.display();

//                                m_model->resetModel(l_currentParameters, true);
                                m_model->resetModelF(l_currentParameters, true);
                                clock_t l_timeTraining = clock();

//                                m_model->launchTraining();
                                m_model->launchTrainingF();
                                double l_time = static_cast<double>((clock() - l_timeTraining)) / CLOCKS_PER_SEC;
                                m_model->retrieveTrainSentences();

//                                m_model->launchTestsF();
//                                m_model->retrieveTestsSentences();

                                m_model->displayResults(false,true);

                                std::vector<double> l_sizeDifferencePercentage, l_sentenceRightAbsolutePercentage, l_correctPositionAndWordPercentage;
                                std::vector<double> l_correctPositionAndWordPercentageCCW, l_sentenceRightAbsolutePercentageCCW;

                                int l_nbTotalWords, l_nbTotalCorrectWords;
                                m_model->compareResults(true, l_sizeDifferencePercentage, l_sentenceRightAbsolutePercentage, l_correctPositionAndWordPercentage, l_nbTotalWords, l_nbTotalCorrectWords);
                                m_model->computeCCWResult(true, l_correctPositionAndWordPercentageCCW, l_sentenceRightAbsolutePercentageCCW);

                                double l_res1 = 0, l_res2 = 0, l_res3 = 0, l_res4 = 0, l_res5 = 0;
                                for(int bb = 0; bb < l_sizeDifferencePercentage.size(); ++bb)
                                {
                                    l_res1 += l_sizeDifferencePercentage[bb];
                                    l_res2 += l_sentenceRightAbsolutePercentage[bb];

                                    l_res4 += l_correctPositionAndWordPercentageCCW[bb];
                                    l_res5 += l_sentenceRightAbsolutePercentageCCW[bb];
                                }

                                l_res1 /= l_sizeDifferencePercentage.size();
                                l_res2 /= l_sizeDifferencePercentage.size();

                                l_res3 = 100.0*l_nbTotalCorrectWords / l_nbTotalWords;

                                l_res4 /= l_correctPositionAndWordPercentageCCW.size();
                                l_res5 /= l_sentenceRightAbsolutePercentageCCW.size();

                                // retrieve string values from parameters
                                    std::ostringstream l_os1,l_os2,l_os3,l_os4,l_os5,l_os6,l_os7,l_os8,l_os9,l_os10,l_os11, l_os12, l_os13;
                                    l_os4.precision(4);l_os8.precision(6),l_os9.precision(3); l_os10.precision(3); l_os11.precision(3); l_os12.precision(3); l_os13.precision(3);
                                    l_os1 << aa; l_os2 << l_currentParameters.m_nbNeurons; l_os3 <<  l_currentParameters.m_leakRate;
                                    l_os4 << l_currentParameters.m_sparcity; l_os5 << l_currentParameters.m_inputScaling; l_os6 << l_currentParameters.m_ridge;
                                    l_os7 << l_currentParameters.m_spectralRadius; l_os8 << l_time; l_os9 << l_res1; l_os10 << l_res2; l_os11 << l_res3;
                                    l_os12 << l_res4; l_os13 << l_res5;

                                    std::vector<std::string> l_parameters;
                                    l_parameters.push_back(l_os1.str()); l_parameters.push_back(l_os2.str()); l_parameters.push_back(l_os3.str()); l_parameters.push_back(l_os4.str());
                                    l_parameters.push_back(l_os5.str()); l_parameters.push_back(l_os6.str()); l_parameters.push_back(l_os7.str()); l_parameters.push_back(l_os8.str());
                                    l_parameters.push_back(l_os9.str()); l_parameters.push_back(l_os10.str()); l_parameters.push_back(l_os11.str()); l_parameters.push_back(l_os12.str());
                                    l_parameters.push_back(l_os13.str());

                                // read raw data
                                    for(int oo = 0; oo < l_parameters.size(); ++oo)
                                    {
                                        l_flowResFileRawData << l_parameters[oo] << " ";
                                    }
                                    l_flowResFileRawData << std::endl;

                                // read readable data
                                    int l_nbSpaces,l_nbDivSpaces1,l_nbDivSpaces2;
                                    std::string l_spaces;

                                    for(int oo = 0; oo < l_parameters.size(); ++oo)
                                    {
                                        l_nbSpaces = l_nbCharParams[oo] - static_cast<int>(l_parameters[oo].size());
                                        l_nbDivSpaces1 = l_nbSpaces/2;
                                        l_nbDivSpaces2 = l_nbSpaces/2 + l_nbSpaces%2;
                                        l_spaces.append(l_nbDivSpaces1, ' ');
                                        l_flowResFileReadableData << l_spaces; l_spaces.clear();
                                        l_flowResFileReadableData << l_parameters[oo];
                                        l_spaces.append(l_nbDivSpaces2, ' ');
                                        l_flowResFileReadableData << l_spaces << "|"; l_spaces.clear();
                                    }

                                    l_flowResFileReadableData << std::endl;
                            }
                        }
                    }
                }
            }
        }

        l_flowResFileRawData << "\n" << std::endl;
    }
}

void GridSearch::setParameterValues(const GridSearch::ReservoirParameter parameterId, cdouble startValue, cdouble endValue, const std::string operation)
{
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

    if(startValue <= endValue)
    {
        while(l_value <= endValue)
        {
            if(parameterId != NEURONS_NB)
            {
                l_valuesD.push_back(l_value);
            }
            else
            {
                l_valuesI.push_back(static_cast<int>(l_value));
            }

            l_value = applyOperation(l_value, operation);
        }
    }
    else
    {
        while(l_value >= endValue)
        {
            if(parameterId != NEURONS_NB)
            {
                l_valuesD.push_back(l_value);
            }
            else
            {
                l_valuesI.push_back(static_cast<int>(l_value));
            }

            l_value = applyOperation(l_value, operation);
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

    if(l_valuesI.size() > 0)
    {
        display(l_valuesI);
    }
    else
    {
        display(l_valuesD);
    }
}

void GridSearch::setCorpusList(const std::vector<std::string> &corpusList)
{
    m_corpusList = corpusList;
}


