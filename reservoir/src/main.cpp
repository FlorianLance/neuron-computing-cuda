

#include "GridSearchQtObject.h"
#include "Generalization.h"

#include "gpuMat/cudaMultiplications.h"

int main(int argc, char* argv[])
{
    srand(1);
    culaWarmup(1);

//    convertCorpus("../data/input/Corpus/old_format/26.txt", "../data/input/Corpus/26.txt");
//    convertCorpus("../data/input/Corpus/old_format/354.txt", "../data/input/Corpus/354.txt");

    // ############################################## GENERALISATION 1

    /*
        ModelParameters l_parameters;
        l_parameters.m_ridge = 1e-5;
        l_parameters.m_corpusFilePath = "../data/input/Corpus/120.txt";
//        l_parameters.m_corpusFilePath = "../data/input/Corpus/japan.txt";
//        l_parameters.m_corpusFilePath = "../data/input/Corpus/462.txt";

        Model l_modelGeneralization(l_parameters);

        std::string l_grammarStd[] ={"and","is","of","the","to",".","-ed","-ing","-s","by","it","that","was","did",",","from"};
        std::string l_structureStd[] = {"P0","A1","O2","R3"};

//        std::string l_grammarStd[] ={"-ga","-ni","-wo","-yotte","-o","-to","sore"};
//        std::string l_structureStd[] = {"P0","A1","O2","R3", "Q0"};

        Sentence l_grammar = Sentence(l_grammarStd, l_grammarStd + sizeof(l_grammarStd) / sizeof(std::string));
        Sentence l_structure = Sentence(l_structureStd, l_structureStd + sizeof(l_structureStd) / sizeof(std::string));
        l_modelGeneralization.setGrammar(l_grammar, l_structure);

        Generalization l_generalization(l_modelGeneralization);
//        l_generalization.randomChangeCorpusGeneralization(5,  "../data/input/Corpus/randomizedCorpus_24_5-m.txt", Generalization::MEANING);
//        l_generalization.randomChangeCorpusGeneralization(10, "../data/input/Corpus/randomizedCorpus_24_10-m.txt", Generalization::MEANING);
//        l_generalization.randomChangeCorpusGeneralization(15, "../data/input/Corpus/randomizedCorpus_24_15-m.txt", Generalization::MEANING);
//        l_generalization.randomChangeCorpusGeneralization(20, "../data/input/Corpus/randomizedCorpus_24_20-m.txt", Generalization::MEANING);
//        l_generalization.randomChangeCorpusGeneralization(24, "../data/input/Corpus/randomizedCorpus_24_24-m.txt", Generalization::MEANING);

//        l_generalization.randomChangeCorpusGeneralization(5,  "../data/input/Corpus/randomizedCorpus_24_5-i.txt", Generalization::INFOS);
//        l_generalization.randomChangeCorpusGeneralization(10, "../data/input/Corpus/randomizedCorpus_24_10-i.txt", Generalization::INFOS);
//        l_generalization.randomChangeCorpusGeneralization(15, "../data/input/Corpus/randomizedCorpus_24_15-i.txt", Generalization::INFOS);
//        l_generalization.randomChangeCorpusGeneralization(20, "../data/input/Corpus/randomizedCorpus_24_20-i.txt", Generalization::INFOS);
//        l_generalization.randomChangeCorpusGeneralization(24, "../data/input/Corpus/randomizedCorpus_24_24-i.txt", Generalization::INFOS);

//        l_generalization.randomChangeCorpusGeneralization(5,  "../data/input/Corpus/randomizedCorpus_24_5-s.txt", Generalization::SENTENCES);
//        l_generalization.randomChangeCorpusGeneralization(10, "../data/input/Corpus/randomizedCorpus_24_10-s.txt", Generalization::SENTENCES);
//        l_generalization.randomChangeCorpusGeneralization(15, "../data/input/Corpus/randomizedCorpus_24_15-s.txt", Generalization::SENTENCES);
//        l_generalization.randomChangeCorpusGeneralization(20, "../data/input/Corpus/randomizedCorpus_24_20-s.txt", Generalization::SENTENCES);
//        l_generalization.randomChangeCorpusGeneralization(24, "../data/input/Corpus/randomizedCorpus_24_24-s.txt", Generalization::SENTENCES);

//        l_generalization.randomChangeCorpusGeneralization(30, "../data/input/Corpus/randomizedCorpus_120_30-m.txt", Generalization::MEANING);
//        l_generalization.randomChangeCorpusGeneralization(60, "../data/input/Corpus/randomizedCorpus_120_60-m.txt", Generalization::MEANING);
//        l_generalization.randomChangeCorpusGeneralization(90, "../data/input/Corpus/randomizedCorpus_120_90-m.txt", Generalization::MEANING);
//        l_generalization.randomChangeCorpusGeneralization(120, "../data/input/Corpus/randomizedCorpus_120_120-m.txt", Generalization::MEANING);

//        l_generalization.randomChangeCorpusGeneralization(30, "../data/input/Corpus/randomizedCorpus_120_30-i.txt", Generalization::INFOS);
//        l_generalization.randomChangeCorpusGeneralization(60, "../data/input/Corpus/randomizedCorpus_120_60-i.txt", Generalization::INFOS);
//        l_generalization.randomChangeCorpusGeneralization(90, "../data/input/Corpus/randomizedCorpus_120_90-i.txt", Generalization::INFOS);
//        l_generalization.randomChangeCorpusGeneralization(120, "../data/input/Corpus/randomizedCorpus_120_120-i.txt", Generalization::INFOS);

//        l_generalization.randomChangeCorpusGeneralization(30, "../data/input/Corpus/randomizedCorpus_120_30-s.txt", Generalization::SENTENCES);
//        l_generalization.randomChangeCorpusGeneralization(60, "../data/input/Corpus/randomizedCorpus_120_60-s.txt", Generalization::SENTENCES);
//        l_generalization.randomChangeCorpusGeneralization(90, "../data/input/Corpus/randomizedCorpus_120_90-s.txt", Generalization::SENTENCES);
//        l_generalization.randomChangeCorpusGeneralization(120, "../data/input/Corpus/randomizedCorpus_120_120-s.txt", Generalization::SENTENCES);

        GridSearch l_gridSearch(l_modelGeneralization);
        l_gridSearch.setCudaParameters(true, true);
        l_gridSearch.setParameterValues(GridSearch::NEURONS_NB,     5000, 5000, "+50", true, 1);
        l_gridSearch.setParameterValues(GridSearch::LEAK_RATE,      0.25, 0.25,   "+0.05");
        l_gridSearch.setParameterValues(GridSearch::INPUT_SCALING,  0.2, 0.2, "+0.2");
        l_gridSearch.setParameterValues(GridSearch::SPECTRAL_RADIUS,7, 7, "+1.0");

        std::vector<std::string> l_corpusList;
//        l_corpusList.push_back("../data/input/Corpus/japan.txt");
        // random meaning
//        l_corpusList.push_back("../data/input/Corpus/randomizedCorpus_24_5-m.txt");
//        l_corpusList.push_back("../data/input/Corpus/randomizedCorpus_24_10-m.txt");
//        l_corpusList.push_back("../data/input/Corpus/randomizedCorpus_24_15-m.txt");
//        l_corpusList.push_back("../data/input/Corpus/randomizedCorpus_24_20-m.txt");
//        l_corpusList.push_back("../data/input/Corpus/randomizedCorpus_24_24-m.txt");
        // random infos
//        l_corpusList.push_back("../data/input/Corpus/randomizedCorpus_24_5-i.txt");
//        l_corpusList.push_back("../data/input/Corpus/randomizedCorpus_24_10-i.txt");
//        l_corpusList.push_back("../data/input/Corpus/randomizedCorpus_24_15-i.txt");
//        l_corpusList.push_back("../data/input/Corpus/randomizedCorpus_24_20-i.txt");
//        l_corpusList.push_back("../data/input/Corpus/randomizedCorpus_24_24-i.txt");
        // random sentences
//        l_corpusList.push_back("../data/input/Corpus/randomizedCorpus_24_5-s.txt");
//        l_corpusList.push_back("../data/input/Corpus/randomizedCorpus_24_10-s.txt");
//        l_corpusList.push_back("../data/input/Corpus/randomizedCorpus_24_15-s.txt");
//        l_corpusList.push_back("../data/input/Corpus/randomizedCorpus_24_20-s.txt");
//        l_corpusList.push_back("../data/input/Corpus/randomizedCorpus_24_24-s.txt");


        l_corpusList.push_back("../data/input/Corpus/120.txt");
        l_corpusList.push_back("../data/input/Corpus/200.txt");
        l_corpusList.push_back("../data/input/Corpus/300.txt");
        l_corpusList.push_back("../data/input/Corpus/462.txt");
         // random infos
//        l_corpusList.push_back("../data/input/Corpus/randomizedCorpus_120_30-i.txt");
//        l_corpusList.push_back("../data/input/Corpus/randomizedCorpus_120_60-i.txt");
//        l_corpusList.push_back("../data/input/Corpus/randomizedCorpus_120_90-i.txt");
//        l_corpusList.push_back("../data/input/Corpus/randomizedCorpus_120_120-i.txt");
        // random sentences
//        l_corpusList.push_back("../data/input/Corpus/randomizedCorpus_120_30-s.txt");
//        l_corpusList.push_back("../data/input/Corpus/randomizedCorpus_120_60-s.txt");
//        l_corpusList.push_back("../data/input/Corpus/randomizedCorpus_120_90-s.txt");
//        l_corpusList.push_back("../data/input/Corpus/randomizedCorpus_120_120-s.txt");
        // random meaning
//        l_corpusList.push_back("../data/input/Corpus/randomizedCorpus_120_30-m.txt");
//        l_corpusList.push_back("../data/input/Corpus/randomizedCorpus_120_60-m.txt");
//        l_corpusList.push_back("../data/input/Corpus/randomizedCorpus_120_90-m.txt");
//        l_corpusList.push_back("../data/input/Corpus/randomizedCorpus_120_120-m.txt");

        l_gridSearch.setCorpusList(l_corpusList);
        l_gridSearch.launchTrainWithAllParameters("../data/Results/random_res/grid_search.txt", "../data/Results/random_res/grid_search_raw.txt", true, false, false);
//        l_gridSearch.launchTrainWithAllParameters("../data/Results/random_res/grid_search_2.txt", "../data/Results/random_res/grid_search_raw_2.txt");
//        l_gridSearch.launchTrainWithAllParameters("../data/Results/random_res/grid_search_3.txt", "../data/Results/random_res/grid_search_raw_3.txt");
//        l_gridSearch.launchTrainWithAllParameters("../data/Results/random_res/grid_search_4.txt", "../data/Results/random_res/grid_search_raw_4.txt");
//        l_gridSearch.launchTrainWithAllParameters("../data/Results/random_res/grid_search_5.txt", "../data/Results/random_res/grid_search_raw_5.txt");

        culaStop();
        return 0;


        */

    // ############################################## GENERALISATION 2

//    ModelParameters l_parameters;
//    l_parameters.m_nbNeurons = 400;
//    l_parameters.m_leakRate  = 0.25;
//    l_parameters.m_inputScaling = 0.2;
//    l_parameters.m_spectralRadius = 7.0;
//    l_parameters.m_ridge = 1e-5;

////    l_parameters.m_corpusFilePath = "../data/input/Corpus/120.txt";
//    l_parameters.m_corpusFilePath = "../data/input/Corpus/japan.txt";

//    l_parameters.m_sparcity = 10.0 / l_parameters.m_nbNeurons;
//    l_parameters.m_useCudaInv= true;
//    l_parameters.m_useCudaMult = true;

//    Model l_modelGeneralization(l_parameters);

////    std::string l_grammarStd[] ={"and","is","of","the","to",".","-ed","-ing","-s","by","it","that","was","did",",","from"};
//    std::string l_grammarStd[] ={"-ga","-ni","-wo","-yotte","-o","-to","sore"};

////    std::string l_structureStd[] = {"P0","A1","O2","R3"};
//    std::string l_structureStd[] = {"P0","A1","O2","R3", "Q0"};

//    Sentence l_grammar = Sentence(l_grammarStd, l_grammarStd + sizeof(l_grammarStd) / sizeof(std::string));
//    Sentence l_structure = Sentence(l_structureStd, l_structureStd + sizeof(l_structureStd) / sizeof(std::string));
//    l_modelGeneralization.setGrammar(l_grammar, l_structure);

//    Generalization l_generalization(l_modelGeneralization);
//    l_generalization.startXVerification();

//    return 0;


    // ############################################## GRIDSEARCH

//    Model l_gridSearchModel;
//    std::string l_grammarStd[] ={"and","is","of","the","to",".","-ed","-ing","-s","by","it","that","was","did",",","from"};
//    std::string l_structureStd[] = {"P0","A1","O2","R3"};

////    std::string l_grammarStd[] ={"-ga","-ni","-wo","-yotte","-o","-to","sore"};
////    std::string l_structureStd[] = {"P0","A1","O2","R3", "Q0"};

//    Sentence l_grammar = Sentence(l_grammarStd, l_grammarStd + sizeof(l_grammarStd) / sizeof(std::string));
//    Sentence l_structure = Sentence(l_structureStd, l_structureStd + sizeof(l_structureStd) / sizeof(std::string));
//    l_gridSearchModel.setGrammar(l_grammar, l_structure);

//    GridSearch l_gridSearch(l_gridSearchModel);

//    l_gridSearch.setCudaParameters(true, true);
////    l_gridSearch.setParameterValues(GridSearch::NEURONS_NB,     100, 1000, "+100");
//    l_gridSearch.setParameterValues(GridSearch::NEURONS_NB,     2000, 2000, "+100");
////    l_gridSearch.setParameterValues(GridSearch::NEURONS_NB,     1000, 6000, "+1000");
////    l_gridSearch.setParameterValues(GridSearch::NEURONS_NB,     500, 1000, "+100");
//    l_gridSearch.setParameterValues(GridSearch::LEAK_RATE,      0.25, 0.25,   "+0.05");
//    l_gridSearch.setParameterValues(GridSearch::INPUT_SCALING,  0.2, 0.2, "+0.2");
//    l_gridSearch.setParameterValues(GridSearch::SPECTRAL_RADIUS,7, 7, "+2.0");
////    l_gridSearch.setParameterValues(GridSearch::SPECTRAL_RADIUS,1, 15, "+1.0");

//    //  l_gridSearch.setParameterValues(GridSearch::INPUT_SCALING,  0.001, 100, "*10");
//    //  l_gridSearch.setParameterValues(GridSearch::RIDGE,          1e-5, 10, "*10.0");
////      l_gridSearch.setParameterValues(GridSearch::SPARCITY,       0.0006, 0.0036, "+0.0006");

//    std::vector<std::string> l_corpusList;
////    l_corpusList.push_back("../data/input/Corpus/japan.txt");
////      l_corpusList.push_back("../data/input/Corpus/10.txt");
//      l_corpusList.push_back("../data/input/Corpus/50.txt");
////      l_corpusList.push_back("../data/input/Corpus/100.txt");
////          l_corpusList.push_back("../data/input/Corpus/120.txt");
////          l_corpusList.push_back("../data/input/Corpus/aaa.txt");
////      l_corpusList.push_back("../data/input/Corpus/200.txt");
////      l_corpusList.push_back("../data/input/Corpus/300.txt");
////      l_corpusList.push_back("../data/input/Corpus/462.txt");
////    l_corpusList.push_back("../data/input/Corpus/d1_50_462.txt");
////    l_corpusList.push_back("../data/input/Corpus/d2_50_462.txt");
////    l_corpusList.push_back("../data/input/Corpus/d3_50_462.txt");
////    l_corpusList.push_back("../data/input/Corpus/d4_50_462.txt");
////    l_corpusList.push_back("../data/input/Corpus/d5_50_462.txt");
////    l_corpusList.push_back("../data/input/Corpus/d6_50_462.txt");
////    l_corpusList.push_back("../data/input/Corpus/d7_50_462.txt");
////    l_corpusList.push_back("../data/input/Corpus/d8_50_462.txt");
////    l_corpusList.push_back("../data/input/Corpus/d9_50_462.txt");
////    l_corpusList.push_back("../data/input/Corpus/d10_50_462.txt");

//    l_gridSearch.setCorpusList(l_corpusList);
//    l_gridSearch.launchTrainWithAllParameters("../data/Results/grid_search.txt", "../data/Results/grid_search_raw.txt");
////    l_gridSearch.setCudaParameters(true, false);
////    l_gridSearch.launchTrainWithAllParameters("../data/Results/grid_search-noCudaMult.txt", "../data/Results/grid_search_raw-noCudaMult.txt");

//    culaStop();
//    return 0;

//     ############################################## MODEL - TEST SAVE LOAD TRAINING
//    ModelParameters l_parameters;

//    l_parameters.m_nbNeurons = 400;
//    l_parameters.m_leakRate  = 0.25;
//    l_parameters.m_inputScaling = 0.2;
//    l_parameters.m_spectralRadius = 7.0;
//    l_parameters.m_ridge = 1e-5;
//    l_parameters.m_corpusFilePath = "../data/input/Corpus/japan_test.txt";

//    l_parameters.m_sparcity = 10.0 / l_parameters.m_nbNeurons;
//    l_parameters.m_useCudaInv= true;
//    l_parameters.m_useCudaMult = true;
//    l_parameters.m_useLoadedTraining = false;

//    Model l_model(l_parameters);

//    std::string l_grammarStd[] ={"-ga","-ni","-wo","-yotte","-o","-to","sore"};
//    std::string l_structureStd[] = {"P0","A1","O2","R3", "Q0"};

//    Sentence l_grammar = Sentence(l_grammarStd, l_grammarStd + sizeof(l_grammarStd) / sizeof(std::string));
//    Sentence l_structure = Sentence(l_structureStd, l_structureStd + sizeof(l_structureStd) / sizeof(std::string));
//    l_model.setGrammar(l_grammar, l_structure);

//    l_model.resetModelParameters(l_parameters,true);
//    l_model.launchTraining();
//    l_model.retrieveTrainSentences();
//    l_model.saveTraining("../data/training/last");

//    std::cout << "load training : " << std::endl;
//    l_model.loadTraining("../data/training/last");
//    l_parameters.m_useLoadedTraining = true;

//    std::cout << "resetModelParameters : " << std::endl;
//    l_model.resetModelParameters(l_parameters,true);

//    std::cout << "launchTests : " << std::endl;
//    l_model.launchTests();
//    l_model.retrieveTestsSentences();

//    culaStop();
//    return 0;

}
