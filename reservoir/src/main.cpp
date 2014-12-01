

#include "GridSearch.h"
#include "Generalization.h"

#include "gpuMat/cudaMultiplications.h"


#include "Interface.h"

template<typename T>
void fillRandomStdVec(std::vector<T> vec)
{
    for(int ii = 0; ii < vec.size(); ++ii)
    {
        vec[ii] = static_cast <T> (rand()) / static_cast <T> (RAND_MAX);
    }
}

void fillRandomMat(cv::Mat &mat)
{
    if(mat.depth() == CV_32FC1)
    {
        cv::MatIterator_<float> it = mat.begin<float>(), it_end = mat.end<float>();
        for(;it != it_end; ++it)
        {
            (*it) = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
        }
    }
    else
    {
        cv::MatIterator_<double> it = mat.begin<double>(), it_end = mat.end<double>();
        for(;it != it_end; ++it)
        {
            (*it) =1000.0* static_cast <double> (rand()) / static_cast <double> (RAND_MAX);
        }
    }
}


void testPerfs()
{
////    cv::Mat mat10D(10, 10, CV_64FC1);
////    cv::Mat mat100D(100, 100, CV_64FC1);
//    cv::Mat mat1000D(3000, 3000, CV_32FC1);
//    fillRandomMat(mat1000D);
////    while(true)
////    {

////    }

////    cv::Mat mat2000D(2000, 2000, CV_64FC1);

////    fillRandomMat(mat10D);
////    fillRandomMat(mat100D);

////    fillRandomMat(mat2000D);


//    clock_t time = clock();

//    int nb;
//    cv::Mat resCuda,resCV;

////    {
////        cv::Mat matCudaS,matCudaU,matCudaVT;
////        swUtil::swCuda::squareMatrixSingularValueDecomposition(mat1000D,matCudaS,matCudaU,matCudaVT);
////        save2DMatrixToTextStd("../data/a1.txt", matCudaS);
////        save2DMatrixToTextStd("../data/a2.txt", matCudaU);
////        save2DMatrixToTextStd("../data/a2.txt", matCudaVT);

////        cv::Mat l_tempCudaMult, invCuda1;
////        swUtil::swCuda::blockMatrixMultiplicationD(matCudaS, matCudaU.t(), l_tempCudaMult, 4);

////        save2DMatrixToTextStd("../data/a3.txt", l_tempCudaMult);

////        swUtil::swCuda::blockMatrixMultiplicationD(matCudaVT.t(), l_tempCudaMult, invCuda1, 4);
////        save2DMatrixToTextStd("../data/a4.txt", invCuda1);

////    }

////    displayTime("_2", time); time = clock();
////    swUtil::swCuda::squareMatrixSingularValueDecomposition(mat2000D,matCudaS,matCudaU,matCudaVT);


//    time = clock();
//    {
//        cv::Mat S, U, VT, invCuda1 , SUt;
//        swCuda::squareMatrixSingularValueDecomposition(mat1000D, S, U, VT);
////        swCuda::new_squareMatrixSingularValueDecomposition(mat1000D, SUt, VT);
////        swCuda::low_memory_squareMatrixSingularValueDecomposition<float>(mat1000D, SUt, VT);
////        save2DMatrixToTextStd("../data/1.txt", VT);
//    }


////    displayTime("_2", time); time = clock();
////    {
////        cv::Mat SUt, VT, invCuda2;
////        swCuda::low_memory_squareMatrixSingularValueDecomposition<double>(mat1000D, SUt, VT);
////        save2DMatrixToTextStd("../data/2.txt", VT);

//////        save2DMatrixToTextStd("../data/b1.txt", SUt);
//////        save2DMatrixToTextStd("../data/b2.txt", Vt);

////        swCuda::blockMatrixMultiplicationD(VT.t(), SUt, invCuda2, 4);
//////        save2DMatrixToTextStd("../data/b3.txt", invCuda2);
////    }

//    displayTime("_2", time); time = clock();
////    resCV = mat1000D * mat1000D;
////    displayTime("_1", time); time = clock();
////    swUtil::swCuda::blockMatrixMultiplicationD(mat1000D,mat1000D,resCuda,2);
////    displayTime("_2", time); time = clock();
////    compareMatrices<double>(resCV, resCuda, nb, 10); time = clock();
////    qDebug() << "-> 2" << " " << nb << " prec : 10 \n";
////    time = clock();
////    swUtil::swCuda::blockMatrixMultiplicationD(mat1000D,mat1000D,resCuda,3);
////    displayTime("_3", time); time = clock();
////    compareMatrices<double>(resCV, resCuda, nb, 10); time = clock();
////    qDebug() << "-> 4" << " " << nb << " prec : 10 \n";

////    swUtil::swCuda::blockMatrixMultiplicationD(mat1000D,mat1000D,resCuda,4);
////    displayTime("_4", time); time = clock();
////    compareMatrices<double>(resCV, resCuda, nb, 10); time = clock();
////    qDebug() << "-> 8" << " " << nb << " prec : 10 \n";

////    swUtil::swCuda::blockMatrixMultiplicationD(mat1000D,mat1000D,resCuda,5);
////    displayTime("_5", time); time = clock();
////    compareMatrices<double>(resCV, resCuda, nb, 10); time = clock();
////    qDebug() << "-> 8" << " " << nb << " prec : 10 \n";


////    swUtil::swCuda::blockMatrixMultiplicationD(mat1000D,mat1000D,resCuda,6);
////    displayTime("_5", time); time = clock();
////    compareMatrices<double>(resCV, resCuda, nb, 10); time = clock();
////    qDebug() << "-> 8" << " " << nb << " prec : 10 \n";


////    save2DMatrixToTextStd("../data/2.txt", resCV);
}


int main(int argc, char* argv[])
{
    srand(1);
    culaWarmup(1);

    // ############################################## INTERFACE

    QApplication l_oApp(argc, argv);
    Interface l_oViewerInterface;
    l_oViewerInterface.resize(1800, 900);
    l_oViewerInterface.move(50,50);
    l_oViewerInterface.show();

    return l_oApp.exec();

    // ############################################## TESTS

//    testPerfs();
//    culaStop();

//    generateSubRandomCorpus("../data/input/Corpus/462.txt", "../data/input/Corpus/120.txt", 120);

//    return 0;

    // ############################################## GENERALISATION 1

//        ModelParameters l_parameters;
//        l_parameters.m_nbNeurons = 1000;
//        l_parameters.m_leakRate  = 0.25;
//        l_parameters.m_inputScaling = 0.2;
//        l_parameters.m_spectralRadius = 7.0;
//        l_parameters.m_ridge = 1e-5;
//        l_parameters.m_corpusFilePath = "../data/input/Corpus/120.txt";
////        l_parameters.m_corpusFilePath = "../data/input/Corpus/462.txt";

//        l_parameters.m_sparcity = 10.0 / l_parameters.m_nbNeurons;
//        l_parameters.m_useCudaInv= true;
//        l_parameters.m_useCudaMult = true;

//        Model l_modelGeneralization(l_parameters);

//        std::string l_grammarStd[] ={"and","is","of","the","to",".","-ed","-ing","-s","by","it","that","was","did",",","from"};
//        std::string l_structureStd[] = {"P0","A1","O2","R3"};

//    //    std::string l_grammarStd[] ={"-ga","-ni","-wo","-yotte","-o","-to","sore"};
//    //    std::string l_structureStd[] = {"P0","A1","O2","R3", "Q0"};

//        Sentence l_grammar = Sentence(l_grammarStd, l_grammarStd + sizeof(l_grammarStd) / sizeof(std::string));
//        Sentence l_structure = Sentence(l_structureStd, l_structureStd + sizeof(l_structureStd) / sizeof(std::string));
//        l_modelGeneralization.setGrammar(l_grammar, l_structure);

//        Generalization l_generalization(l_modelGeneralization);

//        l_generalization.randomChangeCorpusGeneralization(30, "../data/input/Corpus/randomizedCorpus_120_30.txt");
//        l_generalization.randomChangeCorpusGeneralization(60, "../data/input/Corpus/randomizedCorpus_120_60.txt");
//        l_generalization.randomChangeCorpusGeneralization(90, "../data/input/Corpus/randomizedCorpus_120_90.txt");
//        l_generalization.randomChangeCorpusGeneralization(120, "../data/input/Corpus/randomizedCorpus_120_120.txt");
//        l_generalization.randomChangeCorpusGeneralization(30, "../data/input/Corpus/randomizedCorpus_120_30_s.txt", false);
//        l_generalization.randomChangeCorpusGeneralization(60, "../data/input/Corpus/randomizedCorpus_120_60_s.txt", false);
//        l_generalization.randomChangeCorpusGeneralization(90, "../data/input/Corpus/randomizedCorpus_120_90_s.txt", false);
//        l_generalization.randomChangeCorpusGeneralization(120, "../data/input/Corpus/randomizedCorpus_120_120_s.txt", false);

//        l_generalization.randomChangeCorpusGeneralization(100, "../data/input/Corpus/randomizedCorpus_462_100.txt");
//        l_generalization.randomChangeCorpusGeneralization(200, "../data/input/Corpus/randomizedCorpus_462_200.txt");
//        l_generalization.randomChangeCorpusGeneralization(300, "../data/input/Corpus/randomizedCorpus_462_300.txt");
//        l_generalization.randomChangeCorpusGeneralization(462, "../data/input/Corpus/randomizedCorpus_462_462.txt");
//        l_generalization.randomChangeCorpusGeneralization(100, "../data/input/Corpus/randomizedCorpus_462_100_s.txt", false);
//        l_generalization.randomChangeCorpusGeneralization(200, "../data/input/Corpus/randomizedCorpus_462_200_s.txt", false);
//        l_generalization.randomChangeCorpusGeneralization(300, "../data/input/Corpus/randomizedCorpus_462_300_s.txt", false);
//        l_generalization.randomChangeCorpusGeneralization(462, "../data/input/Corpus/randomizedCorpus_462_462_s.txt", false);


    //    GridSearch l_gridSearch(l_modelGeneralization);
    //    l_gridSearch.setCudaParameters(true, true);
    //    l_gridSearch.setParameterValues(GridSearch::NEURONS_NB,     5000, 5000, "+500");
    //    l_gridSearch.setParameterValues(GridSearch::LEAK_RATE,      0.25, 0.25,   "+0.05");
    //    l_gridSearch.setParameterValues(GridSearch::INPUT_SCALING,  0.2, 0.2, "+0.2");
    //    l_gridSearch.setParameterValues(GridSearch::SPECTRAL_RADIUS,7, 7, "+2.0");

    //    std::vector<std::string> l_corpusList;
    //    l_corpusList.push_back("../data/input/Corpus/120.txt");
    //    l_corpusList.push_back("../data/input/Corpus/randomizedCorpus_120_30.txt");
    //    l_corpusList.push_back("../data/input/Corpus/randomizedCorpus_120_60.txt");
    //    l_corpusList.push_back("../data/input/Corpus/randomizedCorpus_120_90.txt");
    //    l_corpusList.push_back("../data/input/Corpus/randomizedCorpus_120_120.txt");
    //    l_corpusList.push_back("../data/input/Corpus/randomizedCorpus_120_30_s.txt");
    //    l_corpusList.push_back("../data/input/Corpus/randomizedCorpus_120_60_s.txt");
    //    l_corpusList.push_back("../data/input/Corpus/randomizedCorpus_120_90_s.txt");
    //    l_corpusList.push_back("../data/input/Corpus/randomizedCorpus_120_120_s.txt");

    //    l_gridSearch.setCorpusList(l_corpusList);
    //    l_gridSearch.launchTrainWithAllParameters("../data/Results/random_res/grid_search_1.txt", "../data/Results/random_res/grid_search_raw_1.txt");
    //    l_gridSearch.launchTrainWithAllParameters("../data/Results/random_res/grid_search_2.txt", "../data/Results/random_res/grid_search_raw_2.txt");
    //    l_gridSearch.launchTrainWithAllParameters("../data/Results/random_res/grid_search_3.txt", "../data/Results/random_res/grid_search_raw_3.txt");
    //    l_gridSearch.launchTrainWithAllParameters("../data/Results/random_res/grid_search_4.txt", "../data/Results/random_res/grid_search_raw_4.txt");
    //    l_gridSearch.launchTrainWithAllParameters("../data/Results/random_res/grid_search_5.txt", "../data/Results/random_res/grid_search_raw_5.txt");

//        culaStop();
//        return 0;

    // ############################################## GENERALISATION 2

//    ModelParameters l_parameters;
//    l_parameters.m_nbNeurons = 1000;
//    l_parameters.m_leakRate  = 0.25;
//    l_parameters.m_inputScaling = 0.2;
//    l_parameters.m_spectralRadius = 7.0;
//    l_parameters.m_ridge = 1e-5;

//    l_parameters.m_corpusFilePath = "../data/input/Corpus/120.txt";
////    l_parameters.m_corpusFilePath = "../data/input/Corpus/japan.txt";

//    l_parameters.m_sparcity = 10.0 / l_parameters.m_nbNeurons;
//    l_parameters.m_useCudaInv= true;
//    l_parameters.m_useCudaMult = true;

//    Model l_modelGeneralization(l_parameters);

//    std::string l_grammarStd[] ={"and","is","of","the","to",".","-ed","-ing","-s","by","it","that","was","did",",","from"};
////    std::string l_grammarStd[] ={"-ga","-ni","-wo","-yotte","-o","-to","sore"};

//    std::string l_structureStd[] = {"P0","A1","O2","R3"};
////    std::string l_structureStd[] = {"P0","A1","O2","R3", "Q0"};

//    Sentence l_grammar = Sentence(l_grammarStd, l_grammarStd + sizeof(l_grammarStd) / sizeof(std::string));
//    Sentence l_structure = Sentence(l_structureStd, l_structureStd + sizeof(l_structureStd) / sizeof(std::string));
//    l_modelGeneralization.setGrammar(l_grammar, l_structure);

//    l_generalization.startXVerification();

//    return 0;


    // ############################################## GRIDSEARCH

    Model l_gridSearchModel;
//    std::string l_grammarStd[] ={"and","is","of","the","to",".","-ed","-ing","-s","by","it","that","was","did",",","from"};
//    std::string l_structureStd[] = {"P0","A1","O2","R3"};

    std::string l_grammarStd[] ={"-ga","-ni","-wo","-yotte","-o","-to","sore"};
    std::string l_structureStd[] = {"P0","A1","O2","R3", "Q0"};

    Sentence l_grammar = Sentence(l_grammarStd, l_grammarStd + sizeof(l_grammarStd) / sizeof(std::string));
    Sentence l_structure = Sentence(l_structureStd, l_structureStd + sizeof(l_structureStd) / sizeof(std::string));
    l_gridSearchModel.setGrammar(l_grammar, l_structure);

    GridSearch l_gridSearch(l_gridSearchModel);

    l_gridSearch.setCudaParameters(true, true);
//    l_gridSearch.setParameterValues(GridSearch::NEURONS_NB,     100, 1000, "+100");
    l_gridSearch.setParameterValues(GridSearch::NEURONS_NB,     1000, 1000, "+100");
//    l_gridSearch.setParameterValues(GridSearch::NEURONS_NB,     1000, 6000, "+1000");
//    l_gridSearch.setParameterValues(GridSearch::NEURONS_NB,     500, 1000, "+100");
    l_gridSearch.setParameterValues(GridSearch::LEAK_RATE,      0.25, 0.25,   "+0.05");
    l_gridSearch.setParameterValues(GridSearch::INPUT_SCALING,  0.2, 0.2, "+0.2");
    l_gridSearch.setParameterValues(GridSearch::SPECTRAL_RADIUS,7, 7, "+2.0");
//    l_gridSearch.setParameterValues(GridSearch::SPECTRAL_RADIUS,1, 15, "+1.0");

    //  l_gridSearch.setParameterValues(GridSearch::INPUT_SCALING,  0.001, 100, "*10");
    //  l_gridSearch.setParameterValues(GridSearch::RIDGE,          1e-5, 10, "*10.0");
//      l_gridSearch.setParameterValues(GridSearch::SPARCITY,       0.0006, 0.0036, "+0.0006");

    std::vector<std::string> l_corpusList;
    l_corpusList.push_back("../data/input/Corpus/japan.txt");
//      l_corpusList.push_back("../data/input/Corpus/10.txt");
//      l_corpusList.push_back("../data/input/Corpus/50.txt");
//      l_corpusList.push_back("../data/input/Corpus/100.txt");
//          l_corpusList.push_back("../data/input/Corpus/120.txt");
//          l_corpusList.push_back("../data/input/Corpus/aaa.txt");
//      l_corpusList.push_back("../data/input/Corpus/200.txt");
//      l_corpusList.push_back("../data/input/Corpus/300.txt");
//      l_corpusList.push_back("../data/input/Corpus/462.txt");
//    l_corpusList.push_back("../data/input/Corpus/d1_50_462.txt");
//    l_corpusList.push_back("../data/input/Corpus/d2_50_462.txt");
//    l_corpusList.push_back("../data/input/Corpus/d3_50_462.txt");
//    l_corpusList.push_back("../data/input/Corpus/d4_50_462.txt");
//    l_corpusList.push_back("../data/input/Corpus/d5_50_462.txt");
//    l_corpusList.push_back("../data/input/Corpus/d6_50_462.txt");
//    l_corpusList.push_back("../data/input/Corpus/d7_50_462.txt");
//    l_corpusList.push_back("../data/input/Corpus/d8_50_462.txt");
//    l_corpusList.push_back("../data/input/Corpus/d9_50_462.txt");
//    l_corpusList.push_back("../data/input/Corpus/d10_50_462.txt");

    l_gridSearch.setCorpusList(l_corpusList);
    l_gridSearch.launchTrainWithAllParameters("../data/Results/grid_search.txt", "../data/Results/grid_search_raw.txt");
//    l_gridSearch.setCudaParameters(true, false);
//    l_gridSearch.launchTrainWithAllParameters("../data/Results/grid_search-noCudaMult.txt", "../data/Results/grid_search_raw-noCudaMult.txt");

    culaStop();
    return 0;
}
