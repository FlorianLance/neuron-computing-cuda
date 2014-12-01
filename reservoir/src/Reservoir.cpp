
/**
 * \file Reservoir.cpp
 * \brief defines Reservoir
 * \author Florian Lance
 * \date 01/10/14
 */

// reservoir-cuda
#include "Reservoir.h"

// Qt
#include <QtGui>


using namespace std;

void fillRandomMat_(cv::Mat &mat)
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

Reservoir::Reservoir()
{
    m_initialized = false;
    m_verbose = true;

    m_useCudaInversion      = true;
    m_useCudaMultiplication = false;
}

Reservoir::Reservoir(cuint nbNeurons, cdouble spectralRadius, cdouble inputScaling, cdouble leakRate, cdouble sparcity, cdouble ridge, cbool verbose)
    : m_nbNeurons(nbNeurons), m_spectralRadius(spectralRadius), m_inputScaling(inputScaling), m_leakRate(leakRate), m_ridge(ridge), m_verbose(verbose)
{
    if(sparcity > 0.0)
    {
        m_sparcity = sparcity;
    }
    else
    {
        m_sparcity = 10.0/m_nbNeurons;
    }
    m_initialized = true;

//    cvNamedWindow("reservoir_display", CV_WINDOW_AUTOSIZE | CV_GUI_NORMAL);
//    cvMoveWindow("reservoir_display",200,200);
}

void Reservoir::setCudaProperties(cbool cudaInv, cbool cudaMult)
{
    m_useCudaInversion = cudaInv;
    m_useCudaMultiplication = cudaMult;
}

void Reservoir::generateMatrixW()
{
    // debug
    displayTime("START : generate W ", m_oTime, false, m_verbose);

    // init w matrix [N x N]
    m_w = cv::Mat(m_nbNeurons, m_nbNeurons, CV_64FC1, cv::Scalar(0.0));

//    #pragma omp parallel for // NO ! rand() not thread safe
        for(int ii = 0; ii < m_w.rows*m_w.cols;++ii)
        {
            if(static_cast <double> (rand()) / static_cast <double> (RAND_MAX) < m_sparcity)
            {
                double r = static_cast <double> (rand()) / static_cast <double> (RAND_MAX);
                m_w.at<double>(ii) = (r -0.5) * m_spectralRadius;
            }
        }

    // compute eigen max value with CUDA
    // l_s[0] is the largest eigen value
    //   m_w *= m_spectralRadius / static_cast<double>(l_s.at<float>(0)); // python version (coslty)

    // debug
    displayTime("END : generate W ", m_oTime, false, m_verbose);
}

void Reservoir::generateWIn(cuint dimInput)
{
    // debug
    displayTime("START : generate WIn ", m_oTime, false, m_verbose);

    // init wIn
        m_wIn = cv::Mat(m_nbNeurons, dimInput + 1, CV_64FC1);

    // fill wIn matrix with random values [0, 1]
        cv::MatIterator_<double> it = m_wIn.begin<double>(), it_end = m_wIn.end<double>();
        double l_randMax = static_cast <double> (RAND_MAX);
        for(;it != it_end; ++it)
        {
            (*it) = (static_cast <double> (rand()) / l_randMax) * m_inputScaling;
        }

    // debug
    displayTime("END : generate WIn ", m_oTime, false, m_verbose);
}

void Reservoir::tikhonovRegularization(const cv::Mat &xTot, const cv::Mat &yTeacher, cuint dimInput)
{
    int l_subdivisionBlocks = 2;
    if(m_nbNeurons > 3000)
    {
        l_subdivisionBlocks = 4;
    }
    if(m_nbNeurons > 6000)
    {
        l_subdivisionBlocks = 6;
    }
    if(m_nbNeurons > 8000)
    {
        l_subdivisionBlocks = 8;
    }

    displayTime("START : tikhonovRegularization ", m_oTime, false, m_verbose);

    cv::Mat l_xTotReshaped(xTot.size[1], xTot.size[0] * xTot.size[2], CV_64FC1);

    #pragma omp parallel for
        for(int ii = 0; ii < xTot.size[0]; ++ii)
        {
            for(int jj = 0; jj < xTot.size[1]; ++jj)
            {
                for(int kk = 0; kk < xTot.size[2]; ++kk)
                {
                    l_xTotReshaped.at<double>(jj, ii*xTot.size[2] + kk) = xTot.at<double>(ii,jj,kk);
                }
            }
        }
    // end pragma

    displayTime("1 : tikhonovRegularization ", m_oTime, false, m_verbose);
    cv::Mat l_mat2inv;

    if(m_useCudaInversion)
    {
        swCuda::blockMatrixMultiplicationD(l_xTotReshaped,l_xTotReshaped.t(), l_mat2inv, l_subdivisionBlocks);
    }
    else
    {
        l_mat2inv = (l_xTotReshaped * l_xTotReshaped.t());
    }

    l_mat2inv += (cv::Mat::eye(1 + dimInput + m_nbNeurons,1 + dimInput + m_nbNeurons,CV_64FC1) * m_ridge);

    cv::Mat invCuda, invCV;
    cv::Mat matCudaS,matCudaU,matCudaVT;

    if(m_useCudaInversion)
    {
        swCuda::squareMatrixSingularValueDecomposition(l_mat2inv,matCudaS,matCudaU,matCudaVT);
        l_mat2inv.release();

        displayTime("2 : tikhonovRegularization ", m_oTime, false, m_verbose);

        for(int ii = 0; ii < matCudaS.rows;++ii)
        {
            if(matCudaS.at<double>(ii,ii) > 1e-6)
            {
                matCudaS.at<double>(ii,ii) = 1.0/matCudaS.at<double>(ii,ii);
            }
            else
            {
                matCudaS.at<double>(ii,ii) = 0.0;
            }
        }

        if(m_useCudaMultiplication)
        {
            cv::Mat l_tempCudaMult;
            swCuda::blockMatrixMultiplicationD(matCudaS, matCudaU.t(), l_tempCudaMult, l_subdivisionBlocks);
            matCudaS.release();
            matCudaU.release();
            swCuda::blockMatrixMultiplicationD(matCudaVT.t(), l_tempCudaMult, invCuda, l_subdivisionBlocks);
            matCudaVT.release();
        }
        else
        {
            invCuda = (matCudaVT.t() * matCudaS * matCudaU.t());
        }

        displayTime("3 : tikhonovRegularization ", m_oTime, false, m_verbose);

        if(m_useCudaMultiplication)
        {
            cv::Mat l_tempCudaMult;
            l_xTotReshaped =l_xTotReshaped.t();
//            swCuda::blockMatrixMultiplicationD(l_xTotReshaped.t(), invCuda, l_tempCudaMult, l_subdivisionBlocks);
            swCuda::blockMatrixMultiplicationD(l_xTotReshaped, invCuda, l_tempCudaMult, l_subdivisionBlocks);
            invCuda.release();
            l_xTotReshaped.release();

            cv::Mat l_yTeacherReshaped(yTeacher.size[0] *yTeacher.size[1],  yTeacher.size[2], CV_64FC1);

            #pragma omp parallel for
                for(int ii = 0; ii < yTeacher.size[0]; ++ii)
                {
                    for(int jj = 0; jj < yTeacher.size[1]; ++jj)
                    {
                        for(int kk = 0; kk < yTeacher.size[2]; ++kk)
                        {
                            l_yTeacherReshaped.at<double>(ii*yTeacher.size[1] + jj,kk) = yTeacher.at<double>(ii,jj,kk);
                        }
                    }
                }
            // end pragma

            m_wOut = l_yTeacherReshaped.t() * l_tempCudaMult;
        }
        else
        {
            cv::Mat l_yTeacherReshaped(yTeacher.size[0] *yTeacher.size[1],  yTeacher.size[2], CV_64FC1);
            #pragma omp parallel for
                for(int ii = 0; ii < yTeacher.size[0]; ++ii)
                {
                    for(int jj = 0; jj < yTeacher.size[1]; ++jj)
                    {
                        for(int kk = 0; kk < yTeacher.size[2]; ++kk)
                        {
                            l_yTeacherReshaped.at<double>(ii*yTeacher.size[1] + jj,kk) = yTeacher.at<double>(ii,jj,kk);
                        }
                    }
                }
            // end pragma

            m_wOut = l_yTeacherReshaped.t() * l_xTotReshaped.t() * invCuda;
        }
    }
    else
    {
        cv::invert(l_mat2inv, invCV, cv::DECOMP_SVD);
        l_mat2inv.release();

        displayTime("2-3 : tikhonovRegularization ", m_oTime, false, m_verbose);

        cv::Mat l_yTeacherReshaped(yTeacher.size[0] *yTeacher.size[1],  yTeacher.size[2], CV_64FC1);
        #pragma omp parallel for
            for(int ii = 0; ii < yTeacher.size[0]; ++ii)
            {
                for(int jj = 0; jj < yTeacher.size[1]; ++jj)
                {
                    for(int kk = 0; kk < yTeacher.size[2]; ++kk)
                    {
                        l_yTeacherReshaped.at<double>(ii*yTeacher.size[1] + jj,kk) = yTeacher.at<double>(ii,jj,kk);
                    }
                }
            }
        // end pragma

        m_wOut = (l_yTeacherReshaped.t() * l_xTotReshaped.t()) * invCV;
    }

    displayTime("END : tikhonovRegularization ", m_oTime, false, m_verbose);
}


void Reservoir::train(const cv::Mat &meaningInputTrain, const cv::Mat &teacher, cv::Mat &sentencesOutputTrain, cv::Mat &xTot)
{
    m_oTime = clock();

    displayTime("START : train ", m_oTime, false, m_verbose);

    // generate matrices
        generateMatrixW();
        generateWIn(meaningInputTrain.size[2]);

    displayTime("START : sub train ", m_oTime, false, m_verbose);

        int l_sizeTot[3] = {meaningInputTrain.size[0], 1 + meaningInputTrain.size[2] + m_nbNeurons,  meaningInputTrain.size[1]};
        xTot = cv::Mat (3,l_sizeTot, CV_64FC1, cv::Scalar(0.0)); //  will contain the internal states of the reservoir for all sentences and all timesteps

        cv::Mat l_X2Copy = cv::Mat::zeros(1 + meaningInputTrain.size[2] + m_nbNeurons, meaningInputTrain.size[1], CV_64FC1); // OPTI

        int l_size[1] = {m_w.rows}; // OPTI
        cv::Mat l_xPrev2Copy(1,l_size, CV_64FC1, cv::Scalar(0.0)); // OPTI

        double l_invLeakRate = 1.0 - m_leakRate;

//        fillRandomMat_(xTot);

//        #pragma omp parallel for num_threads(7)
        #pragma omp parallel for
            for(int ii = 0; ii < meaningInputTrain.size[0]; ++ii)
            {
                if(m_verbose)
                {
                    printf("input train : %d / %d\n", ii,meaningInputTrain.size[0]);
                }

                cv::Mat l_X = l_X2Copy.clone();
                cv::Mat l_xPrev, l_x;
                cv::Mat l_subMean(meaningInputTrain.size[1], meaningInputTrain.size[2], CV_64FC1, meaningInputTrain.data + meaningInputTrain.step[0] *ii);

                for(int jj = 0; jj < meaningInputTrain.size[1]; ++jj)
                {
                    // X will contain all the internal states of the reservoir for all timesteps
                    if(jj == 0)
                    {
                        l_xPrev = l_xPrev2Copy;
                    }
                    else
                    {
                        l_xPrev = l_x;
                    }

                    cv::Mat l_u = l_subMean.row(jj);
                    cv::Mat l_temp(l_subMean.cols+1, 1, CV_64FC1);
                    l_temp.at<double>(0) = 1.0;

                    for(int kk = 0; kk < l_subMean.cols; ++kk)
                    {
                        l_temp.at<double>(kk+1) = l_u.at<double>(kk);
                    }

                    cv::Mat l_xTemp = (m_wIn * l_temp)+ (m_w * l_xPrev);

                    cv::MatIterator_<double> it = l_xTemp.begin<double>(), it_end = l_xTemp.end<double>();
                    for(;it != it_end; ++it)
                    {
                        (*it) = tanh(*it);
                    }

                    l_x = (l_xPrev * l_invLeakRate) + (l_xTemp * m_leakRate);

                    cv::Mat l_temp2(l_temp.rows + l_x.rows, 1, CV_64FC1);

                    for(int kk = 0; kk < l_temp.rows + l_x.rows; ++kk)
                    {
                        if(kk < l_temp.rows)
                        {
                            l_temp2.at<double>(kk) = l_temp.at<double>(kk);
                        }
                        else
                        {
                            l_temp2.at<double>(kk) = l_x.at<double>(kk - l_temp.rows);
                        }
                    }

                    l_temp2.copyTo( l_X.col(jj));
                }

                for(int jj = 0; jj < l_X.rows; ++jj)
                {
                    for(int kk = 0; kk < l_X.cols; ++kk)
                    {
                        xTot.at<double>(ii,jj,kk) = l_X.at<double>(jj,kk);
                    }
                }
            }
        // end pragma

        l_X2Copy.release();
        l_xPrev2Copy.release();

        displayTime("END : sub train ", m_oTime, false, m_verbose);

        tikhonovRegularization(xTot, teacher, meaningInputTrain.size[2]);

        sentencesOutputTrain = teacher.clone();
        sentencesOutputTrain.setTo(0.0);

        #pragma omp parallel for
            for(int ii = 0; ii < xTot.size[0]; ++ii)
            {
                cv::Mat l_X = cv::Mat::zeros(xTot.size[1], xTot.size[2], CV_64FC1);

                for(int jj = 0; jj < l_X.rows; ++jj)
                {
                    for(int kk = 0; kk < l_X.cols; ++kk)
                    {
                        l_X.at<double>(jj,kk) = xTot.at<double>(ii,jj,kk);
                    }
                }

                cv::Mat res;

                res = (m_wOut * l_X).t();

                for(int jj = 0; jj < sentencesOutputTrain.size[1]; ++jj)
                {
                    for(int kk = 0; kk < sentencesOutputTrain.size[2]; ++kk)
                    {
                        sentencesOutputTrain.at<double>(ii,jj,kk) = res.at<double>(jj,kk);
                    }
                }
            }
        // end pragma

    displayTime("END : train ", m_oTime, false, m_verbose);

}

void Reservoir::test(const cv::Mat &meaningInputTest, cv::Mat &sentencesOutputTest, cv::Mat &xTot)
{

    m_oTime = clock();

    displayTime("START : test", m_oTime, false, m_verbose);

    int l_sizeTot[3] = {meaningInputTest.size[0], 1 + meaningInputTest.size[2] + m_nbNeurons,  meaningInputTest.size[1]};
    xTot = cv::Mat (3,l_sizeTot, CV_64FC1); //  will contain the internal states of the reservoir for all sentences and all timesteps

    int l_sizeOut[3] = {l_sizeTot[0], l_sizeTot[2], m_wOut.rows};
    sentencesOutputTest = cv::Mat(3, l_sizeOut, CV_64FC1);

    #pragma omp parallel for
        for(int ii = 0; ii < meaningInputTest.size[0]; ++ii)
        {
            cv::Mat l_X = cv::Mat::zeros(1 + meaningInputTest.size[2] + m_nbNeurons, meaningInputTest.size[1],CV_64FC1);
            cv::Mat l_xPrev, l_x;
            cv::Mat l_subMean(meaningInputTest.size[1], meaningInputTest.size[2], CV_64FC1, meaningInputTest.data + meaningInputTest.step[0] *ii);

            for(int jj = 0; jj < meaningInputTest.size[1]; ++jj)
            {
                // X will contain all the internal states of the reservoir for all timesteps
                if(jj == 0)
                {
                    int l_size[1] = {m_w.rows};
                    l_xPrev = cv::Mat(1,l_size, CV_64FC1, cv::Scalar(0.0));
                }
                else
                {
                    l_xPrev = l_x;
                }

                cv::Mat l_u = l_subMean.row(jj);

                cv::Mat l_temp(l_subMean.cols+1, 1, CV_64FC1);
                l_temp.at<double>(0) = 1.0;
                for(int kk = 0; kk < l_subMean.cols; ++kk)
                {
                    l_temp.at<double>(kk+1) = l_u.at<double>(kk);
                }

                cv::Mat l_xTemp = (m_wIn * l_temp) + (m_w * l_xPrev);

                cv::MatIterator_<double> it = l_xTemp.begin<double>(), it_end = l_xTemp.end<double>();
                for(;it != it_end; ++it)
                {
                    (*it) = tanh(*it);
                }

                l_x = (l_xPrev * (1.0 - m_leakRate)) + (l_xTemp * m_leakRate);

                cv::Mat l_temp2(l_temp.rows + l_x.rows, 1, CV_64FC1);
                for(int kk = 0; kk < l_temp.rows + l_x.rows; ++kk)
                {
                    if(kk < l_temp.rows)
                    {
                        l_temp2.at<double>(kk) = l_temp.at<double>(kk);
                    }
                    else
                    {
                        l_temp2.at<double>(kk) = l_x.at<double>(kk - l_temp.rows);
                    }
                }

                l_temp2.copyTo( l_X.col(jj));

                cv::Mat l_temp3(l_x.rows + l_subMean.cols+1, 1, CV_64FC1);
                l_temp3.at<double>(0) = 1.0;
                for(int kk = 0; kk < l_subMean.cols; ++kk)
                {
                    l_temp3.at<double>(kk+1) = l_u.at<double>(kk);
                }
                for(int kk = 0; kk < l_x.rows; ++kk)
                {
                    l_temp3.at<double>(kk+l_subMean.cols+1) = l_x.at<double>(kk);
                }

                cv::Mat l_y = m_wOut * l_temp3;
                for(int kk = 0; kk < sentencesOutputTest.size[2]; ++kk)
                {
                    sentencesOutputTest.at<double>(ii,jj,kk) = l_y.at<double>(kk);
                }
            }
            for(int jj = 0; jj < l_X.rows; ++jj)
            {
                for(int kk = 0; kk < l_X.cols; ++kk)
                {
                    xTot.at<double>(ii,jj,kk) = l_X.at<double>(jj,kk);
                }
            }
        }
    // end omp parallel

     displayTime("END : test", m_oTime, false, m_verbose);
}









// ###################################### TESTS FLOAT

Reservoir::Reservoir(cuint nbNeurons, cfloat spectralRadius, cfloat inputScaling, cfloat leakRate, cfloat sparcity, cfloat ridge, cbool verbose) :
 m_nbNeurons(nbNeurons), m_spectralRadiusF(spectralRadius), m_inputScalingF(inputScaling), m_leakRateF(leakRate), m_ridgeF(ridge), m_verbose(verbose)
{
    if(sparcity > 0.f)
    {
        m_sparcityF = sparcity;
    }
    else
    {
        m_sparcityF = 10.f/m_nbNeurons;
    }
    m_initialized = true;


    cvNamedWindow("reservoir_display", CV_WINDOW_AUTOSIZE | CV_GUI_NORMAL);
    cvMoveWindow("reservoir_display",200,200);
}


void Reservoir::generateMatrixWF()
{
    // debug
    displayTime("START : generate W ", m_oTime, false, m_verbose);

    // init w matrix [N x N]
    m_wF = cv::Mat(m_nbNeurons, m_nbNeurons, CV_32FC1, cv::Scalar(0.f));

    // fill w matrix with random values [-0.5, 0.5]
        for(int ii = 0; ii < m_wF.rows*m_wF.cols;++ii)
        {
            if(static_cast <float> (rand()) / static_cast <float> (RAND_MAX) < m_sparcityF)
            {
                float r = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
                m_wF.at<float>(ii) = (r -0.5f) * m_spectralRadiusF;
            }
        }

    // debug
    displayTime("END : generate W ", m_oTime, false, m_verbose);
}

void Reservoir::generateWInF(cuint dimInput)
{
    // debug
    displayTime("START : generate WIn ", m_oTime, false, m_verbose);

    // init wIn
        m_wInF = cv::Mat(m_nbNeurons, dimInput + 1, CV_32FC1);

    // fill wIn matrix with random values [0, 1]
        cv::MatIterator_<float> it = m_wInF.begin<float>(), it_end = m_wInF.end<float>();
        float l_randMax = static_cast <float> (RAND_MAX);
        for(;it != it_end; ++it)
        {
            (*it) = (static_cast <float> (rand()) / l_randMax) * m_inputScalingF;
        }

    // debug
    displayTime("END : generate WIn ", m_oTime, false, m_verbose);
}



void Reservoir::tikhonovRegularizationF(const cv::Mat &xTot, const cv::Mat &yTeacher, cuint dimInput)
{
    int l_subdivisionBlocks = 2;
    if(m_nbNeurons > 3000)
    {
        l_subdivisionBlocks = 4;
    }
    if(m_nbNeurons > 6000)
    {
        l_subdivisionBlocks = 6;
    }
    if(m_nbNeurons > 8000)
    {
        l_subdivisionBlocks = 8;
    }

    displayTime("START : tikhonovRegularization ", m_oTime, false, m_verbose);

    cv::Mat l_xTotReshaped(xTot.size[1], xTot.size[0] * xTot.size[2], CV_32FC1);

    #pragma omp parallel for
        for(int ii = 0; ii < xTot.size[0]; ++ii)
        {
            for(int jj = 0; jj < xTot.size[1]; ++jj)
            {
                for(int kk = 0; kk < xTot.size[2]; ++kk)
                {
                    l_xTotReshaped.at<float>(jj, ii*xTot.size[2] + kk) = xTot.at<float>(ii,jj,kk);
                }
            }
        }
    // end pragma

    displayTime("1 : tikhonovRegularization ", m_oTime, false, m_verbose);
    cv::Mat l_mat2inv;

    if(m_useCudaInversion)
    {
        swCuda::blockMatrixMultiplicationF(l_xTotReshaped,l_xTotReshaped.t(), l_mat2inv, l_subdivisionBlocks);
    }
    else
    {
        l_mat2inv = (l_xTotReshaped * l_xTotReshaped.t());
    }

    l_mat2inv += (cv::Mat::eye(1 + dimInput + m_nbNeurons,1 + dimInput + m_nbNeurons,CV_32FC1) * m_ridgeF);

    cv::Mat invCuda, invCV;
    cv::Mat matCudaS,matCudaU,matCudaVT;

    if(m_useCudaInversion)
    {
        swCuda::squareMatrixSingularValueDecomposition(l_mat2inv,matCudaS,matCudaU,matCudaVT);
        l_mat2inv.release();

        displayTime("2 : tikhonovRegularization ", m_oTime, false, m_verbose);

        for(int ii = 0; ii < matCudaS.rows;++ii)
        {
            if(matCudaS.at<float>(ii,ii) > 1e-6f)
            {
                matCudaS.at<float>(ii,ii) = 1.f/matCudaS.at<float>(ii,ii);
            }
            else
            {
                matCudaS.at<float>(ii,ii) = 0.f;
            }
        }

        if(m_useCudaMultiplication)
        {
            cv::Mat l_tempCudaMult;
            swCuda::blockMatrixMultiplicationF(matCudaS, matCudaU.t(), l_tempCudaMult, l_subdivisionBlocks);
            matCudaS.release();
            matCudaU.release();
            swCuda::blockMatrixMultiplicationF(matCudaVT.t(), l_tempCudaMult, invCuda, l_subdivisionBlocks);
            matCudaVT.release();
        }
        else
        {
            invCuda = (matCudaVT.t() * matCudaS * matCudaU.t());
        }

        displayTime("3 : tikhonovRegularization ", m_oTime, false, m_verbose);

        if(m_useCudaMultiplication)
        {
            cv::Mat l_tempCudaMult;
            l_xTotReshaped =l_xTotReshaped.t();

            swCuda::blockMatrixMultiplicationF(l_xTotReshaped, invCuda, l_tempCudaMult, l_subdivisionBlocks);
            invCuda.release();
            l_xTotReshaped.release();

            cv::Mat l_yTeacherReshaped(yTeacher.size[0] *yTeacher.size[1],  yTeacher.size[2], CV_32FC1);

            #pragma omp parallel for
                for(int ii = 0; ii < yTeacher.size[0]; ++ii)
                {
                    for(int jj = 0; jj < yTeacher.size[1]; ++jj)
                    {
                        for(int kk = 0; kk < yTeacher.size[2]; ++kk)
                        {
                            l_yTeacherReshaped.at<float>(ii*yTeacher.size[1] + jj,kk) = yTeacher.at<float>(ii,jj,kk);
                        }
                    }
                }
            // end pragma

            m_wOutF = l_yTeacherReshaped.t() * l_tempCudaMult;
        }
        else
        {
            cv::Mat l_yTeacherReshaped(yTeacher.size[0] *yTeacher.size[1],  yTeacher.size[2], CV_32FC1);
            #pragma omp parallel for
                for(int ii = 0; ii < yTeacher.size[0]; ++ii)
                {
                    for(int jj = 0; jj < yTeacher.size[1]; ++jj)
                    {
                        for(int kk = 0; kk < yTeacher.size[2]; ++kk)
                        {
                            l_yTeacherReshaped.at<float>(ii*yTeacher.size[1] + jj,kk) = yTeacher.at<float>(ii,jj,kk);
                        }
                    }
                }
            // end pragma

            m_wOutF = l_yTeacherReshaped.t() * l_xTotReshaped.t() * invCuda;
        }
    }
    else
    {
        cv::invert(l_mat2inv, invCV, cv::DECOMP_SVD);
        l_mat2inv.release();

        displayTime("2-3 : tikhonovRegularization ", m_oTime, false, m_verbose);

        cv::Mat l_yTeacherReshaped(yTeacher.size[0] *yTeacher.size[1],  yTeacher.size[2], CV_32FC1);
        #pragma omp parallel for
            for(int ii = 0; ii < yTeacher.size[0]; ++ii)
            {
                for(int jj = 0; jj < yTeacher.size[1]; ++jj)
                {
                    for(int kk = 0; kk < yTeacher.size[2]; ++kk)
                    {
                        l_yTeacherReshaped.at<float>(ii*yTeacher.size[1] + jj,kk) = yTeacher.at<float>(ii,jj,kk);
                    }
                }
            }
        // end pragma

        m_wOutF = (l_yTeacherReshaped.t() * l_xTotReshaped.t()) * invCV;
    }

    displayTime("END : tikhonovRegularization ", m_oTime, false, m_verbose);
}

void Reservoir::trainF(const cv::Mat &meaningInputTrain, const cv::Mat &teacher, cv::Mat &sentencesOutputTrain, cv::Mat &xTot)
{
    m_oTime = clock();

    displayTime("START : train ", m_oTime, false, m_verbose);

    // generate matrices
        generateMatrixWF();
        generateWInF(meaningInputTrain.size[2]);

    displayTime("START : sub train ", m_oTime, false, m_verbose);

        int l_sizeTot[3] = {meaningInputTrain.size[0], 1 + meaningInputTrain.size[2] + m_nbNeurons,  meaningInputTrain.size[1]};
        xTot = cv::Mat (3,l_sizeTot, CV_32FC1, cv::Scalar(0.f)); //  will contain the internal states of the reservoir for all sentences and all timesteps

        std::cout << "xTot : " << xTot.size[0] << " " << xTot.size[1] << " " << xTot.size[2] << std::endl;

        cv::Mat l_X2Copy = cv::Mat::zeros(1 + meaningInputTrain.size[2] + m_nbNeurons, meaningInputTrain.size[1], CV_32FC1); // OPTI

        int l_size[1] = {m_wF.rows}; // OPTI
        cv::Mat l_xPrev2Copy(1,l_size, CV_32FC1, cv::Scalar(0.f)); // OPTI

        float l_invLeakRate = 1.f - m_leakRateF;

//        fillRandomMat_(xTot);


//        #pragma omp parallel for num_threads(7)
        #pragma omp parallel for
            for(int ii = 0; ii < meaningInputTrain.size[0]; ++ii)
            {
                if(m_verbose)
                {
                    printf("input train : %d / %d\n", ii,meaningInputTrain.size[0]);
                }

                cv::Mat l_X = l_X2Copy.clone();
                cv::Mat l_xPrev, l_x;
                cv::Mat l_subMean(meaningInputTrain.size[1], meaningInputTrain.size[2], CV_32FC1, meaningInputTrain.data + meaningInputTrain.step[0] *ii);

                for(int jj = 0; jj < meaningInputTrain.size[1]; ++jj)
                {
                    // X will contain all the internal states of the reservoir for all timesteps
                    if(jj == 0)
                    {
                        l_xPrev = l_xPrev2Copy;
                    }
                    else
                    {
                        l_xPrev = l_x;
                    }

                    cv::Mat l_u = l_subMean.row(jj);
                    cv::Mat l_temp(l_subMean.cols+1, 1, CV_32FC1);
                    l_temp.at<float>(0) = 1.f;

                    for(int kk = 0; kk < l_subMean.cols; ++kk)
                    {
                        l_temp.at<float>(kk+1) = l_u.at<float>(kk);
                    }

                    cv::Mat l_xTemp = (m_wInF * l_temp)+ (m_wF * l_xPrev);

                    cv::MatIterator_<float> it = l_xTemp.begin<float>(), it_end = l_xTemp.end<float>();
                    for(;it != it_end; ++it)
                    {
                        (*it) = tanh(*it);
                    }

                    l_x = (l_xPrev * l_invLeakRate) + (l_xTemp * m_leakRateF);


//                    cv::Mat display(l_X.rows, l_X.cols, CV_8UC3);
//                    for(int oo = 0; oo < display.rows * display.cols; ++oo)
//                    {
//                        float l_val = l_X.at<float>(oo);
//                        if(l_val < 0)
//                        {
//                            int l_val2 = static_cast<int>(255*l_val);
//                            if(l_val2 > 255)
//                            {
//                                l_val = 255;
//                            }

//                            display.at<cv::Vec3b>(oo) = cv::Vec3b(l_val2,0,122);
//                        }
//                        else
//                        {
//                            int l_val2 =  -static_cast<int>(255*l_val);
//                            if(l_val2 > 255)
//                            {
//                                l_val = 255;
//                            }
//                            display.at<cv::Vec3b>(oo) = cv::Vec3b(0,l_val2,122);
//                        }
//                    }


//                    save2DMatrixToTextStd("../data/display.txt", l_X);
//                    display *= 255;
//                    cv::imshow("reservoir_display", display);
//                    cv::waitKey(5);

                    cv::Mat l_temp2(l_temp.rows + l_x.rows, 1, CV_32FC1);

                    for(int kk = 0; kk < l_temp.rows + l_x.rows; ++kk)
                    {
                        if(kk < l_temp.rows)
                        {
                            l_temp2.at<float>(kk) = l_temp.at<float>(kk);
                        }
                        else
                        {
                            l_temp2.at<float>(kk) = l_x.at<float>(kk - l_temp.rows);
                        }
                    }

                    l_temp2.copyTo( l_X.col(jj));
                }

                for(int jj = 0; jj < l_X.rows; ++jj)
                {
                    for(int kk = 0; kk < l_X.cols; ++kk)
                    {
                        xTot.at<float>(ii,jj,kk) = l_X.at<float>(jj,kk);
                    }
                }
            }
        // end pragma

        l_X2Copy.release();
        l_xPrev2Copy.release();

        displayTime("END : sub train ", m_oTime, false, m_verbose);

        tikhonovRegularizationF(xTot, teacher, meaningInputTrain.size[2]);

        sentencesOutputTrain = teacher.clone();
        sentencesOutputTrain.setTo(0.f);

        #pragma omp parallel for
            for(int ii = 0; ii < xTot.size[0]; ++ii)
            {
                cv::Mat l_X = cv::Mat::zeros(xTot.size[1], xTot.size[2], CV_32FC1);

                for(int jj = 0; jj < l_X.rows; ++jj)
                {
                    for(int kk = 0; kk < l_X.cols; ++kk)
                    {
                        l_X.at<float>(jj,kk) = xTot.at<float>(ii,jj,kk);
                    }
                }

                cv::Mat res;

                res = (m_wOutF * l_X).t();

                for(int jj = 0; jj < sentencesOutputTrain.size[1]; ++jj)
                {
                    for(int kk = 0; kk < sentencesOutputTrain.size[2]; ++kk)
                    {
                        sentencesOutputTrain.at<float>(ii,jj,kk) = res.at<float>(jj,kk);
                    }
                }
            }
        // end pragma

    displayTime("END : train ", m_oTime, false, m_verbose);

}

void Reservoir::testF(const cv::Mat &meaningInputTest, cv::Mat &sentencesOutputTest, cv::Mat &xTot)
{

    m_oTime = clock();

    displayTime("START : test", m_oTime, false, m_verbose);

    int l_sizeTot[3] = {meaningInputTest.size[0], 1 + meaningInputTest.size[2] + m_nbNeurons,  meaningInputTest.size[1]};
    xTot = cv::Mat (3,l_sizeTot, CV_32FC1); //  will contain the internal states of the reservoir for all sentences and all timesteps

    int l_sizeOut[3] = {l_sizeTot[0], l_sizeTot[2], m_wOutF.rows};
    sentencesOutputTest = cv::Mat(3, l_sizeOut, CV_32FC1);

    float l_invLeakRate = 1.f - m_leakRateF;

    #pragma omp parallel for
        for(int ii = 0; ii < meaningInputTest.size[0]; ++ii)
        {
            cv::Mat l_X = cv::Mat::zeros(1 + meaningInputTest.size[2] + m_nbNeurons, meaningInputTest.size[1],CV_32FC1);
            cv::Mat l_xPrev, l_x;
            cv::Mat l_subMean(meaningInputTest.size[1], meaningInputTest.size[2], CV_32FC1, meaningInputTest.data + meaningInputTest.step[0] *ii);

            for(int jj = 0; jj < meaningInputTest.size[1]; ++jj)
            {
                // X will contain all the internal states of the reservoir for all timesteps
                if(jj == 0)
                {
                    int l_size[1] = {m_wF.rows};
                    l_xPrev = cv::Mat(1,l_size, CV_32FC1, cv::Scalar(0.f));
                }
                else
                {
                    l_xPrev = l_x;
                }

                cv::Mat l_u = l_subMean.row(jj);
                cv::Mat l_temp(l_subMean.cols+1, 1, CV_32FC1);
                l_temp.at<float>(0) = 1.f;
                for(int kk = 0; kk < l_subMean.cols; ++kk)
                {
                    l_temp.at<float>(kk+1) = l_u.at<float>(kk);
                }

                cv::Mat l_xTemp = (m_wInF * l_temp) + (m_wF * l_xPrev);

                cv::MatIterator_<float> it = l_xTemp.begin<float>(), it_end = l_xTemp.end<float>();
                for(;it != it_end; ++it)
                {
                    (*it) = tanh(*it);
                }

                l_x = (l_xPrev * l_invLeakRate) + (l_xTemp * m_leakRateF);

                cv::Mat l_temp2(l_temp.rows + l_x.rows, 1, CV_32FC1);
                for(int kk = 0; kk < l_temp.rows + l_x.rows; ++kk)
                {
                    if(kk < l_temp.rows)
                    {
                        l_temp2.at<float>(kk) = l_temp.at<float>(kk);
                    }
                    else
                    {
                        l_temp2.at<float>(kk) = l_x.at<float>(kk - l_temp.rows);
                    }
                }

                l_temp2.copyTo( l_X.col(jj));

                cv::Mat l_temp3(l_x.rows + l_subMean.cols+1, 1, CV_32FC1);
                l_temp3.at<float>(0) = 1.f;
                for(int kk = 0; kk < l_subMean.cols; ++kk)
                {
                    l_temp3.at<float>(kk+1) = l_u.at<float>(kk);
                }
                for(int kk = 0; kk < l_x.rows; ++kk)
                {
                    l_temp3.at<float>(kk+l_subMean.cols+1) = l_x.at<float>(kk);
                }

                cv::Mat l_y = m_wOutF * l_temp3;
                for(int kk = 0; kk < sentencesOutputTest.size[2]; ++kk)
                {
                    sentencesOutputTest.at<float>(ii,jj,kk) = l_y.at<float>(kk);
                }
            }
            for(int jj = 0; jj < l_X.rows; ++jj)
            {
                for(int kk = 0; kk < l_X.cols; ++kk)
                {
                    xTot.at<float>(ii,jj,kk) = l_X.at<float>(jj,kk);
                }
            }
        }
    // end omp parallel

     displayTime("END : test", m_oTime, false, m_verbose);
}

