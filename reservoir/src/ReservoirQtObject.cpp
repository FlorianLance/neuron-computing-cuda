

/**
 * \file ReservoirQtObject.cpp
 * \brief defines ReservoirQt
 * \author Florian Lance
 * \date 02/12/14
 */

// reservoir-cuda
#include "ReservoirQtObject.h"

#include "../moc/moc_ReservoirQtObject.cpp"

// Qt
#include <QtGui>


#include <omp.h>

using namespace std;

static void fillRandomMat_(cv::Mat &mat)
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

ReservoirQt::ReservoirQt()
{
    m_numThread = omp_get_max_threads( );

    m_initialized = false;
    m_verbose = true;

    m_useCudaInversion      = true;
    m_useCudaMultiplication = false;
    m_sendMatrices = false;
}

void ReservoirQt::setCudaProperties(cbool cudaInv, cbool cudaMult)
{
    m_useCudaInversion = cudaInv;
    m_useCudaMultiplication = cudaMult;
}


// ###################################### TESTS FLOAT

ReservoirQt::ReservoirQt(cuint nbNeurons, cfloat spectralRadius, cfloat inputScaling, cfloat leakRate, cfloat sparcity, cfloat ridge, cbool verbose) :
 m_nbNeurons(nbNeurons), m_spectralRadius(spectralRadius), m_inputScaling(inputScaling), m_leakRate(leakRate), m_ridge(ridge), m_verbose(verbose)
{

    m_numThread = omp_get_max_threads( );
    m_sendMatrices = false;

    if(sparcity > 0.f)
    {
        m_sparcity = sparcity;
    }
    else
    {
        m_sparcity = 10.f/m_nbNeurons;
    }
    m_initialized = true;
}

void ReservoirQt::setParameters(cuint nbNeurons, cfloat spectralRadius, cfloat inputScaling, cfloat leakRate, cfloat sparcity, cfloat ridge, cbool verbose)
{
    m_nbNeurons = nbNeurons;
    m_spectralRadius = spectralRadius;
    m_inputScaling = inputScaling;
    m_leakRate = leakRate;
    m_ridge = ridge;
    m_verbose = verbose;

     if(sparcity > 0.f)
     {
         m_sparcity = sparcity;
     }
     else
     {
         m_sparcity = 10.f/m_nbNeurons;
     }
     m_initialized = true;
}


void ReservoirQt::generateMatrixW()
{
    // debug
    emit sendLogInfo(QString::fromStdString(displayTime("START : generate W ", m_oTime, false, m_verbose)));

    // init w matrix [N x N]
    m_w = cv::Mat(m_nbNeurons, m_nbNeurons, CV_32FC1, cv::Scalar(0.f));

    // fill w matrix with random values [-0.5, 0.5]
        for(int ii = 0; ii < m_w.rows*m_w.cols;++ii)
        {
            if(static_cast <float> (rand()) / static_cast <float> (RAND_MAX) < m_sparcity)
            {
                float r = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
                m_w.at<float>(ii) = (r -0.5f) * m_spectralRadius;
            }
        }

    // debug
    emit sendLogInfo(QString::fromStdString(displayTime("END : generate W ", m_oTime, false, m_verbose)));
}

void ReservoirQt::generateWIn(cuint dimInput)
{
    emit sendLogInfo(QString::fromStdString(displayTime("START : generate WIn ", m_oTime, false, m_verbose)));

    // init wIn
        m_wIn = cv::Mat(m_nbNeurons, dimInput + 1, CV_32FC1);

    // fill wIn matrix with random values [0, 1]
        cv::MatIterator_<float> it = m_wIn.begin<float>(), it_end = m_wIn.end<float>();
        float l_randMax = static_cast <float> (RAND_MAX);
        for(;it != it_end; ++it)
        {
            (*it) = (static_cast <float> (rand()) / l_randMax) * m_inputScaling;
        }

    emit sendLogInfo(QString::fromStdString(displayTime("END : generate WIn ", m_oTime, false, m_verbose)));
}


void ReservoirQt::tikhonovRegularization(const cv::Mat &xTot, const cv::Mat &yTeacher, cuint dimInput)
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

    emit sendLogInfo(QString::fromStdString(displayTime("START : tikhonovRegularization ", m_oTime, false, m_verbose)));

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

    emit sendLogInfo(QString::fromStdString(displayTime("1 : tikhonovRegularization ", m_oTime, false, m_verbose)));
    emit sendComputingState(60, 100, QString("Tikhonov-1"));

    cv::Mat l_mat2inv;

    if(m_useCudaInversion)
    {
        swCuda::blockMatrixMultiplicationF(l_xTotReshaped,l_xTotReshaped.t(), l_mat2inv, l_subdivisionBlocks);
    }
    else
    {
        l_mat2inv = (l_xTotReshaped * l_xTotReshaped.t());
    }

    emit sendComputingState(70, 100, QString("Tikhonov-2"));

    l_mat2inv += (cv::Mat::eye(1 + dimInput + m_nbNeurons,1 + dimInput + m_nbNeurons,CV_32FC1) * m_ridge);

    cv::Mat invCuda, invCV;
    cv::Mat matCudaS,matCudaU,matCudaVT;

    if(m_useCudaInversion)
    {
        if(!swCuda::squareMatrixSingularValueDecomposition(l_mat2inv,matCudaS,matCudaU,matCudaVT))
        {
            std::string l_error("-ERROR : squareMatrixSingularValueDecomposition");
            std::cerr << l_error << std::endl;
            emit sendLogInfo(QString::fromStdString(l_error));
        }
        l_mat2inv.release();

        emit sendLogInfo(QString::fromStdString(displayTime("2 : tikhonovRegularization ", m_oTime, false, m_verbose)));
        emit sendComputingState(80, 100, QString("Tikhonov-3"));

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

        emit sendLogInfo(QString::fromStdString(displayTime("3 : tikhonovRegularization ", m_oTime, false, m_verbose)));
        emit sendComputingState(90, 100, QString("Tikhonov-4"));

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
            m_wOut = l_yTeacherReshaped.t() * l_tempCudaMult;
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

            m_wOut = l_yTeacherReshaped.t() * l_xTotReshaped.t() * invCuda;
        }
    }
    else
    {
        emit sendComputingState(60, 100, QString("Tikhonov-3"));
        cv::invert(l_mat2inv, invCV, cv::DECOMP_SVD);
        l_mat2inv.release();

        emit sendLogInfo(QString::fromStdString(displayTime("2-3 : tikhonovRegularization ", m_oTime, false, m_verbose)));
        emit sendComputingState(90, 100, QString("Tikhonov-4"));


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

        m_wOut = (l_yTeacherReshaped.t() * l_xTotReshaped.t()) * invCV;

    }

    emit sendLogInfo(QString::fromStdString(displayTime("END : tikhonovRegularization ", m_oTime, false, m_verbose)));
}


void ReservoirQt::train(const cv::Mat &meaningInputTrain, const cv::Mat &teacher, cv::Mat &sentencesOutputTrain, cv::Mat &xTot)
{
    // update progress bar
        int l_steps = 0;
        emit sendComputingState(0, meaningInputTrain.size[0]*2, QString("Build X"));

    // init time
        m_oTime = clock();

    emit sendLogInfo(QString::fromStdString(displayTime("START : train ", m_oTime, false, m_verbose)));

    // generate matrices
        generateMatrixW();
        generateWIn(meaningInputTrain.size[2]);

    emit sendLogInfo(QString::fromStdString(displayTime("START : sub train ", m_oTime, false, m_verbose)));

    // init x tot
        int l_sizeTot[3] = {meaningInputTrain.size[0], 1 + meaningInputTrain.size[2] + m_nbNeurons,  meaningInputTrain.size[1]};
        xTot = cv::Mat (3,l_sizeTot, CV_32FC1, cv::Scalar(0.f)); //  will contain the internal states of the reservoir for all sentences and all timesteps

    // init x
        cv::Mat l_X2Copy = cv::Mat::zeros(1 + meaningInputTrain.size[2] + m_nbNeurons, meaningInputTrain.size[1], CV_32FC1);

    // init x prev
        int l_size[1] = {m_w.rows};
        cv::Mat l_xPrev2Copy(1,l_size, CV_32FC1, cv::Scalar(0.f));

    float l_invLeakRate = 1.f - m_leakRate;

    // mutex for openmp threads
        QMutex l_lockerMainThread;

    // info for generating the real time plot
        int l_numberCurve = 10; // TODO  : parameterize
        int l_lengthCurve = meaningInputTrain.size[0] * meaningInputTrain.size[1];
        emit sendInfoPlot(l_numberCurve, l_lengthCurve, QString("train"));

        #pragma omp parallel for num_threads(m_numThread)
            for(int ii = 0; ii < meaningInputTrain.size[0]; ++ii)
            {
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

                    cv::Mat l_xTemp = (m_wIn * l_temp)+ (m_w * l_xPrev);

                    cv::MatIterator_<float> it = l_xTemp.begin<float>(), it_end = l_xTemp.end<float>();
                    for(;it != it_end; ++it)
                    {
                        (*it) = tanh(*it);
                    }

                    l_x = (l_xPrev * l_invLeakRate) + (l_xTemp * m_leakRate);


                    cv::Mat display(l_X.rows, l_X.cols, CV_8UC3);
                    for(int oo = 0; oo < display.rows * display.cols; ++oo)
                    {
                        float l_val = l_X.at<float>(oo);
                        if(l_val < 0)
                        {
                            int l_val2 = static_cast<int>(255*l_val);
                            if(l_val2 > 255)
                            {
                                l_val = 255;
                            }

                            display.at<cv::Vec3b>(oo) = cv::Vec3b(l_val2,0,122);
                        }
                        else
                        {
                            int l_val2 =  -static_cast<int>(255*l_val);
                            if(l_val2 > 255)
                            {
                                l_val = 255;
                            }
                            display.at<cv::Vec3b>(oo) = cv::Vec3b(0,l_val2,122);
                        }
                    }

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


                l_lockerMainThread.lock();
                    emit sendComputingState(++l_steps, meaningInputTrain.size[0]*2, QString("Build X"));
                l_lockerMainThread.unlock();

                // send data only if the multi thread is disabled
                if(m_sendMatrices)
                {
                    if(m_numThread == 1)
                    {
                        cv::Mat l_rgb2Display(100,l_X.cols, CV_8UC3);
                        int l_currentCol = 0;
                        QVector<QVector<double> > l_values;

                        for(int jj = 100; jj < 200; ++jj) // l_X.rows
                        {
                            QVector<double> l_line;
                            for(int kk = 0; kk < l_X.cols; ++kk)
                            {
                                l_rgb2Display.at<cv::Vec3b>(l_currentCol,kk) = cv::Vec3b(255*l_X.at<float>(jj,kk), 255*l_X.at<float>(jj,kk),255*l_X.at<float>(jj,kk));
                                l_line.push_back(l_X.at<float>(jj,kk));
                            }
                            ++l_currentCol;

                            l_values.push_back(l_line);
                        }

                        QImage l_image2Send = mat2QImage(l_rgb2Display);

                        l_image2Send = l_image2Send.scaled(500, 500,Qt::KeepAspectRatio);

                        emit sendMatriceImage2Display(l_image2Send);
                        emit sendMatriceData(l_values);
                    }
                }
            }
        // end pragma

        // clean inused matrices
            l_X2Copy.release();
            l_xPrev2Copy.release();

        emit sendLogInfo(QString::fromStdString(displayTime("END : sub train ", m_oTime, false, m_verbose)));

        emit sendComputingState(50, 100, QString("Tychonov-start"));
        tikhonovRegularization(xTot, teacher, meaningInputTrain.size[2]);
        emit sendComputingState(95, 100, QString("Tychonov-end"));

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

                res = (m_wOut * l_X).t();

                for(int jj = 0; jj < sentencesOutputTrain.size[1]; ++jj)
                {
                    for(int kk = 0; kk < sentencesOutputTrain.size[2]; ++kk)
                    {
                        sentencesOutputTrain.at<float>(ii,jj,kk) = res.at<float>(jj,kk);
                    }
                }
            }
        // end pragma

    emit sendLogInfo(QString::fromStdString(displayTime("END : train ", m_oTime, false, m_verbose)));
    emit sendComputingState(100, 100, QString("End training"));
}


void ReservoirQt::test(const cv::Mat &meaningInputTest, cv::Mat &sentencesOutputTest, cv::Mat &xTot)
{
    // update progress bar
        int l_steps = 0;
        emit sendComputingState(0, meaningInputTest.size[0], QString("Build X"));

    // init time
        m_oTime = clock();

    // mutex for openmp threads
        QMutex l_lockerMainThread;

    emit sendLogInfo(QString::fromStdString(displayTime("START : test", m_oTime, false, m_verbose)));

    // init x tot
        int l_sizeTot[3] = {meaningInputTest.size[0], 1 + meaningInputTest.size[2] + m_nbNeurons,  meaningInputTest.size[1]};
        xTot = cv::Mat (3,l_sizeTot, CV_32FC1); //  will contain the internal states of the reservoir for all sentences and all timesteps

    // init sentences output
        int l_sizeOut[3] = {l_sizeTot[0], l_sizeTot[2], m_wOut.rows};
        sentencesOutputTest = cv::Mat(3, l_sizeOut, CV_32FC1);

    float l_invLeakRate = 1.f - m_leakRate;

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
                    int l_size[1] = {m_w.rows};
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

                cv::Mat l_xTemp = (m_wIn * l_temp) + (m_w * l_xPrev);

                cv::MatIterator_<float> it = l_xTemp.begin<float>(), it_end = l_xTemp.end<float>();
                for(;it != it_end; ++it)
                {
                    (*it) = tanh(*it);
                }

                l_x = (l_xPrev * l_invLeakRate) + (l_xTemp * m_leakRate);

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

                cv::Mat l_y = m_wOut * l_temp3;
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

            l_lockerMainThread.lock();
                emit sendComputingState(++l_steps, meaningInputTest.size[0], QString("Build X"));
            l_lockerMainThread.unlock();

        }
    // end omp parallel

    emit sendLogInfo(QString::fromStdString(displayTime("END : test", m_oTime, false, m_verbose)));
    emit sendComputingState(100, 100, QString("End test"));
}

void ReservoirQt::saveTraining(const string &path)
{
    save2DMatrixToTextStd(path + "/wOut.txt", m_wOut);
    save2DMatrixToTextStd(path + "/wIn.txt", m_wIn);
    save2DMatrixToTextStd(path + "/w.txt", m_w);
}

void ReservoirQt::loadTraining(const string &path)
{
    load2DMatrixStd<float>(path + "/wOut.txt", m_wOutLoaded);
    load2DMatrixStd<float>(path + "/wIn.txt", m_wInLoaded);
    load2DMatrixStd<float>(path + "/w.txt", m_wLoaded);
}

void ReservoirQt::updateMatricesWithLoadedTraining()
{
    if(m_wOutLoaded.rows > 0)
    {
        m_wOut = m_wOutLoaded.clone();
        m_wIn  = m_wInLoaded.clone();
        m_w    = m_wLoaded.clone();
    }
    else
    {
        std::string l_error("-ERROR : updateMatricesWithLoadedTraining, no matrices loaded, can not update training matrices. ");
        std::cerr << l_error << std::endl;
        emit sendLogInfo(QString::fromStdString(l_error));
    }
}

void ReservoirQt::disableMaxOmpThreadNumber(bool disable)
{
    if(!disable)
    {
        m_numThread = omp_get_max_threads( );
    }
    else
    {
        m_numThread = 1;
    }
}

void ReservoirQt::enableDisplay(bool enable)
{
    m_sendMatrices = enable;
}












//    double min, max;
//    cv::Mat l_subRes(m_wOut.rows, m_wOut.cols, CV_32FC3);
//    cv::minMaxLoc(m_wOut, &min, &max);
//    save3Channel2DMatrixToTextStd("../data/Results/mat_out/img_before_" + l_os1.str() + ".txt", l_subRes);


//    save2DMatrixToTextStd("../data/Results/Wout.txt", m_wOut);
//    save2DMatrixToTextStd("../data/Results/WoutT.txt", m_wOut.t());


//    for(int ii = 0; ii < m_wOut.rows * m_wOut.cols; ++ii)
//    {
//        l_subRes.at<cv::Vec3f>(ii) = cv::Vec3f(-m_wOut.at<float>(ii),-m_wOut.at<float>(ii),-m_wOut.at<float>(ii));

//        cv::Vec3f l_value = l_subRes.at<cv::Vec3f>(ii);
//        l_value[0] *= 255./max;
//        l_value[1] *= 255./max;
//        l_value[2] *= 255./max;

//        l_subRes.at<cv::Vec3f>(ii) = l_value;
//    }

//    save3Channel2DMatrixToTextStd("../data/Results/aaaa.txt", l_subRes);

//    cv::imwrite("../data/Results/a.png", l_subRes);
//    cv::cvtColor(l_subRes, l_subRes, CV_BGR2GRAY );
//    cv::imwrite("../data/Results/b.png", l_subRes);

////    cv::imshow("reservoir_display", l_subRes);
//    cv::waitKey(5);

//    std::cout << "m_wOut : " << m_wOut.rows << " " <<  m_wOut.cols << " | " << std::endl;
