
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
 * \file Reservoir.cpp
 * \brief defines Reservoir
 * \author Florian Lance
 * \date 02/12/14
 */

// reservoir-cuda
#include "Reservoir.h"

#include "../moc/moc_Reservoir.cpp"

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

Reservoir::Reservoir()
{
    m_numThread = omp_get_max_threads( );

    m_initialized = false;
    m_verbose = true;

    m_useCudaInversion      = true;
    m_useCudaMultiplication = false;
    m_sendMatrices = true;

    m_useW   = false;
    m_useWIn = false;

    m_stopLoop = false;

}

void Reservoir::setCudaProperties(cbool cudaInv, cbool cudaMult)
{
    m_useCudaInversion = cudaInv;
    m_useCudaMultiplication = cudaMult;
}


// ###################################### TESTS FLOAT

Reservoir::Reservoir(cuint nbNeurons, cfloat spectralRadius, cfloat inputScaling, cfloat leakRate, cfloat sparcity, cfloat ridge, cbool verbose) :
 m_nbNeurons(nbNeurons), m_spectralRadius(spectralRadius), m_inputScaling(inputScaling), m_leakRate(leakRate), m_ridge(ridge), m_verbose(verbose)
{

    m_numThread = omp_get_max_threads( );
    m_sendMatrices = true;

    if(sparcity > 0.f)
    {
        m_sparcity = sparcity;
    }
    else
    {
        m_sparcity = 10.f/m_nbNeurons;
    }
    m_initialized = true;

    m_useW   = false;
    m_useWIn = false;

}

void Reservoir::setParameters(cuint nbNeurons, cfloat spectralRadius, cfloat inputScaling, cfloat leakRate, cfloat sparcity, cfloat ridge, cbool verbose)
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


void Reservoir::generateMatrixW()
{
    if(!m_useW)
    {
        emit sendLogInfo(QString::fromStdString(displayTime("START : generate W ", m_oTime, false, m_verbose)), QColor(Qt::black));

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

        emit sendLogInfo(QString::fromStdString(displayTime("END : generate W ", m_oTime, false, m_verbose)), QColor(Qt::black));
    }
    else
    {
        m_w = m_wLoaded.clone();
    }
}

void Reservoir::generateWIn(cuint dimInput)
{
    if(!m_useWIn)
    {
        emit sendLogInfo(QString::fromStdString(displayTime("START : generate WIn ", m_oTime, false, m_verbose)), QColor(Qt::black));

        // init wIn
            m_wIn = cv::Mat(m_nbNeurons, dimInput + 1, CV_32FC1);

        // fill wIn matrix with random values [0, 1]
            cv::MatIterator_<float> it = m_wIn.begin<float>(), it_end = m_wIn.end<float>();
            float l_randMax = static_cast <float> (RAND_MAX);
            for(;it != it_end; ++it)
            {
                (*it) = (static_cast <float> (rand()) / l_randMax) * m_inputScaling;
            }

        emit sendLogInfo(QString::fromStdString(displayTime("END : generate WIn ", m_oTime, false, m_verbose)), QColor(Qt::black));
    }
    else
    {
        m_wIn = m_wInLoaded.clone();
    }
}

bool Reservoir::train(const cv::Mat &meaningInputTrain, const cv::Mat &teacher, cv::Mat &sentencesOutputTrain, cv::Mat &xTot)
{
    // update progress bar
        int l_steps = 0;
        emit sendComputingState(0, meaningInputTrain.size[0]*2, QString("Build X"));

    // init time
        m_oTime = clock();

    emit sendLogInfo(QString::fromStdString(displayTime("START : train ", m_oTime, false, m_verbose)), QColor(Qt::black));

    // generate matrices
        generateMatrixW();
        generateWIn(meaningInputTrain.size[2]);

    // check if loaded w and loaded wIn have the same dimension 0
        if(m_useW && m_useWIn)
        {
            if(m_w.rows != m_wIn.rows)
            {
                emit sendLogInfo("Loaded w and loaded wIn don't use the same number of neurons : W -> " + QString::number(m_w.rows) +" wIn -> " + QString::number(m_wIn.rows) + "\n", QColor(Qt::red));
                emit sendComputingState(0, 100, QString("Error with loaded matrices."));
                return false;
            }
        }
        else if(m_useW)
        {
            if(m_w.rows != m_nbNeurons)
            {
                emit sendLogInfo("Loaded w number of neurons is different from the current Neurons number : W -> " + QString::number(m_w.rows) +" N -> " + QString::number(m_nbNeurons) + "\n", QColor(Qt::red));
                emit sendComputingState(0, 100, QString("Error with loaded matrices."));
                return false;
            }
        }
        else if(m_useWIn)
        {
            if(m_wIn.rows != m_nbNeurons)
            {
                emit sendLogInfo("Loaded wIn number of neurons is different from the current Neurons number : W -> " + QString::number(m_wIn.rows) +" N -> " + QString::number(m_nbNeurons) + "\n", QColor(Qt::red));
                emit sendComputingState(0, 100, QString("Error with loaded matrices."));
                return false;
            }
        }

    emit sendLogInfo(QString::fromStdString(displayTime("START : sub train ", m_oTime, false, m_verbose)), QColor(Qt::black));

    // init x tot
        int l_sizeTot[3] = {meaningInputTrain.size[0], 1 + meaningInputTrain.size[2] + m_nbNeurons,  meaningInputTrain.size[1]};
        xTot = cv::Mat (3,l_sizeTot, CV_32FC1, cv::Scalar(0.f)); //  will contain the internal states of the reservoir for all sentences and all timesteps

    // init x
        cv::Mat l_X2Copy = cv::Mat::zeros(1 + meaningInputTrain.size[2] + m_nbNeurons, meaningInputTrain.size[1], CV_32FC1);

    // init x prev
        int l_size[1] = {m_w.rows};
        cv::Mat l_xPrev2Copy(1,l_size, CV_32FC1, cv::Scalar(0.f));

    float l_invLeakRate = 1.f - m_leakRate;

//m_stopLoop = true;

    #pragma omp parallel for num_threads(m_numThread)
        for(int ii = 0; ii < meaningInputTrain.size[0]; ++ii)
        {
            m_stopLocker.lockForRead();
                bool l_espaceLoop = m_stopLoop;
            m_stopLocker.unlock();

            if(l_espaceLoop)
            {
                continue;
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

            emit sendComputingState(++l_steps, meaningInputTrain.size[0]*2, QString("Build X"));
        }
    // end pragma

    if(m_stopLoop)
    {
        emit sendLogInfo("Stop X construction loop.\n", QColor(Qt::red));
        emit sendComputingState(0, 100, QString("Aborted."));
        m_stopLoop = false;
        return false;
    }


    // clean inused matrices
        l_X2Copy.release();
        l_xPrev2Copy.release();

    emit sendLogInfo(QString::fromStdString(displayTime("END : sub train ", m_oTime, false, m_verbose)), QColor(Qt::black));

    emit sendComputingState(50, 100, QString("Tychonov-start"));
    if(!tikhonovRegularization(xTot, teacher, meaningInputTrain.size[2]))
    {
        emit sendLogInfo("Stop tikhonovRegularization.\n", QColor(Qt::red));
        emit sendComputingState(0, 100, QString("Aborted."));
        m_stopLoop = false;
        return false;
    }
    emit sendComputingState(95, 100, QString("Tychonov-end"));

    sentencesOutputTrain = teacher.clone();
    sentencesOutputTrain.setTo(0.f);


    #pragma omp parallel for
        for(int ii = 0; ii < xTot.size[0]; ++ii)
        {
            m_stopLocker.lockForRead();
                bool l_espaceLoop = m_stopLoop;
            m_stopLocker.unlock();

            if(l_espaceLoop)
            {
                continue;
            }


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

    if(m_stopLoop)
    {
        emit sendLogInfo("Stop sentencesOutputTrain construction loop.\n", QColor(Qt::red));
        emit sendComputingState(0, 100, QString("Aborted."));
        m_stopLoop = false;
        return false;
    }

    emit sendLogInfo(QString::fromStdString(displayTime("END : train ", m_oTime, false, m_verbose)), QColor(Qt::black));
    emit sendComputingState(100, 100, QString("End training"));    

    return true;
}


void Reservoir::test(const cv::Mat &meaningInputTest, cv::Mat &sentencesOutputTest, cv::Mat &xTot)
{
    // update progress bar
        int l_steps = 0;
        emit sendComputingState(0, meaningInputTest.size[0], QString("Build X"));

    // init time
        m_oTime = clock();

    // mutex for openmp threads
        QMutex l_lockerMainThread;

    emit sendLogInfo(QString::fromStdString(displayTime("START : test", m_oTime, false, m_verbose)), QColor(Qt::black));

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

    emit sendLogInfo(QString::fromStdString(displayTime("END : test", m_oTime, false, m_verbose)), QColor(Qt::black));
    emit sendComputingState(100, 100, QString("End test"));
}

bool Reservoir::checkStop()
{
    m_stopLocker.lockForRead();
        bool l_stopLoop = m_stopLoop;
    m_stopLocker.unlock();

    if(l_stopLoop)
    {
        return false;
    }

    return true;
}


bool Reservoir::tikhonovRegularization(const cv::Mat &xTot, const cv::Mat &yTeacher, cuint dimInput)
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

    emit sendLogInfo(QString::fromStdString(displayTime("START : tikhonovRegularization ", m_oTime, false, m_verbose)), QColor(Qt::black));

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

    if(!checkStop())
    {
        return false;
    }


    emit sendLogInfo(QString::fromStdString(displayTime("1 : tikhonovRegularization ", m_oTime, false, m_verbose)), QColor(Qt::black));
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

    if(!checkStop())
    {
        return false;
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
            emit sendLogInfo(QString::fromStdString(l_error), QColor(Qt::red));
            return false;
        }
        l_mat2inv.release();

        emit sendLogInfo(QString::fromStdString(displayTime("2 : tikhonovRegularization ", m_oTime, false, m_verbose)), QColor(Qt::black));
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


        if(!checkStop())
        {
            return false;
        }

        emit sendLogInfo(QString::fromStdString(displayTime("3 : tikhonovRegularization ", m_oTime, false, m_verbose)), QColor(Qt::black));
        emit sendComputingState(90, 100, QString("Tikhonov-4"));

        if(m_useCudaMultiplication)
        {
            cv::Mat l_tempCudaMult;
            l_xTotReshaped =l_xTotReshaped.t();

            swCuda::blockMatrixMultiplicationF(l_xTotReshaped, invCuda, l_tempCudaMult, l_subdivisionBlocks);
            invCuda.release();
            l_xTotReshaped.release();

            if(!checkStop())
            {
                return false;
            }

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

            if(!checkStop())
            {
                return false;
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

            if(!checkStop())
            {
                return false;
            }

            m_wOut = l_yTeacherReshaped.t() * l_xTotReshaped.t() * invCuda;
        }
    }
    else
    {
        emit sendComputingState(60, 100, QString("Tikhonov-3"));

        cv::invert(l_mat2inv, invCV, cv::DECOMP_SVD);
        l_mat2inv.release();

        emit sendLogInfo(QString::fromStdString(displayTime("2-3 : tikhonovRegularization ", m_oTime, false, m_verbose)), QColor(Qt::black));
        emit sendComputingState(90, 100, QString("Tikhonov-4"));

        if(!checkStop())
        {
            return false;
        }

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

        if(!checkStop())
        {
            return false;
        }

        m_wOut = (l_yTeacherReshaped.t() * l_xTotReshaped.t()) * invCV;

    }

    if(!checkStop())
    {
        return false;
    }

    emit sendLogInfo(QString::fromStdString(displayTime("END : tikhonovRegularization ", m_oTime, false, m_verbose)), QColor(Qt::black));
    return true;
}

void Reservoir::saveParamFile(const std::string &path)
{
    QFile l_paramFile(QString::fromStdString(path) + "/param.txt");
    if(l_paramFile.open(QIODevice::WriteOnly | QIODevice::Text))
    {
        QVector<QString> l_parameters;
        l_parameters << QString::number(m_nbNeurons) << QString::number(m_sparcity) << QString::number(m_spectralRadius) <<
                        QString::number(m_inputScaling) << QString::number(m_leakRate) << QString::number(m_ridge);
        QTextStream l_stream(&l_paramFile);

        for(int ii = 0; ii < l_parameters.size(); ++ii)
        {
            l_stream << l_parameters[ii];
            if(ii < l_parameters.size()-1)
            {
                l_stream << " ";
            }
        }
    }
}

void Reservoir::saveWIn(const std::string &path)
{
    save2DMatrixToTextStd(path + "/wIn.txt", m_wIn);
}

void Reservoir::saveW(const std::string &path)
{
    save2DMatrixToTextStd(path + "/w.txt", m_w);
}

void Reservoir::loadParam(const std::string &path)
{
    QFile l_paramFile(QString::fromStdString(path) + "/param.txt");
    if(l_paramFile.open(QIODevice::ReadOnly | QIODevice::Text))
    {
        QStringList parameters;
        QTextStream l_stream(&l_paramFile);
        QString l_content = l_stream.readAll();
        parameters = l_content.split(' ');
        sendLoadedWParameters(parameters);
    }
}


void Reservoir::saveTraining(const std::string &path)
{
    save2DMatrixToTextStd(path + "/wOut.txt", m_wOut);
    save2DMatrixToTextStd(path + "/wIn.txt", m_wIn);
    save2DMatrixToTextStd(path + "/w.txt", m_w);    
    saveParamFile(path);
}

void Reservoir::loadTraining(const std::string &path)
{
    load2DMatrixStd<float>(path + "/wOut.txt", m_wOutLoaded);
    load2DMatrixStd<float>(path + "/wIn.txt", m_wInLoaded);
    load2DMatrixStd<float>(path + "/w.txt", m_wLoaded);
    loadParam(path);
}

void Reservoir::loadW(const std::string &path)
{
    load2DMatrixStd<float>(path, m_wLoaded);
}

void Reservoir::loadWIn(const std::string &path)
{
    load2DMatrixStd<float>(path, m_wInLoaded);
}


void Reservoir::updateMatricesWithLoadedTraining()
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
        emit sendLogInfo(QString::fromStdString(l_error), QColor(Qt::red));
    }
}

void Reservoir::setMatricesUse(cbool useCustomW, cbool useCustomWIn)
{
    m_useW   = useCustomW;
    m_useWIn = useCustomWIn;
}



void Reservoir::enableMaxOmpThreadNumber(bool enable)
{
    if(enable)
    {
        m_numThread = omp_get_max_threads( );
    }
    else
    {
        m_numThread = 1;
    }
}

void Reservoir::stopLoop()
{
    m_stopLocker.lockForWrite();
        m_stopLoop = true;
    m_stopLocker.unlock();
}
