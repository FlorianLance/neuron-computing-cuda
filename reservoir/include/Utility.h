
/**
 * \file Utility.h
 * \brief defines utility functions for manipuling strings and matrices.
 * \author Florian Lance
 * \date 01/10/14
 */

#ifndef UTILITY_H
#define UTILITY_H

// std
#include <iostream>
#include <fstream>
#include <iomanip>

// qt
#include <QtGui>

// opencv
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

// others
#include <time.h>

/**
 * @brief Sentence
 */
typedef std::vector<std::string> Sentence;

/**
 * @brief Sentences
 */
typedef std::vector<Sentence> Sentences;

template<typename T>
/**
 * @brief tanhMan
 * @param value
 * @return
 */
inline T tanhMan(T value)
{
    T e = exp(-2 * value);
    return (1 - e)/(1 + e);
}

/**
 * @brief displaySentence
 * @param sentence
 */
static void displaySentence(const Sentence &sentence)
{
    for(uint ii = 0; ii < static_cast<uint>(sentence.size()); ++ii)
    {
        std::cout << sentence[ii] << " ";
    }
    std::cout << std::endl;
}

/**
 * @brief displaySentence
 * @param sentences
 */
static void displaySentence(const Sentences &sentences)
{
    for(uint ii = 0; ii < static_cast<uint>(sentences.size()); ++ii)
    {
        displaySentence(sentences[ii]);
    }
}

template<typename T>
/**
 * @brief initMatrix
 * @param mat
 * @param rows
 * @param cols
 * @param initWithZeros
 */
static void initMatrix(cv::Mat &mat, const int rows, const int cols, bool initWithZeros = false)
{
    if(typeid(T) == typeid(float))
    {
        if(initWithZeros)
        {
            mat = cv::Mat(rows, cols, CV_32FC1, cv::Scalar(0.f));
        }
        else
        {
            mat = cv::Mat(rows, cols, CV_32FC1);
        }
    }
    else if(typeid(T) == typeid(double))
    {
        if(initWithZeros)
        {
            mat = cv::Mat(rows, cols, CV_64FC1, cv::Scalar(0.0));
        }
        else
        {
            mat = cv::Mat(rows, cols, CV_64FC1);
        }
    }
    else
    {
        std::cerr << "-ERROR : initMatrix -> type not managed. " << std::endl;
    }
}

template<typename T>
/**
 * @brief initMatrix
 * @param mat
 * @param dim1
 * @param dim2
 * @param dim3
 * @param initWithZeros
 */
static void initMatrix(cv::Mat &mat, const int dim1, const int dim2, const int dim3, bool initWithZeros = false)
{
    int l_size[3] = {dim1,dim2,dim3};

    if(typeid(T) == typeid(float))
    {
        if(initWithZeros)
        {
            mat = cv::Mat(3, l_size, CV_32FC1, cv::Scalar(0.f));
        }
        else
        {
            mat = cv::Mat(3, l_size, CV_32FC1);
        }
    }
    else if(typeid(T) == typeid(double))
    {
        if(initWithZeros)
        {
            mat = cv::Mat(3, l_size, CV_64FC1, cv::Scalar(0.0));
        }
        else
        {
            mat = cv::Mat(3, l_size, CV_64FC1);
        }
    }
    else
    {
        std::cerr << "-ERROR : initMatrix -> type not managed. " << std::endl;
    }
}



template<typename T>
/**
 * @brief initMatrix
 * @param mat
 * @param numDim
 * @param sizes
 * @param initWithZeros
 */
static void initMatrix(cv::Mat &mat, const int numDim, int *sizes, bool initWithZeros = false)
{
    if(typeid(T) == typeid(float))
    {
        if(initWithZeros)
        {
            mat = cv::Mat(numDim, sizes, CV_32FC1, cv::Scalar(0.f));
        }
        else
        {
            mat = cv::Mat(numDim, sizes, CV_32FC1);
        }
    }
    else if(typeid(T) == typeid(double))
    {
        if(initWithZeros)
        {
            mat = cv::Mat(numDim, sizes, CV_64FC1, cv::Scalar(0.0));
        }
        else
        {
            mat = cv::Mat(numDim, sizes, CV_64FC1);
        }
    }
    else
    {
        std::cerr << "-ERROR : initMatrix -> type not managed. " << std::endl;
    }
}

/**
 * @brief displayTime
 * @param info
 * @param time
 * @param skipNextLine
 * @param verboseAcivated
 */
static void displayTime(const std::string info, const clock_t time, const bool skipNextLine = false, const bool verboseAcivated = true)
{
    if(!verboseAcivated)
    {
        return;
    }

    std::ostringstream l_oss;
    l_oss << ((float)(clock() - time) / CLOCKS_PER_SEC);

    std::cout << "  [TIME] " << info << " : " << l_oss.str() << std::endl;

    if(skipNextLine)
    {
        std::cout << std::endl;
    }
}

/**
 * @brief convQt2DString2Std2DString
 * @param qt2DArray
 * @param std2DArray
 */
static void convQt2DString2Std2DString(const QVector<QStringList> &qt2DArray, std::vector<std::vector<std::string> > &std2DArray)
{
    std2DArray.clear();

    for(uint ii = 0; ii < static_cast<uint>(qt2DArray.size()); ++ii)
    {
        std::vector<std::string> l_array;

        for(uint jj = 0; jj < static_cast<uint>(qt2DArray[ii].size()); ++jj)
        {

            l_array.push_back(qt2DArray[ii][jj].toStdString());
        }

        std2DArray.push_back(l_array);
    }
}

/**
 * @brief displayDenseMatrix
 * @param oMat
 */
static void displayDenseMatrix(const cv::Mat &oMat)
{
    int l_dim = oMat.dims;
    bool l_32b = false;
    if(oMat.depth() == CV_32FC1)
    {
        l_32b = true;
    }

    if(l_dim == 2)
    {
        for(int ii = 0; ii < oMat.rows; ++ii)
        {
            for(int jj = 0; jj < oMat.cols; ++jj)
            {
                if(l_32b)
                {
                    std::cout << oMat.at<float>(ii,jj) << " ";
                }
                else
                {
                    std::cout << oMat.at<double>(ii,jj) << " ";
                }
            }
            std::cout << std::endl;
        }
    }
    else if(l_dim == 3)
    {
        for(int ii = 0; ii < oMat.size[0]; ++ii)
        {
            std::cout << "New slice : " << std::endl;

            for(int jj = 0; jj < oMat.size[1]; ++jj)
            {
                for(int kk = 0; kk < oMat.size[2]; ++kk)
                {
                    if(l_32b)
                    {
                        std::cout << oMat.at<float>(ii,jj,kk) << " ";
                    }
                    else
                    {
                        std::cout << oMat.at<double>(ii,jj,kk) << " ";
                    }
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }
    }
}

/**
 * @brief save1DMatrixToText
 * @param pathFile
 * @param mat1D
 */
static void save1DMatrixToText(const QString &pathFile, const cv::Mat &mat1D)
{
    bool l_32b = false;
    if(mat1D.depth() == CV_32FC1)
    {
        l_32b = true;
    }

    QFile l_file(pathFile);
    if(l_file.open(QIODevice::WriteOnly | QIODevice::Text))
    {
        QTextStream out(&l_file);

        if(l_32b)
        {
            for(int ii = 0; ii < mat1D.size[0]; ++ii)
            {
                out << mat1D.at<float>(ii) << " ";
            }
        }
        else
        {
            for(int ii = 0; ii < mat1D.size[0]; ++ii)
            {
                out << mat1D.at<double>(ii) << " ";
            }
        }

    }
    else
    {
        std::cerr << "Can not write 1D matrix in file. " << std::endl;
    }
}


/**
 * @brief save2DStringArrayToTextStd
 * @param pathFile
 * @param strin2DArray
 */
static void save2DStringArrayToTextStd(const std::string &pathFile, const std::vector<std::vector<std::string> >  &strin2DArray)
{
    std::ofstream l_oFlowFile(pathFile);

    if(l_oFlowFile)
    {
        for(int ii = 0; ii < strin2DArray.size(); ++ii)
        {
            for(int jj = 0; jj < strin2DArray[ii].size(); ++jj)
            {
                l_oFlowFile << strin2DArray[ii][jj] << " ";
            }
            l_oFlowFile << "\n";
        }
    }
    else
    {
        std::cerr << "Can not write 2D string array in file. " << std::endl;
    }
}

template<typename T>
/**
 * @brief load2DMatrixStd
 * @param pathFile
 * @param mat2D
 */
static void load2DMatrixStd(const std::string &pathFile, cv::Mat &mat2D)
{
    std::ifstream  l_fileStream(pathFile);
    std::vector<std::vector<T> > l_2DArray;

    bool l_endFile = false;
    if (l_fileStream.is_open())
    {
        while(!l_endFile)
        {
            std::vector<T> l_1DArray;

            std::string l_line;
            std::getline(l_fileStream, l_line);
            std::stringstream l_stringStream(l_line);

            if(l_line.size() > 0)
            {
                bool l_endLine = false;
                while(!l_endLine)
                {
                    std::string l_string;
                    l_stringStream >> l_string;

                    if(l_string.size() == 0)
                    {
                        l_endLine = true;
                    }
                    else
                    {
                        std::istringstream l_buffer(l_string);

                        T l_value;
                        l_buffer >> l_value;
                        l_1DArray.push_back(l_value);
                    }
                }

                l_2DArray.push_back(l_1DArray);
            }
            else
            {
                l_endFile = true;
            }
        }
    }
    else
    {
        std::cerr << "-ERROR : load2DMatrixStd -> file cannot be opened. " << std::endl;
    }

    if(l_2DArray.size() == 0)
    {
        std::cerr << "- ERROR : load2DMatrixStd -> empty matrix. " << std::endl;
        return;
    }

    if(l_2DArray[0].size() == 0)
    {
        std::cerr << "- ERROR : load2DMatrixStd -> 1D matrix. " << std::endl;
        return;
    }

    cv::Size l_size2DMatrix(static_cast<int>(l_2DArray[0].size()), static_cast<int>(l_2DArray.size()));

    if(typeid(T) == typeid(float))
    {
        mat2D = cv::Mat(l_size2DMatrix, CV_32FC1);
    }
    else if(typeid(T) == typeid(double))
    {
        mat2D = cv::Mat(l_size2DMatrix, CV_64FC1);
    }
    else if(typeid(T) == typeid(int))
    {
        mat2D = cv::Mat(l_size2DMatrix, CV_32SC1);
    }
    else
    {
        std::cerr << "- ERROR : load2DMatrixStd -> type not managed. " << std::endl;
        return;
    }

    for(int ii = 0; ii < l_2DArray.size(); ++ii)
    {
        for(int jj = 0; jj < l_2DArray[0].size(); ++jj)
        {
            mat2D.at<T>(ii,jj) = l_2DArray[ii][jj];
        }
    }
}


/**
 * @brief save2DMatrixToTextStd
 * @param pathFile
 * @param mat2D
 */
static void save2DMatrixToTextStd(const std::string &pathFile, const cv::Mat &mat2D)
{
    std::ofstream l_oFlowFile(pathFile);

    // check depth input data
        bool l_32b = false;
        if(mat2D.depth() == CV_32FC1)
        {
            l_32b = true;
        }

    if(l_oFlowFile)
    {
        if(l_32b)
        {
            for(int ii = 0; ii < mat2D.size[0]; ++ii)
            {
                for(int jj = 0; jj < mat2D.size[1]; ++jj)
                {
                    std::ostringstream l_osV1;
                    l_osV1.precision(15);
                    l_osV1 << mat2D.at<float>(ii,jj) << " ";
                    l_oFlowFile << l_osV1.str();
                }

                l_oFlowFile << "\n";
            }
        }
        else
        {
            for(int ii = 0; ii < mat2D.size[0]; ++ii)
            {
                for(int jj = 0; jj < mat2D.size[1]; ++jj)
                {
                    std::ostringstream l_osV1;
                    l_osV1.precision(15);
                    l_osV1 << mat2D.at<double>(ii,jj) << " ";
                    l_oFlowFile << l_osV1.str();
                }

                l_oFlowFile << "\n";
            }
        }
    }
    else
    {
        std::cerr << "Can not write 3D matrix in file. " << std::endl;
    }
}


/**
 * @brief save3DMatrixToTextStd
 * @param pathFile
 * @param mat3D
 */
static void save3DMatrixToTextStd(const std::string &pathFile, const cv::Mat &mat3D)
{
    std::ofstream l_oFlowFile(pathFile);

    // check depth input data
        bool l_32b = false;
        if(mat3D.depth() == CV_32FC1)
        {
            l_32b = true;
        }


    if(l_oFlowFile)
    {
        if(l_32b)
        {
            for(int ii = 0; ii < mat3D.size[0]; ++ii)
            {
                l_oFlowFile << "New slice : \n";

                for(int jj = 0; jj < mat3D.size[1]; ++jj)
                {
                    for(int kk = 0; kk < mat3D.size[2]; ++kk)
                    {
                         std::ostringstream l_osV1;
                         l_osV1 << mat3D.at<float>(ii,jj,kk) << " ";
                         l_oFlowFile << l_osV1.str();
                    }
                    l_oFlowFile << "\n";
                }
            }
        }
        else
        {
            for(int ii = 0; ii < mat3D.size[0]; ++ii)
            {
                l_oFlowFile << "New slice : \n";

                for(int jj = 0; jj < mat3D.size[1]; ++jj)
                {
                    for(int kk = 0; kk < mat3D.size[2]; ++kk)
                    {
                         std::ostringstream l_osV1;
                         l_osV1 << mat3D.at<double>(ii,jj,kk) << " ";
                         l_oFlowFile << l_osV1.str();
                    }
                    l_oFlowFile << "\n";
                }
            }
        }
    }
    else
    {
        std::cerr << "Can not write 3D matrix in file. " << std::endl;
    }
}


/**
 * @brief save3DMatrixToText
 * @param pathFile
 * @param mat3D
 */
static void save3DMatrixToText(const QString &pathFile, const cv::Mat &mat3D)
{
    // check depth input data
        bool l_32b = false;
        if(mat3D.depth() == CV_32FC1)
        {
            l_32b = true;
        }


    QFile l_file(pathFile);
    if(l_file.open(QIODevice::WriteOnly | QIODevice::Text))
    {
        QTextStream out(&l_file);

        if(l_32b)
        {
            for(int ii = 0; ii < mat3D.size[0]; ++ii)
            {
                out << "New slice : \n";
                for(int jj = 0; jj < mat3D.size[1]; ++jj)
                {
                    for(int kk = 0; kk < mat3D.size[2]; ++kk)
                    {
                        out << mat3D.at<float>(ii,jj,kk) << " ";
                    }
                    out << "\n";
                }
            }
        }
        else
        {
            for(int ii = 0; ii < mat3D.size[0]; ++ii)
            {
                out << "New slice : \n";
                for(int jj = 0; jj < mat3D.size[1]; ++jj)
                {
                    for(int kk = 0; kk < mat3D.size[2]; ++kk)
                    {
                        out << mat3D.at<double>(ii,jj,kk) << " ";
                    }
                    out << "\n";
                }
            }
        }
    }
    else
    {
        std::cerr << "Can not write 3D matrix in file. " << std::endl;
    }
}

/**
 * @brief load3DMatrixFromNpPythonSaveText
 * @param pathFile
 * @param mat3D
 */
static void load3DMatrixFromNpPythonSaveText(const QString &pathFile, std::vector<cv::Mat> &mat3D)
{
    QFile l_file(pathFile);

    if(l_file.open(QIODevice::ReadOnly | QIODevice::Text))
    {
        QTextStream in(&l_file), inLine;
        in.setRealNumberPrecision(15);
        inLine.setRealNumberPrecision(15);

        int l_sizesMat[3];

        QString l_separator, l_size;
        in >> l_separator;
        in >> l_size;
        l_sizesMat[0] = l_size.toInt();
        in >> l_size;
        l_sizesMat[1] = l_size.toInt();
        in >> l_size;
        l_sizesMat[2] = l_size.toInt();

        QString l_line = in.readLine();
        l_separator ="";

        mat3D = std::vector<cv::Mat>(l_sizesMat[0], cv::Mat(l_sizesMat[1],l_sizesMat[2], CV_64FC1));

        int ii = -1, jj = 0, kk = 0;


        while (!in.atEnd())
        {
            QString l_line = in.readLine();

            if(l_line[0] == '#')
            {
                ++ii;
                jj = 0;

                continue;
            }
            else if(l_line[0] == '\0')
            {
                break;
            }

            inLine.setString(&l_line);

            while(!inLine.atEnd())
            {
                QString l_value;
                inLine >> l_value;

                if(l_value[0] == '\0')
                {
                    continue;
                }

                mat3D[ii].at<double>(jj,kk) = l_value.toDouble();

                ++kk;
            }

            ++jj;
            kk = 0;
        }
    }
    else
    {
        std::cerr << "Can not open python 3D file. " << std::endl;
    }
}

/**
 * @brief load3DMatrixFromNpPythonSaveText
 * @param pathFile
 * @param mat3D
 */
static void load3DMatrixFromNpPythonSaveText(const QString &pathFile, cv::Mat &mat3D)
{
    QFile l_file(pathFile);

    if(l_file.open(QIODevice::ReadOnly | QIODevice::Text))
    {
        QTextStream in(&l_file), inLine;
        in.setRealNumberPrecision(15);
        inLine.setRealNumberPrecision(15);

        int l_sizesMat[3];

        QString l_separator, l_size;
        in >> l_separator;
        in >> l_size;
        l_sizesMat[0] = l_size.toInt();
        in >> l_size;
        l_sizesMat[1] = l_size.toInt();
        in >> l_size;
        l_sizesMat[2] = l_size.toInt();

        QString l_line = in.readLine();
        l_separator ="";

        mat3D = cv::Mat(3, l_sizesMat, CV_64FC1);

        int ii = -1, jj = 0, kk = 0;


        while (!in.atEnd())
        {
            QString l_line = in.readLine();

            if(l_line[0] == '#')
            {
                ++ii;
                jj = 0;

                continue;
            }
            else if(l_line[0] == '\0')
            {
                break;
            }

            inLine.setString(&l_line);

            while(!inLine.atEnd())
            {
                QString l_value;
                inLine >> l_value;

                if(l_value[0] == '\0')
                {
                    continue;
                }

                mat3D.at<double>(ii,jj,kk) = l_value.toDouble();
                ++kk;
            }

            ++jj;
            kk = 0;
        }
    }
    else
    {
        std::cerr << "Can not open python 3D file. " << std::endl;
    }
}


/**
 * @brief load3DMatrixFromNpPythonSaveText
 * @param pathFile
 * @param mat3D
 */
static void load3DMatrixFromNpPythonSaveTextF(const QString &pathFile, cv::Mat &mat3D)
{
    QFile l_file(pathFile);

    if(l_file.open(QIODevice::ReadOnly | QIODevice::Text))
    {
        QTextStream in(&l_file), inLine;
        in.setRealNumberPrecision(15);
        inLine.setRealNumberPrecision(15);

        int l_sizesMat[3];

        QString l_separator, l_size;
        in >> l_separator;
        in >> l_size;
        l_sizesMat[0] = l_size.toInt();
        in >> l_size;
        l_sizesMat[1] = l_size.toInt();
        in >> l_size;
        l_sizesMat[2] = l_size.toInt();

        QString l_line = in.readLine();
        l_separator ="";

        mat3D = cv::Mat(3, l_sizesMat, CV_32FC1);

        int ii = -1, jj = 0, kk = 0;


        while (!in.atEnd())
        {
            QString l_line = in.readLine();

            if(l_line[0] == '#')
            {
                ++ii;
                jj = 0;

                continue;
            }
            else if(l_line[0] == '\0')
            {
                break;
            }

            inLine.setString(&l_line);

            while(!inLine.atEnd())
            {
                QString l_value;
                inLine >> l_value;

                if(l_value[0] == '\0')
                {
                    continue;
                }

                mat3D.at<float>(ii,jj,kk) = l_value.toFloat();
                ++kk;
            }

            ++jj;
            kk = 0;
        }
    }
    else
    {
        std::cerr << "Can not open python 3D file. " << std::endl;
    }
}

/**
 * @brief load3DMatrixFromNpPythonSaveText
 * @param pathFile
 * @param mat3D
 */
static void load3DMatrixFromNpPythonSaveTextF(const QString &pathFile, std::vector<cv::Mat> &mat3D)
{
    QFile l_file(pathFile);

    if(l_file.open(QIODevice::ReadOnly | QIODevice::Text))
    {
        QTextStream in(&l_file), inLine;
        in.setRealNumberPrecision(15);
        inLine.setRealNumberPrecision(15);

        int l_sizesMat[3];

        QString l_separator, l_size;
        in >> l_separator;
        in >> l_size;
        l_sizesMat[0] = l_size.toInt();
        in >> l_size;
        l_sizesMat[1] = l_size.toInt();
        in >> l_size;
        l_sizesMat[2] = l_size.toInt();

        QString l_line = in.readLine();
        l_separator ="";

        mat3D = std::vector<cv::Mat>(l_sizesMat[0], cv::Mat(l_sizesMat[1],l_sizesMat[2], CV_32FC1));

        int ii = -1, jj = 0, kk = 0;


        while (!in.atEnd())
        {
            QString l_line = in.readLine();

            if(l_line[0] == '#')
            {
                ++ii;
                jj = 0;

                continue;
            }
            else if(l_line[0] == '\0')
            {
                break;
            }

            inLine.setString(&l_line);

            while(!inLine.atEnd())
            {
                QString l_value;
                inLine >> l_value;

                if(l_value[0] == '\0')
                {
                    continue;
                }

                mat3D[ii].at<float>(jj,kk) = l_value.toFloat();

                ++kk;
            }

            ++jj;
            kk = 0;
        }
    }
    else
    {
        std::cerr << "Can not open python 3D file. " << std::endl;
    }
}


template<typename T>
/**
 * @brief compareMatrices
 * @param mat1
 * @param mat2
 * @param nbValuesDiff
 * @param precision
 * @return
 */
static int compareMatrices(const cv::Mat &mat1, const cv::Mat &mat2, int &nbValuesDiff, int precision)
{
    if(mat1.rows != mat2.rows || mat1.cols != mat2.cols || mat1.depth() != mat2.depth())
    {
        std::cerr << "Error : comparaeMAtrices, bad input. " << std::endl;
        return -1;
    }

    nbValuesDiff = 0;

    for(int ii = 0; ii < mat1.rows*mat1.cols; ++ii)
    {
        std::stringstream ss1,ss2;
        ss1 << std::setprecision(10) << std::fixed << mat1.at<T>(ii);
        ss2 << std::setprecision(10) << std::fixed << mat2.at<T>(ii);
        std::string l_num1Str = ss1.str();
        std::string l_num2Str = ss2.str();

        if(l_num1Str.size() > precision)
        {
            l_num1Str.erase(precision, l_num1Str.size()-precision);
        }
        if(l_num2Str.size() > precision)
        {
            l_num2Str.erase(precision, l_num2Str.size()-precision);
        }

        if(l_num1Str != l_num2Str)
        {
            ++nbValuesDiff;
        }
    }

    return 0;
}


#endif // UTILITY_H
