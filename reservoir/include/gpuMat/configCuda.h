
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
 * \file configCuda.h
 * \brief defines CUDA configuration values
 * \author Florian Lance
 * \date 27/09/13
 */

#ifndef _CONFIGCUDA_
#define _CONFIGCUDA_

typedef unsigned int uint;          /**< typedef for unsigned int*/
typedef const unsigned int cuint;   /**< typedef for const unsigned int */
typedef const bool cbool;           /**< typedef for const bool */
typedef const int cint;             /**< typedef for const int */
typedef const float cfloat;         /**< typedef for const float */
typedef const double cdouble;       /**< typedef for const double */

#define BLOCKSIZE 16            /**< BLOCKSIZE macro used by CUDA */

/**
 * @brief A matrix of float
 * data stored in row-major order:
 * M(row, col) = *(M.elements + row * M.width + col)
 */
struct Matrix
{
  int width;        /**< width of the matrix */
  int height;       /**< height of the matrix */
  float* elements;  /**< data */
  int stride;       /**< stride */
};

/**
 * @brief A matrix of double
 * data stored in row-major order:
 * M(row, col) = *(M.elements + row * M.width + col)
 */
struct MatrixD{
    int width;        /**< width of the matrix */
    int height;       /**< height of the matrix */
    double* elements;  /**< data */
    int stride;       /**< stride */
};

#endif
