/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#include <user_functions/GaussianPulseMixFracAuxFunction.h>
#include <algorithm>

// basic c++
#include <cmath>
#include <vector>
#include <stdexcept>

namespace sierra{
namespace nalu{

GaussianPulseMixFracAuxFunction::GaussianPulseMixFracAuxFunction() :
  AuxFunction(0,1),
  aX_(0.1),
  tX_(1.0),
  yTr_(1.0),
  dTr_(0.20),
  pi_(acos(-1.0))
{
  // does nothing
}

void
GaussianPulseMixFracAuxFunction::do_evaluate(
  const double *coords,
  const double /*time*/,
  const unsigned spatialDimension,
  const unsigned numPoints,
  double * fieldPtr,
  const unsigned fieldSize,
  const unsigned /*beginPos*/,
  const unsigned /*endPos*/) const
{
  double xc = 0;
  double yc = 0;
  double rad = 1.0;

  for(unsigned p=0; p < numPoints; ++p) {

    const double x = (coords[0] - xc)/rad;
    const double y = (coords[1] - yc)/rad;

    double rsq = x*x+y*y;
    fieldPtr[0] = std::exp(-rsq)/std::exp(0.);

    fieldPtr += fieldSize;
    coords += spatialDimension;
  }
}

} // namespace nalu
} // namespace Sierra
