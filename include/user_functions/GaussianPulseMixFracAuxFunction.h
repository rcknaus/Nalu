/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef GaussianPulseMixFracAuxFunction_h
#define GaussianPulseMixFracAuxFunction_h

#include <AuxFunction.h>

#include <vector>

namespace sierra{
namespace nalu{

class GaussianPulseMixFracAuxFunction : public AuxFunction
{
public:

  GaussianPulseMixFracAuxFunction();

  virtual ~GaussianPulseMixFracAuxFunction() {}
  
  virtual void do_evaluate(
    const double * coords,
    const double time,
    const unsigned spatialDimension,
    const unsigned numPoints,
    double * fieldPtr,
    const unsigned fieldSize,
    const unsigned beginPos,
    const unsigned endPos) const;
  
private:
  const double aX_;
  const double tX_;
  const double yTr_;
  const double dTr_;
  const double pi_;
};

} // namespace nalu
} // namespace Sierra

#endif
