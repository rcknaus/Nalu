/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef Channel2DVelocityAuxFunction_h
#define Channel2DVelocityAuxFunction_h

#include <AuxFunction.h>

#include <vector>

namespace sierra{
namespace nalu{

class Channel2DVelocityAuxFunction : public AuxFunction
{
public:

  Channel2DVelocityAuxFunction(
    const unsigned beginPos,
    const unsigned endPos);

  virtual ~Channel2DVelocityAuxFunction() {}
  
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
  double u_m;
  double Ly_top;
  double Ly_bottom;
};

} // namespace nalu
} // namespace Sierra

#endif
