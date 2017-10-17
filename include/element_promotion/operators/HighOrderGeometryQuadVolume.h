/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level NaluUnit      */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/
#ifndef HighOrderGeometryQuadVolume_h
#define HighOrderGeometryQuadVolume_h

#include <element_promotion/operators/CoefficientMatrices.h>
#include <element_promotion/operators/DirectionEnums.h>
#include <element_promotion/CVFEMTypeDefs.h>
#include <AlgTraits.h>

#include <stk_util/environment/ReportHandler.hpp>

namespace sierra {
namespace nalu {
namespace high_order_metrics
{
  template <int p, typename Scalar>
  void compute_volume_metric_linear(
    const CVFEMOperators<p, Scalar, stk::topology::QUAD_4_2D>& ops,
    const nodal_vector_view<AlgTraitsQuad<p>, Scalar>& coord,
    nodal_scalar_view<AlgTraitsQuad<p>, Scalar>& vol)
  {
    enum { E_0 = 0, E_1 = 1, E_2 = 2, E_3 = 3 };
    const Scalar edgev[4][2] = {
         { coord(XH, 0, p) - coord(XH, 0, 0), coord(YH, 0, p) - coord(YH, 0, 0) },
         { coord(XH, p, p) - coord(XH, 0, p), coord(YH, p, p) - coord(YH, 0, p) },
         { coord(XH, p, 0) - coord(XH, p, p), coord(YH, p, 0) - coord(YH, p, p) },
         { coord(XH, 0, 0) - coord(XH, p, 0), coord(YH, 0, 0) - coord(YH, p, 0) },
    };

    const auto& linear_nodal_interp = ops.mat_.linear_nodal_interp;

    for (int j = 0; j < p + 1; ++j) {
      // lift top-bottom edges
      const Scalar dx_dyh = linear_nodal_interp(0,j) * edgev[E_0][XH] - linear_nodal_interp(1,j) * edgev[E_2][XH];
      const Scalar dy_dyh = linear_nodal_interp(0,j) * edgev[E_0][YH] - linear_nodal_interp(1,j) * edgev[E_2][YH];

      for (int i = 0; i < p + 1; ++i) {
        //lift left-right edges
        const Scalar dx_dxh = -linear_nodal_interp(0,i) * edgev[E_1][XH] + linear_nodal_interp(1,i) * edgev[E_3][XH];
        const Scalar dy_dxh = -linear_nodal_interp(0,i) * edgev[E_1][YH] + linear_nodal_interp(1,i) * edgev[E_3][YH];

        // times divided by 1/4 for missing factor of a half in the derivatives
        vol(j,i) = 0.25 * (dx_dxh * dy_dyh - dx_dyh * dy_dxh);
      }
    }
  }
} // namespace HighOrderGeometryQuad
} // namespace naluUnit
} // namespace Sierra

#endif
