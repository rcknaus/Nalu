/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level NaluUnit      */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/
#ifndef HighOrderGeometryQuadAdvection_h
#define HighOrderGeometryQuadAdvection_h

#include <element_promotion/operators/CoefficientMatrices.h>
#include <element_promotion/operators/HighOrderOperatorsQuad.h>
#include <element_promotion/operators/DirectionEnums.h>
#include <element_promotion/CVFEMTypeDefs.h>
#include <AlgTraits.h>

#include <stk_util/environment/ReportHandler.hpp>

namespace sierra { namespace nalu { namespace high_order_metrics {

  template <int p> // polynomial order
  void compute_advection_metric_linear(
    const CVFEMOperators<p, stk::topology::QUADRILATERAL_4_2D> ops,
    const nodal_vector_view<AlgTraitsQuad<p>> coord,
    const nodal_vector_view<AlgTraitsQuad<p>> rhou,
    scs_vector_view<AlgTraitsQuad<p>> metric)
  {
    // four edges of the quad
    enum { E_0 = 0, E_1 = 1, E_2 = 2, E_3 = 3 };
    using ftype = typename scs_vector_view<AlgTraitsQuad<p>>::value_type;

    constexpr int numEdges = 4;
    constexpr int nodesPerEdge = 2;

    const ftype edgev[numEdges][nodesPerEdge] = {
         { coord(XH, 0, p) - coord(XH, 0, 0), coord(YH, 0, p) - coord(YH, 0, 0) },
         { coord(XH, p, p) - coord(XH, 0, p), coord(YH, p, p) - coord(YH, 0, p) },
         { coord(XH, p, 0) - coord(XH, p, p), coord(YH, p, 0) - coord(YH, p, p) },
         { coord(XH, 0, 0) - coord(XH, p, 0), coord(YH, 0, 0) - coord(YH, p, 0) },
    };

    const auto scs_interp = ops.mat_.linear_scs_interp;
    const auto nodal_interp = ops.mat_.linear_nodal_interp;

//    nodal_vector_view<AlgTraitsQuad<p>> rhouIp{""};
//    ops.scs_xhat_interp(rhou, rhouIp);

    for (int j = 0; j < p; ++j) {
      const ftype dx_dxh = -scs_interp(0,j) * edgev[E_3][XH] + scs_interp(1,j) * edgev[E_1][XH];
      for (int i = 0; i < p + 1; ++i) {
        const ftype dy_dxh = -nodal_interp(0,i) * edgev[E_3][YH] + nodal_interp(1,i) * edgev[E_1][YH];

//        std::cout << "(" <<  rhou(XH,j,i) << ", " << rhou(YH,j,i) << ")" << std::endl;
        metric(XH,j,i) =  dy_dxh * rhou(XH, j, i) - dx_dxh * rhou(YH, j, i);
      }
    }

//    ops.scs_yhat_interp(rhou, rhouIp);

    for (int j = 0; j < p; ++j) {
      const ftype dy_dyh = scs_interp(0,j) * edgev[E_0][YH] - scs_interp(1,j) * edgev[E_2][YH];
      for (int i = 0; i < p + 1; ++i) {
        const ftype dx_dyh = nodal_interp(0,i) * edgev[E_0][XH] - nodal_interp(1,i) * edgev[E_2][XH];
        metric(YH,j,i) =  -dy_dyh * rhou(XH, j, i) + dx_dyh * rhou(YH, j, i);
      }
    }
  }

  template <int p> // polynomial order
    void compute_advection_metric_linear(
      const CVFEMOperators<p, stk::topology::QUADRILATERAL_4_2D> ops,
      const nodal_vector_view<AlgTraitsQuad<p>> coord,
      const nodal_vector_view<AlgTraitsQuad<p>> rhou,
      const nodal_vector_view<AlgTraitsQuad<p>> Gp_bar,
      const nodal_scalar_view<AlgTraitsQuad<p>> pressure,
      scs_vector_view<AlgTraitsQuad<p>> metric)
    {
      ThrowRequireMsg(false, "Mass-corrected advection metric not implemented yet");
    }


} // namespace HighOrderGeometryQuad
} // namespace naluUnit
} // namespace Sierra

#endif
