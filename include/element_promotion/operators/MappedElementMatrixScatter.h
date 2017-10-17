/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level NaluUnit      */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/
#ifndef MappedElementMatrixScatter_h
#define MappedElementMatrixScatter_h

#include <element_promotion/operators/HighOrderOperatorsQuad.h>
#include <element_promotion/operators/CoefficientMatrices.h>
#include <element_promotion/CVFEMTypeDefs.h>

// todo(rcknaus): consider removing the need for this

namespace sierra {
namespace nalu {
namespace tensor_assembly {

  template <int p, typename LHSViewType, typename RHSViewType>
  void mapped_scatter(
    node_map_view<AlgTraitsQuad<p>>& node_map,
    const matrix_view<AlgTraitsQuad<p>>& mapped_lhs,
    const nodal_scalar_view<AlgTraitsQuad<p>>& mapped_rhs,
    LHSViewType& lhs,
    RHSViewType& rhs)
  {
    // directly modify the suminto?  probably move this to the assemblelemsolver at least

    constexpr int npe = AlgTraitsQuad<p>::nodesPerElement_;
    constexpr int n1D = AlgTraitsQuad<p>::nodes1D_;

    for (int j = 0; j < npe; ++j) {
      for (int i = 0; i < npe; ++i) {
        lhs(node_map(j), node_map(i)) += mapped_lhs(j,i);
      }
    }

    for (int j = 0; j < n1D; ++j) {
      for (int i = 0; i < n1D; ++i) {
        rhs(node_map(j*n1D +i)) += mapped_rhs(j,i);
      }
    }
  }

  template <int p,typename RHSViewType>
  void mapped_scatter(
    node_map_view<AlgTraitsQuad<p>>& node_map,
    const nodal_scalar_view<AlgTraitsQuad<p>>& mapped_rhs,
    RHSViewType& rhs)
  {
    constexpr int n1D = AlgTraitsQuad<p>::nodes1D_;
    for (int j = 0; j < n1D; ++j) {
      for (int i = 0; i < n1D; ++i) {
        rhs(node_map(j*n1D +i)) += mapped_rhs(j,i);
      }
    }
  }

} // namespace TensorAssembly
} // namespace naluUnit
} // namespace Sierra

#endif
