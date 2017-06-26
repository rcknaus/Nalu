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

//  template <int p, typename FlatViewType>
//  void mapped_gather(CVFEMOperatorsQuad<p> ops, FlatViewType vin,  nodal_scalar_view<AlgTraitsQuad<p>> vout)
//  {
//    static_assert(Kokkos::rank(vin) == 1u, "");
//
//    for (int j = 0; j < AlgTraitsQuad<p>::nodes1D_; ++j) {
//      for (int i = 0; i < AlgTraitsQuad<p>::nodes1D_; ++i) {
//        vout(j,i) = vin(ops.node_map_(j*AlgTraitsQuad<p>::nodes1D_+i));
//      }
//    }
//  }
//
//  template <int p, typename TwoDFlatViewType>
//  void mappe(
//    node_map_view<AlgTraitsQuad<p>> node_map,
//    TwoDFlatViewType vin,
//    nodal_vector_view<AlgTraitsQuad<p>> vout)
//  {
//    static_assert(Kokkos::rank(vin) == 2, "");
//
//    for (int j = 0; j < AlgTraitsQuad<p>::nodes1D_; ++j) {
//      for (int i = 0; i < AlgTraitsQuad<p>::nodes1D_; ++i) {
//        auto nodeId = node_map(j*AlgTraitsQuad<p>::nodes1D_+i);
//        for (int d = 0; d < AlgTraitsQuad<p>::nDim_; ++d) {
//          vout(d,j,i) = vin(nodeId,d);
//        }
//      }
//    }
//  }

  template <int p, typename LHSViewType, typename RHSViewType>
  void mapped_scatter(
    node_map_view<AlgTraitsQuad<p>> node_map,
    const matrix_view<AlgTraitsQuad<p>> mapped_lhs,
    const nodal_scalar_view<AlgTraitsQuad<p>> mapped_rhs,
    LHSViewType lhs,
    RHSViewType rhs)
  {
    constexpr int npe = AlgTraitsQuad<p>::nodesPerElement_;
    constexpr int n1D = AlgTraitsQuad<p>::nodes1D_;

    for (int j = 0; j < npe; ++j) {
      for (int i = 0; i < npe; ++i) {
        ThrowAssert(std::isfinite(mapped_lhs(j,i)));
        lhs(node_map(j), node_map(i)) += mapped_lhs(j,i);
      }
    }

    for (int j = 0; j < n1D; ++j) {
      for (int i = 0; i < n1D; ++i) {
        ThrowAssert(std::isfinite(mapped_rhs(j,i)));
        rhs(node_map(j*n1D +i)) += mapped_rhs(j,i);
      }
    }
  }

} // namespace TensorAssembly
} // namespace naluUnit
} // namespace Sierra

#endif
