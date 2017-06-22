/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#include <element_promotion/computational_kernels/ScalarAdvDiffHOElemKernel.h>

#include <element_promotion/operators/HighOrderAdvectionDiffusionQuad.h>
#include <element_promotion/operators/HighOrderGeometryQuadDiffusion.h>
#include <element_promotion/operators/CoefficientMatrices.h>
#include <element_promotion/operators/DirectionEnums.h>
#include <element_promotion/ElementDescription.h>

#include <Kernel.h>
#include <SolutionOptions.h>
#include <FieldTypeDef.h>
#include <Realm.h>

// template and scratch space
#include <BuildTemplates.h>
#include <ScratchViews.h>

// stk_mesh/base/fem
#include <stk_mesh/base/Entity.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/Field.hpp>

// topology
#include <stk_topology/topology.hpp>

// Kokkos
#include <Kokkos_Core.hpp>

namespace sierra{
namespace nalu{

template<class AlgTraits>
ScalarAdvDiffHOElemKernel<AlgTraits>::ScalarAdvDiffHOElemKernel(
  const stk::mesh::BulkData& bulkData,
  SolutionOptions& solnOpts,
  ScalarFieldType *scalarQ,
  ScalarFieldType *diffFluxCoeff,
  const ElementDescription& desc,
  ElemDataRequests& dataPreReqs)
  : Kernel(),
    scalarQ_(scalarQ),
    diffFluxCoeff_(diffFluxCoeff)
{
  // save off fields
  const stk::mesh::MetaData& meta_data = bulkData.mesh_meta_data();
  coordinates_ = meta_data.get_field<VectorFieldType>(stk::topology::NODE_RANK, solnOpts.get_coordinates_name());
  massFlowRate_ = meta_data.get_field<GenericFieldType>(stk::topology::ELEMENT_RANK, "mass_flow_rate_scs");

  // map from "exodus-style" node ordering to usual tensor-product based node ordering.
  for (int j = 0; j < AlgTraits::nodes1D_; ++j) {
    for (int i = 0; i < AlgTraits::nodes1D_; ++i) {
      v_node_map_(j*AlgTraits::nodes1D_+i) = desc.node_map(i,j);
    }
  }

  // only necessary for correctly sizing scratch views
  dataPreReqs.add_cvfem_surface_me(get_surface_master_element(AlgTraits::topo_));
  dataPreReqs.add_element_field(*massFlowRate_, AlgTraits::numScsIp_);

  dataPreReqs.add_gathered_nodal_field(*coordinates_, AlgTraits::nDim_);
  dataPreReqs.add_gathered_nodal_field(*scalarQ, 1);
  dataPreReqs.add_gathered_nodal_field(*diffFluxCoeff, 1);
}
//--------------------------------------------------------------------------
template<class AlgTraits> void
ScalarAdvDiffHOElemKernel<AlgTraits>::execute(
  SharedMemView<double **>& lhs,
  SharedMemView<double *>& rhs,
  ScratchViews& scratchViews)
{
  SharedMemView<double**> v_flatCoords = scratchViews.get_scratch_view_2D(*coordinates_);
  SharedMemView<double*> v_flatScalar = scratchViews.get_scratch_view_1D(*scalarQ_);
  SharedMemView<double*> v_flatDiff  = scratchViews.get_scratch_view_1D(*diffFluxCoeff_);
  SharedMemView<double*> v_flatMdot = scratchViews.get_scratch_view_1D(*massFlowRate_);

  // reorder fields into the ordering expected by the alg
  for (int j = 0; j < AlgTraits::nodes1D_; ++j) {
    for (int i = 0; i < AlgTraits::nodes1D_; ++i) {
      int nodeId = v_node_map_(j*AlgTraits::nodes1D_+i);
      v_scalar_(j,i) = v_flatScalar(nodeId);
      v_diff_(j,i)   = v_flatDiff(nodeId);
      for (int d = 0; d < AlgTraits::nDim_; ++d) {
        v_coords_(d,j,i) = v_flatCoords(nodeId, d);
      }
    }
  }

  for (int p = 0; p < AlgTraits::nscs_; ++p) {
    for (int n = 0; n < AlgTraits::nodes1D_; ++n) {
      v_mdot_(0, p, n) = v_flatMdot(p*AlgTraits::nodes1D_ + n);
      v_mdot_(1, p, n) = v_flatMdot(p*AlgTraits::nodes1D_ + n + AlgTraits::nodes1D_ * AlgTraits::nscs_);
    }
  }

  Kokkos::deep_copy(v_lhs_, 0.0);
  Kokkos::deep_copy(v_rhs_, 0.0);

  // todo(rcknaus): specialized, high order mdot calculation
  // for now, just use default mdot calc
  high_order_metrics::compute_diffusion_metric_linear(ops_, v_coords_, v_diff_, v_diff_metric_);
  tensor_assembly::elemental_advection_diffusion_jacobian(ops_, v_mdot_, v_diff_metric_, v_lhs_);

  //  tensor_assembly::elemental_advection_diffusion_action(ops_, v_diff_metric_, v_scalar_, v_rhs_);
  Teuchos::BLAS<int, typename matrix_view<AlgTraits>::value_type>().GEMV(
    Teuchos::TRANS, // row v column
    AlgTraits::nodesPerElement_, AlgTraits::nodesPerElement_,
    -1.0,
    v_lhs_.ptr_on_device(), AlgTraits::nodesPerElement_,
    v_lhs_.ptr_on_device(), 1,
    +1.0,
    v_rhs_.ptr_on_device(), 1
  );

  // map lhs/rhs back to the usual ordering
  for (int j = 0; j < AlgTraits::nodesPerElement_; ++j) {
    for (int i = 0; i < AlgTraits::nodesPerElement_; ++i) {
      lhs(v_node_map_(j), v_node_map_(i)) += v_lhs_(j,i);
    }
  }

  for (int j = 0; j < AlgTraits::nodes1D_; ++j) {
    for (int i = 0; i < AlgTraits::nodes1D_; ++i) {
      rhs(v_node_map_(j*AlgTraits::nodes1D_+i)) += v_rhs_(j,i);
    }
  }
}

INSTANTIATE_HOQUAD_ALGORITHM(ScalarAdvDiffHOElemKernel)

} // namespace nalu
} // namespace Sierra
