/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include <element_promotion/computational_kernels/ScalarAdvDiffHOElemKernel.h>

#include <element_promotion/operators/HighOrderAdvectionDiffusionQuad.h>
#include <element_promotion/operators/HighOrderGeometryQuadDiffusion.h>
#include <element_promotion/operators/HighOrderGeometryQuadAdvection.h> // advection-diffusion metric instead?
#include <element_promotion/operators/MappedElementMatrixScatter.h>
#include <element_promotion/operators/CoefficientMatrices.h>
#include <element_promotion/operators/DirectionEnums.h>
#include <element_promotion/ElementDescription.h>

#include <Kernel.h>
#include <SolutionOptions.h>
#include <FieldTypeDef.h>
#include <Realm.h>
#include <AlgTraits.h>

// template and scratch space
#include <BuildTemplates.h>
#include <ScratchViews.h>

// stk_mesh/base/fem
#include <stk_mesh/base/Entity.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/Field.hpp>

#include <Teuchos_BLAS.hpp>

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
  velocity_ = meta_data.get_field<VectorFieldType>(stk::topology::NODE_RANK, "velocity");
  density_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "density");

  // only necessary for correctly sizing scratch views
  dataPreReqs.add_cvfem_surface_me(get_surface_master_element(AlgTraits::topo_));

  dataPreReqs.add_gathered_nodal_field(*velocity_, AlgTraits::nDim_);
  dataPreReqs.add_gathered_nodal_field(*density_, 1);
  dataPreReqs.add_gathered_nodal_field(*coordinates_, AlgTraits::nDim_);
  dataPreReqs.add_gathered_nodal_field(*scalarQ, 1);
  dataPreReqs.add_gathered_nodal_field(*diffFluxCoeff, 1);
}
//--------------------------------------------------------------------------
template <class AlgTraits> void
ScalarAdvDiffHOElemKernel<AlgTraits>::execute(
  SharedMemView<double **>& lhs,
  SharedMemView<double *>& rhs,
  ScratchViews& scratchViews)
{
  constexpr int npe = AlgTraits::nodesPerElement_;
  constexpr int n1D = AlgTraits::nodes1D_;
  constexpr int poly_order = AlgTraits::polyOrder_;

  SharedMemView<double**> v_flatCoords = scratchViews.get_scratch_view_2D(*coordinates_);
  SharedMemView<double*> v_flatScalar = scratchViews.get_scratch_view_1D(*scalarQ_);
  SharedMemView<double*> v_flatDiff  = scratchViews.get_scratch_view_1D(*diffFluxCoeff_);
  SharedMemView<double**> v_flatVelocity = scratchViews.get_scratch_view_2D(*velocity_);
  SharedMemView<double*> v_flatDensity = scratchViews.get_scratch_view_1D(*density_);

  // reorder fields into the ordering expected by the alg
  for (int j = 0; j < n1D; ++j) {
    for (int i = 0; i < n1D; ++i) {
      int nodeId = v_node_map_(j*n1D+i);
      v_scalar_(j,i) = v_flatScalar(nodeId);
      v_diff_(j,i)   = 1.0e-10;
      for (int d = 0; d < AlgTraits::nDim_; ++d) {
        v_coords_(d,j,i) = v_flatCoords(nodeId, d);
        v_rhou_(d,j,i) = v_flatVelocity(nodeId, d) * v_flatDensity(nodeId);
      }
    }
  }
//
  Kokkos::deep_copy(v_lhs_, 0.0);
  Kokkos::deep_copy(v_rhs_, 0.0);
//
//  // fixme(rcknaus): can't just interpolate the current rhou.  Need a templated mdot calculation +
//  // which requires a strategy for dealing with nonsolver entities
  high_order_metrics::compute_diffusion_metric_linear(ops_, v_coords_, v_diff_, v_diff_metric_);
  high_order_metrics::compute_advection_metric_linear(ops_, v_coords_, v_rhou_, v_adv_metric_);

  Kokkos::deep_copy(v_adv_metric_,0.0);

//  Kokkos::deep_copy(v_adv_metric_ , 0.0);

  for (int j = 0; j < n1D-1; ++j) {
    for (int i = 0; i < n1D-1; ++i) {
      v_adv_metric_(XH,j,i) =  1.0/16.0;
//      std::cout << "adv: (" << v_adv_metric_(XH,j,i) << ", " << v_adv_metric_(YH,j,i)  << ")" << std::endl;
    }
  }

  tensor_assembly::elemental_advection_diffusion_jacobian(ops_, v_adv_metric_, v_diff_metric_, v_lhs_);
//  tensor_assembly::elemental_advection_diffusion_action(ops_, v_adv_metric_, v_diff_metric_, v_scalar_, v_rhs_);
//
  Teuchos::BLAS<int, typename matrix_view<AlgTraits>::value_type>().GEMV(
    Teuchos::TRANS, // row v column
    npe, npe,
    -1.0,
    v_lhs_.ptr_on_device(), npe,
    v_scalar_.ptr_on_device(), 1,
    +1.0,
    v_rhs_.ptr_on_device(), 1
  );

  tensor_assembly::mapped_scatter<poly_order>(v_node_map_, v_lhs_, v_rhs_, lhs, rhs);
//  ThrowRequire(false);
}

INSTANTIATE_HOQUAD_ALGORITHM(ScalarAdvDiffHOElemKernel)

} // namespace nalu
} // namespace Sierra
