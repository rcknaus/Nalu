/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#include <element_promotion/computational_kernels/ScalarDiffHOElemKernel.h>

#include <element_promotion/operators/HighOrderDiffusionQuad.h>
#include <element_promotion/operators/HighOrderGeometryQuadDiffusion.h>
#include <element_promotion/operators/CoefficientMatrices.h>
#include <element_promotion/operators/DirectionEnums.h>
#include <element_promotion/operators/MappedElementMatrixScatter.h>
#include <element_promotion/ElementDescription.h>
#include <SolutionOptions.h>

#include <Kernel.h>
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
#include <KokkosInterface.h>
#include <SimdInterface.h>

namespace sierra{
namespace nalu{

template<class AlgTraits>
ScalarDiffHOElemKernel<AlgTraits>::ScalarDiffHOElemKernel(
  const stk::mesh::BulkData& bulkData,
  SolutionOptions& solnOpts,
  ScalarFieldType *scalarQ,
  ScalarFieldType *diffFluxCoeff,
  ElemDataRequests& dataPreReqs)
  : Kernel(),
    scalarQ_(scalarQ),
    diffFluxCoeff_(diffFluxCoeff)
{
  const stk::mesh::MetaData& meta_data = bulkData.mesh_meta_data();
  coordinates_ = meta_data.get_field<VectorFieldType>(stk::topology::NODE_RANK, solnOpts.get_coordinates_name());

  // only necessary for correctly sizing scratch views
  dataPreReqs.add_cvfem_surface_me(MasterElementRepo::get_surface_master_element(AlgTraits::topo_));

  dataPreReqs.add_gathered_nodal_field(*coordinates_, AlgTraits::nDim_);
  dataPreReqs.add_gathered_nodal_field(*scalarQ, 1);
  dataPreReqs.add_gathered_nodal_field(*diffFluxCoeff, 1);
}
//--------------------------------------------------------------------------
template <class AlgTraits> void
ScalarDiffHOElemKernel<AlgTraits>::execute(
  SharedMemView<DoubleType**>& lhs,
  SharedMemView<DoubleType*>& rhs,
  ScratchViews<DoubleType>& scratchViews)
{
  nodal_vector_view<AlgTraits, DoubleType> v_coords(scratchViews.get_scratch_view_3D(*coordinates_));
  nodal_scalar_view<AlgTraits, DoubleType> v_scalar(scratchViews.get_scratch_view_2D(*scalarQ_));
  nodal_scalar_view<AlgTraits, DoubleType> v_diff(scratchViews.get_scratch_view_2D(*diffFluxCoeff_));

  DoubleType scratch_lhs[AlgTraits::nodesPerElement_*AlgTraits::nodesPerElement_];
  matrix_view<AlgTraits, DoubleType> v_lhs(scratch_lhs);

  DoubleType scratch_rhs[AlgTraits::nodesPerElement_];
  nodal_scalar_view<AlgTraits, DoubleType> v_rhs(scratch_rhs);

  Kokkos::deep_copy(v_lhs, DoubleType(0.0));
  Kokkos::deep_copy(v_rhs, DoubleType(0.0));

  DoubleType scratch_metric[AlgTraits::nDim_*AlgTraits::nDim_*AlgTraits::nscs_*AlgTraits::nodes1D_];
  scs_tensor_view<AlgTraits, DoubleType> v_metric(scratch_metric);

  high_order_metrics::compute_diffusion_metric_linear(ops_, v_coords, v_diff, v_metric);
  tensor_assembly::elemental_diffusion_jacobian(ops_, v_metric, v_lhs);
  tensor_assembly::elemental_diffusion_action(ops_, v_metric, v_scalar, v_rhs);
  tensor_assembly::mapped_scatter<AlgTraits::polyOrder_>(v_node_map_, v_lhs, v_rhs, lhs, rhs);
}

INSTANTIATE_KERNEL_2D_HOSGL(ScalarDiffHOElemKernel)

} // namespace nalu
} // namespace Sierra
