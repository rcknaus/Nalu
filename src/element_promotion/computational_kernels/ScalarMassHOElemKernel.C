/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include <element_promotion/computational_kernels/ScalarMassHOElemKernel.h>

#include <element_promotion/operators/HighOrderGeometryQuadVolume.h>
#include <element_promotion/operators/HighOrderScalarBDF2TimeDerivativeQuad.h>
#include <element_promotion/operators/CoefficientMatrices.h>
#include <element_promotion/operators/DirectionEnums.h>
#include <element_promotion/operators/MappedElementMatrixScatter.h>

#include <element_promotion/ElementDescription.h>

#include <Kernel.h>
#include <SolutionOptions.h>
#include <FieldTypeDef.h>
#include <Realm.h>
#include <AlgTraits.h>
#include <TimeIntegrator.h>

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

#include <KokkosInterface.h>
#include <SimdInterface.h>

namespace sierra{
namespace nalu{

template<class AlgTraits>
ScalarMassHOElemKernel<AlgTraits>::ScalarMassHOElemKernel(
  const stk::mesh::BulkData& bulkData,
  SolutionOptions& solnOpts,
  ScalarFieldType *scalarQ,
  ElemDataRequests& dataPreReqs)
  : Kernel()
{
  // save off fields
  const stk::mesh::MetaData& meta_data = bulkData.mesh_meta_data();
  coordinates_ = meta_data.get_field<VectorFieldType>(stk::topology::NODE_RANK, solnOpts.get_coordinates_name());

  ThrowRequireMsg(scalarQ != nullptr, "Scalar field not valid");
  scalarN_   = &(scalarQ->field_of_state(stk::mesh::StateN));
  scalarNp1_ = &(scalarQ->field_of_state(stk::mesh::StateNP1));
  scalarNm1_ = (scalarQ->number_of_states() == 2) ? scalarN_ : &(scalarQ->field_of_state(stk::mesh::StateNM1));


  density_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "density");
  ThrowRequireMsg(density_ != nullptr, "density field not valid");

  specificHeat_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "density");
  ThrowRequireMsg(specificHeat_ != nullptr, "specific heat field not valid");

  coordinates_ = meta_data.get_field<VectorFieldType>( stk::topology::NODE_RANK, solnOpts.get_coordinates_name());

  // only necessary for correctly sizing scratch views
  dataPreReqs.add_cvfem_surface_me(MasterElementRepo::get_surface_master_element(AlgTraits::topo_));

  dataPreReqs.add_coordinates_field(*coordinates_, AlgTraits::nDim_, CURRENT_COORDINATES);
  dataPreReqs.add_gathered_nodal_field(*scalarNm1_, 1);
  dataPreReqs.add_gathered_nodal_field(*scalarN_, 1);
  dataPreReqs.add_gathered_nodal_field(*scalarNp1_, 1);
  dataPreReqs.add_gathered_nodal_field(*density_, 1);
  dataPreReqs.add_gathered_nodal_field(*specificHeat_, 1);
}
//--------------------------------------------------------------------------
template<typename AlgTraits>
void
ScalarMassHOElemKernel<AlgTraits>::setup(const TimeIntegrator& timeIntegrator)
{
  gamma_[0] = timeIntegrator.get_gamma1() / timeIntegrator.get_time_step();
  gamma_[1] = timeIntegrator.get_gamma2() / timeIntegrator.get_time_step();
  gamma_[2] = timeIntegrator.get_gamma3() / timeIntegrator.get_time_step();
}
//--------------------------------------------------------------------------
template <class AlgTraits> void
ScalarMassHOElemKernel<AlgTraits>::execute(
  SharedMemView<DoubleType**>& lhs,
  SharedMemView<DoubleType*>& rhs,
  ScratchViews<DoubleType>& scratchViews)
{
  nodal_vector_view<AlgTraits, DoubleType> v_coords(scratchViews.get_scratch_view_3D(*coordinates_));
  nodal_scalar_view<AlgTraits, DoubleType> v_scalarNm1(scratchViews.get_scratch_view_2D(*scalarNm1_));
  nodal_scalar_view<AlgTraits, DoubleType> v_scalarNp0(scratchViews.get_scratch_view_2D(*scalarN_));
  nodal_scalar_view<AlgTraits, DoubleType> v_scalarNp1(scratchViews.get_scratch_view_2D(*scalarNp1_));
  nodal_scalar_view<AlgTraits, DoubleType> v_density(scratchViews.get_scratch_view_2D(*density_));
  nodal_scalar_view<AlgTraits, DoubleType> v_specificHeat(scratchViews.get_scratch_view_2D(*specificHeat_));

  DoubleType vol[AlgTraits::nodesPerElement_];
  nodal_scalar_view<AlgTraits, DoubleType> v_vol(vol);
  high_order_metrics::compute_volume_metric_linear(ops_, v_coords, v_vol);

  DoubleType* p_vol = v_vol.ptr_on_device(); // (vol)
  const DoubleType* p_dens = v_density.ptr_on_device();
  const DoubleType* p_cp = v_specificHeat.ptr_on_device();
  for (int j = 0; j < AlgTraits::nodesPerElement_; ++j) {
    p_vol[j] *= p_dens[j] * p_cp[j];
  }

  DoubleType elem_lhs[AlgTraits::lhsSize_];
  matrix_view<AlgTraits, DoubleType> v_lhs(elem_lhs);
  Kokkos::deep_copy(v_lhs, 0.0);

  DoubleType elem_rhs[AlgTraits::nodesPerElement_];
  nodal_scalar_view<AlgTraits, DoubleType> v_rhs(elem_rhs);
  Kokkos::deep_copy(v_rhs, 0.0);

  tensor_assembly::elemental_time_derivative_jacobian(ops_, v_vol, gamma_[0], v_lhs);
  tensor_assembly::elemental_time_derivative_action(
    ops_, v_vol, gamma_[0], gamma_[1], gamma_[2],
    v_scalarNm1, v_scalarNp0, v_scalarNp1, v_rhs
  );
  tensor_assembly::mapped_scatter<AlgTraits::polyOrder_>(v_node_map_, v_lhs, v_rhs, lhs, rhs);
}

INSTANTIATE_KERNEL_2D_HOSGL(ScalarMassHOElemKernel)

} // namespace nalu
} // namespace Sierra
