/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include <element_promotion/computational_kernels/ScalarMassHOElemKernel.h>

#include <element_promotion/operators/HighOrderGeometryQuadVolume.h>
#include <element_promotion/operators/HighOrderSourceQuad.h>
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

// Kokkos
#include <Kokkos_Core.hpp>

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

  scalarN_   = &(scalarQ->field_of_state(stk::mesh::StateN));
  scalarNp1_ = &(scalarQ->field_of_state(stk::mesh::StateNP1));
  scalarNm1_ = (scalarQ->number_of_states() == 2) ? scalarN_ : &(scalarQ->field_of_state(stk::mesh::StateNM1));

  ScalarFieldType* density = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "density");

  densityN_ = &(density->field_of_state(stk::mesh::StateN));
  densityNp1_ = &(density->field_of_state(stk::mesh::StateNP1));
  densityNm1_ = (density->number_of_states() == 2) ? densityN_ : &(density->field_of_state(stk::mesh::StateNM1));

  coordinates_ = meta_data.get_field<VectorFieldType>( stk::topology::NODE_RANK, solnOpts.get_coordinates_name());

  // only necessary for correctly sizing scratch views
  dataPreReqs.add_cvfem_surface_me(get_surface_master_element(AlgTraits::topo_));

  dataPreReqs.add_coordinates_field(*coordinates_, AlgTraits::nDim_, CURRENT_COORDINATES);
  dataPreReqs.add_gathered_nodal_field(*scalarNm1_, 1);
  dataPreReqs.add_gathered_nodal_field(*scalarN_, 1);
  dataPreReqs.add_gathered_nodal_field(*scalarNp1_, 1);
  dataPreReqs.add_gathered_nodal_field(*densityNm1_, 1);
  dataPreReqs.add_gathered_nodal_field(*densityN_, 1);
  dataPreReqs.add_gathered_nodal_field(*densityNp1_, 1);
}
//--------------------------------------------------------------------------
template<typename AlgTraits>
void
ScalarMassHOElemKernel<AlgTraits>::setup(const TimeIntegrator& timeIntegrator)
{
  dt_ = timeIntegrator.get_time_step();
  gamma1_ = timeIntegrator.get_gamma1();
  gamma2_ = timeIntegrator.get_gamma2();
  gamma3_ = timeIntegrator.get_gamma3(); // gamma3 may be zero
}
//--------------------------------------------------------------------------
namespace
{
  template <int nodes1D> int idx(int i, int j) { return i*nodes1D+j; };
}
//--------------------------------------------------------------------------
template <class AlgTraits> void
ScalarMassHOElemKernel<AlgTraits>::execute(
  SharedMemView<double **>& lhs,
  SharedMemView<double *>& rhs,
  ScratchViews<double>& scratchViews)
{
  constexpr int n1D = AlgTraits::nodes1D_;
  constexpr int poly_order = AlgTraits::polyOrder_;

  SharedMemView<double**> v_flatCoords = scratchViews.get_scratch_view_2D(*coordinates_);

  SharedMemView<double*> v_scalarNm1 = scratchViews.get_scratch_view_1D(*scalarNm1_);
  SharedMemView<double*> v_scalarN   = scratchViews.get_scratch_view_1D(*scalarN_);
  SharedMemView<double*> v_scalarNp1 = scratchViews.get_scratch_view_1D(*scalarNp1_);

  SharedMemView<double*> v_densityNm1 = scratchViews.get_scratch_view_1D(*densityNm1_);
  SharedMemView<double*> v_densityN   = scratchViews.get_scratch_view_1D(*densityN_);
  SharedMemView<double*> v_densityNp1 = scratchViews.get_scratch_view_1D(*densityNp1_);


  // reorder fields into the ordering expected by the alg
  for (int j = 0; j < n1D; ++j) {
    for (int i = 0; i < n1D; ++i) {
      int nodeId = v_node_map_(j*n1D+i);
      v_rhoNm1_(j,i) = v_densityNm1(nodeId);
      v_rhoNp0_(j,i) = v_densityN(nodeId);
      v_rhoNp1_(j,i) = v_densityNp1(nodeId);

      v_scalarNm1_(j,i) = v_scalarNm1(nodeId);
      v_scalarNp0_(j,i) = v_scalarN(nodeId);
      v_scalarNp1_(j,i) = v_scalarNp1(nodeId);

      for (int d = 0; d < AlgTraits::nDim_; ++d) {
        v_coords_(d,j,i) = v_flatCoords(nodeId, d);
      }
    }
  }
  Kokkos::deep_copy(v_lhs_, 0.0);
  Kokkos::deep_copy(v_rhs_, 0.0);

  const auto& weight = ops_.mat_.nodalWeights;

  high_order_metrics::compute_volume_metric_linear(ops_, v_coords_, v_vol_);
  double inv_dt = 1.0 / dt_;

  for (int n = 0; n < n1D; ++n) {
    for (int m = 0; m < n1D; ++m) {
      for (int j = 0; j < n1D; ++j) {
        for (int i = 0; i < n1D; ++i) {
          v_lhs_(idx<n1D>(n,m), idx<n1D>(j,i)) +=  gamma1_ * inv_dt * weight(n,j) * weight(m,i) * v_vol_(j,i) * v_rhoNp1_(j,i);
        }
      }
    }
  }

  for (int j = 0; j < n1D; ++j) {
    for (int i = 0; i < n1D; ++i) {
      v_time_derivative_(j,i) = -(
          gamma1_ * v_rhoNp1_(j,i) * v_scalarNp1_(j,i)
        + gamma2_ * v_rhoNp0_(j,i) * v_scalarNp0_(j,i)
        + gamma3_ * v_rhoNm1_(j,i) * v_scalarNm1_(j,i)) * inv_dt;
    }
  }

  tensor_assembly::add_volumetric_source(ops_, v_vol_, v_time_derivative_, v_rhs_);
  tensor_assembly::mapped_scatter<poly_order>(v_node_map_, v_lhs_, v_rhs_, lhs, rhs);
}

INSTANTIATE_KERNEL_2D_HOSGL(ScalarMassHOElemKernel)

} // namespace nalu
} // namespace Sierra
