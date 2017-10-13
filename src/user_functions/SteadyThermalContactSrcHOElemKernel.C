/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#include <user_functions/SteadyThermalContactSrcHOElemKernel.h>
#include <Kernel.h>
#include <FieldTypeDef.h>
#include <Realm.h>
#include <SolutionOptions.h>

#include <element_promotion/ElementDescription.h>
#include <element_promotion/operators/HighOrderOperatorsQuad.h>
#include <element_promotion/operators/HighOrderGeometryQuadVolume.h>
#include <element_promotion/operators/HighOrderSourceQuad.h>

#include <BuildTemplates.h>
#include <ScratchViews.h>

// stk_mesh/base/fem
#include <stk_mesh/base/Entity.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Field.hpp>

namespace sierra{
namespace nalu{

template <class AlgTraits>
SteadyThermalContactSrcHOElemKernel<AlgTraits>::SteadyThermalContactSrcHOElemKernel(
  const stk::mesh::BulkData& bulkData,
  SolutionOptions& solnOpts,
  ElemDataRequests& dataPreReqs)
  : Kernel(),
    a_(1.0),
    k_(1.0),
    pi_(std::acos(-1.0)),
    ops_(),
    v_node_map_(make_node_map<AlgTraits::polyOrder_, AlgTraits::baseTopo_>())
{
  // save off fields
  const stk::mesh::MetaData& meta_data = bulkData.mesh_meta_data();
  coordinates_ = meta_data.get_field<VectorFieldType>(stk::topology::NODE_RANK, solnOpts.get_coordinates_name());

  dataPreReqs.add_cvfem_volume_me(MasterElementRepo::get_volume_master_element(AlgTraits::topo_));
  dataPreReqs.add_coordinates_field(*coordinates_, AlgTraits::nDim_, CURRENT_COORDINATES);
}
//--------------------------------------------------------------------------
template<class AlgTraits>
void
SteadyThermalContactSrcHOElemKernel<AlgTraits>::execute(
  SharedMemView<DoubleType**>& /*lhs*/,
  SharedMemView<DoubleType*>& rhs,
  ScratchViews<DoubleType>& scratchViews)
{
//  SharedMemView<DoubleType**>& v_flatCoords = scratchViews.get_scratch_view_2D(*coordinates_);
//
//  for (int j = 0; j < AlgTraits::nodes1D_; ++j) {
//    for (int i = 0; i < AlgTraits::nodes1D_; ++i) {
//      int nodeId = v_node_map_(j*AlgTraits::nodes1D_+i);
//      v_coords_(XH,j,i) = v_flatCoords(nodeId, XH);
//      v_coords_(YH,j,i) = v_flatCoords(nodeId, YH);
//      v_nodalSource_(j,i) = k_/4.0*(2.0*a_*pi_)*(2.0*a_*pi_)*
//          (stk::math::cos(2.0*a_*pi_*v_coords_(XH,j,i)) + stk::math::cos(2.0*a_*pi_*v_coords_(YH,j,i)));
//    }
//  }
//
//  Kokkos::deep_copy(v_rhs_, DoubleType(0.0));
//  high_order_metrics::compute_volume_metric_linear(ops_, v_coords_, v_vol_);
//  tensor_assembly::add_volumetric_source(ops_, v_vol_, v_nodalSource_, v_rhs_);
//
//  for (int j = 0; j < AlgTraits::nodes1D_; ++j) {
//    for (int i = 0; i < AlgTraits::nodes1D_; ++i) {
//      rhs(v_node_map_(j* AlgTraits::nodes1D_ +i)) += v_rhs_(j,i);
//    }
//  }
}

INSTANTIATE_KERNEL_2D_HOSGL(SteadyThermalContactSrcHOElemKernel);

} // namespace nalu
} // namespace Sierra
