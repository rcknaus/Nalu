  /*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef SteadyThermalContactSrcHOElemKernel_h
#define SteadyThermalContactSrcHOElemKernel_h

#include <element_promotion/CVFEMTypeDefs.h>
#include <element_promotion/operators/HighOrderOperatorsQuad.h>

#include <Kernel.h>
#include <FieldTypeDef.h>
#include <AlgTraits.h>

#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Entity.hpp>
#include <memory>

namespace sierra{
namespace nalu{

class Realm;
class ElementDescription;
class ElemDataRequests;

template <class AlgTraits>
class SteadyThermalContactSrcHOElemKernel final : public Kernel
{
public:
  SteadyThermalContactSrcHOElemKernel(
    const stk::mesh::BulkData& bulkData,
    SolutionOptions& solnOpts,
    const ElementDescription& desc,
    ElemDataRequests& dataPreReqs);

  void execute(
    SharedMemView<double **>& lhs,
    SharedMemView<double *>& rhs,
    ScratchViews& scratchViews) final;

private:
  VectorFieldType *coordinates_;

  const double a_;
  const double k_;
  const double pi_;

  // fixed scratch space
  CVFEMOperatorsQuad<AlgTraits::polyOrder_> ops_;
  nodal_scalar_view<AlgTraits, int> v_node_map_{"tensor_product_node_map"};
  nodal_scalar_view<AlgTraits> v_nodalSource_{"v_nodal_source"};
  nodal_scalar_view<AlgTraits> v_vol_{"v_volume_metric"};
  nodal_scalar_view<AlgTraits> v_rhs_{"v_rhs"};
  nodal_vector_view<AlgTraits> v_coords_{"v_coords"};
};

} // namespace nalu
} // namespace Sierra

#endif
