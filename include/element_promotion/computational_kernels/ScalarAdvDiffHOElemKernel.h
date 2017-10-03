/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#ifndef ScalarAdvDiffHOElemKernel_h
#define ScalarAdvDiffHOElemKernel_h

#include <Kernel.h>
#include <AlgTraits.h>

#include <element_promotion/operators/HighOrderOperatorsQuad.h>
#include <element_promotion/CVFEMTypeDefs.h>
#include <FieldTypeDef.h>

#include <stk_mesh/base/Entity.hpp>

// Kokkos
#include <Kokkos_Core.hpp>

namespace sierra{
namespace nalu{

class ElemDataRequests;
class ScratchViews;
class MasterElement;

template<class AlgTraits>
class ScalarAdvDiffHOElemKernel final: public Kernel
{
public:
  ScalarAdvDiffHOElemKernel(
    const stk::mesh::BulkData& bulkData,
    SolutionOptions& solnOpts,
    ScalarFieldType *scalarQ,
    ScalarFieldType *diffFluxCoeff,
    const ElementDescription* desc,
    ElemDataRequests& dataPreReqs);

  void execute(
    SharedMemView<double**>& lhs,
    SharedMemView<double*>& rhs,
    ScratchViews& scratchViews) final;
private:
  ScalarFieldType *scalarQ_{nullptr};
  ScalarFieldType *diffFluxCoeff_{nullptr};
  VectorFieldType *coordinates_{nullptr};
  VectorFieldType* velocity_{nullptr};
  ScalarFieldType* density_{nullptr};

  CVFEMOperators<AlgTraits::polyOrder_, AlgTraits::baseTopo_> ops_{};
  node_map_view<AlgTraits> v_node_map_{ make_node_map<AlgTraits::polyOrder_, AlgTraits::baseTopo_>() };
  nodal_scalar_view<AlgTraits> v_diff_{"scalar_ho_diffusion_coeff"};
  nodal_scalar_view<AlgTraits> v_scalar_{"scalar_ho_q"};
  nodal_vector_view<AlgTraits> v_coords_{"coords"};
  nodal_vector_view<AlgTraits> v_rhou_{"rhou"};
  scs_vector_view<AlgTraits> v_adv_metric_{"adv_metric"};
  scs_tensor_view<AlgTraits> v_diff_metric_{"diff_metric"};
  matrix_view<AlgTraits> v_lhs_{"scalar_ho_advdiff_lhs"};
  nodal_scalar_view<AlgTraits> v_rhs_{"scalar_ho_advdiff_rhs"};
};

} // namespace nalu
} // namespace Sierra

#endif
