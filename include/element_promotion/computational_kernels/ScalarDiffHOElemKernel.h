/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#ifndef ScalarDiffHOElemKernel_h
#define ScalarDiffHOElemKernel_h

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
class Realm;
class MasterElement;

template<class AlgTraits>
class ScalarDiffHOElemKernel final: public Kernel
{
public:
  ScalarDiffHOElemKernel(
    const stk::mesh::BulkData& bulkData,
    SolutionOptions& solnOpts,
    ScalarFieldType *scalarQ,
    ScalarFieldType *diffFluxCoeff,
    ElemDataRequests& dataPreReqs);

  void execute(
    SharedMemView<DoubleType**>&,
    SharedMemView<DoubleType*>&,
    ScratchViews<DoubleType>&) final;

private:
  ScalarFieldType *scalarQ_;
  ScalarFieldType *diffFluxCoeff_;
  VectorFieldType *coordinates_;

  CVFEMOperators<AlgTraits::polyOrder_, DoubleType, AlgTraits::baseTopo_> ops_{};
  node_map_view<AlgTraits> v_node_map_{make_node_map<AlgTraits::polyOrder_, AlgTraits::baseTopo_>()};
  nodal_scalar_view<AlgTraits> v_diff_{"scalar_ho_diffusion_coeff"};
  nodal_scalar_view<AlgTraits> v_scalar_{"scalar_ho_q"};
  nodal_vector_view<AlgTraits> v_coords_{"coords"};
  scs_tensor_view<AlgTraits> v_metric_{"metric"};
  matrix_view<AlgTraits> v_lhs_{"scalar_ho_diff_lhs"};
  nodal_scalar_view<AlgTraits> v_rhs_{"scalar_ho_diff_rhs"};
};

} // namespace nalu
} // namespace Sierra

#endif
