/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#ifndef ScalarMassHOElemKernel_h
#define ScalarMassHOElemKernel_h

#include <Kernel.h>
#include <AlgTraits.h>

#include <element_promotion/operators/HighOrderOperatorsQuad.h>
#include <element_promotion/CVFEMTypeDefs.h>
#include <FieldTypeDef.h>

#include <stk_mesh/base/Entity.hpp>

// Kokkos
#include <KokkosInterface.h>

namespace sierra{
namespace nalu{

class ElemDataRequests;

template<class AlgTraits>
class ScalarMassHOElemKernel final : public Kernel
{
public:
  ScalarMassHOElemKernel(
    const stk::mesh::BulkData& bulkData,
    SolutionOptions& solnOpts,
    ScalarFieldType *scalarQ,
    ElemDataRequests& dataPreReqs);

  void setup(const TimeIntegrator& timeIntegrator) final;

  void execute(
    SharedMemView<DoubleType**>&,
    SharedMemView<DoubleType*>&,
    ScratchViews<DoubleType>&) final;

private:
  ScalarFieldType* scalarNm1_{nullptr};
  ScalarFieldType* scalarN_{nullptr};
  ScalarFieldType* scalarNp1_{nullptr};

  ScalarFieldType* densityNm1_{nullptr};
  ScalarFieldType* densityN_{nullptr};
  ScalarFieldType* densityNp1_{nullptr};

  VectorFieldType* coordinates_{nullptr};

  double dt_{0.0};
  double gamma1_{0.0};
  double gamma2_{0.0};
  double gamma3_{0.0};

  CVFEMOperators<AlgTraits::polyOrder_, AlgTraits::baseTopo_> ops_{};
  node_map_view<AlgTraits> v_node_map_{ make_node_map<AlgTraits::polyOrder_, AlgTraits::baseTopo_>() };

  nodal_scalar_view<AlgTraits> v_scalarNm1_{"scalarnm1"};
  nodal_scalar_view<AlgTraits> v_scalarNp0_{"scalarn"};
  nodal_scalar_view<AlgTraits> v_scalarNp1_{"scalarnp1"};

  nodal_scalar_view<AlgTraits> v_rhoNm1_{"rhonm1"};
  nodal_scalar_view<AlgTraits> v_rhoNp0_{"rhon"};
  nodal_scalar_view<AlgTraits> v_rhoNp1_{"rhonp1"};

  nodal_vector_view<AlgTraits> v_coords_{"coords"};
  nodal_scalar_view<AlgTraits> v_vol_{"volume_metric"};
  nodal_scalar_view<AlgTraits> v_time_derivative_{"dqdt"};
  matrix_view<AlgTraits> v_lhs_{"scalar_ho_advdiff_lhs"};
  nodal_scalar_view<AlgTraits> v_rhs_{"scalar_ho_advdiff_rhs"};
};

} // namespace nalu
} // namespace Sierra

#endif
