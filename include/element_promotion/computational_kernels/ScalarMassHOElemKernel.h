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
#include <element_promotion/NodeMapMaker.h>
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
  ScalarFieldType* density_{nullptr};
  ScalarFieldType* specificHeat_{nullptr};

  VectorFieldType* coordinates_{nullptr};

  double gamma_[3];

  CVFEMOperators<AlgTraits::polyOrder_, DoubleType, AlgTraits::baseTopo_> ops_{};
  node_map_view<AlgTraits> v_node_map_{ make_node_map<AlgTraits::polyOrder_, AlgTraits::baseTopo_>() };
};

} // namespace nalu
} // namespace Sierra

#endif
