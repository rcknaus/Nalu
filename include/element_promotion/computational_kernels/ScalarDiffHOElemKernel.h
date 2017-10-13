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
#include <element_promotion/NodeMapMaker.h>
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
};

} // namespace nalu
} // namespace Sierra

#endif
