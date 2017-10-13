/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level NaluUnit      */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/
#ifndef MappedElementMatrixGather_h
#define MappedElementMatrixGather_h

#include <element_promotion/operators/HighOrderOperatorsQuad.h>
#include <element_promotion/operators/CoefficientMatrices.h>
#include <element_promotion/CVFEMTypeDefs.h>

// todo(rcknaus): consider removing the need for this

namespace sierra {
namespace nalu {
namespace tensor_assembly {

  template <int p>
  void gather_elem_node_field(
    const node_map_view<AlgTraitsQuad<p>>& nodeMap,
    const stk::mesh::FieldBase& field,
    const stk::mesh::Entity* elemNodes,
    SharedMemView<double*>& shmemView)
  {

  }

} // namespace TensorAssembly
} // namespace naluUnit
} // namespace Sierra

#endif
