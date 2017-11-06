/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level NaluUnit      */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/
#ifndef PromotedElementFieldInterp_h
#define PromotedElementFieldInterp_h

#include <stk_mesh/base/FieldBase.hpp>
#include <vector>

namespace stk { namespace mesh { class Part; } }
namespace stk { namespace mesh { class BulkData; } }
namespace stk { namespace mesh { typedef std::vector<Part*> PartVector; } }
namespace sierra { namespace nalu { struct ElementDescription; } }


namespace sierra {
namespace nalu {
namespace promotion {

  void interpolate_field_to_new_nodes(
    const stk::mesh::BulkData& bulk,
    const ElementDescription& desc,
    const stk::mesh::PartVector& baseParts,
    const stk::mesh::FieldBase& field);


} // namespace promotion
} // namespace naluUnit
} // namespace Sierra

#endif
