/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include <element_promotion/PromotedElementFieldInterp.h>
#include <element_promotion/PromotedPartHelper.h>
#include <element_promotion/ElementDescription.h>
#include <master_element/MasterElement.h>
#include <NaluEnv.h>
#include <BucketLoop.h>

#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/GetBuckets.hpp>
#include <stk_mesh/base/GetEntities.hpp>
#include <stk_mesh/base/Part.hpp>
#include <stk_mesh/base/Selector.hpp>
#include <stk_topology/topology.hpp>
#include <stk_util/environment/ReportHandler.hpp>

#include <vector>

namespace sierra {
namespace nalu {
namespace promotion {

void interpolate_field_to_new_nodes(
  const stk::mesh::BulkData& bulk,
  const ElementDescription& desc,
  const stk::mesh::PartVector& promotedParts,
  const stk::mesh::FieldBase& field)
{
  ThrowRequireMsg(field.entity_rank() == stk::topology::NODE_RANK, "Only node-ranked fields allowed");
  ThrowRequireMsg(field.type_is<double>(), "Only double-type fields allowed");

  auto* meBase = MasterElementRepo::get_surface_master_element(desc.baseTopo);
  const int maxSize = field.max_size(stk::topology::NODE_RANK);
  const int numBaseNodes = desc.nodesInBaseElement;
  std::vector<double> baseField(numBaseNodes * maxSize);
  std::vector<double> interpolatedField(maxSize);

  const auto& elem_buckets = bulk.get_buckets(stk::topology::ELEM_RANK, stk::mesh::selectUnion(promotedParts));
  bucket_loop(elem_buckets, [&](const stk::mesh::Entity elem) {
    ThrowAssert(desc.nodesPerElement == static_cast<int>(bulk.num_nodes(elem)));
    const stk::mesh::Entity* node_rels = bulk.begin_nodes(elem);

    for (int ord : desc.baseNodeOrdinals) {
      const double* field_data = static_cast<const double*>(stk::mesh::field_data(field, node_rels[ord]));
      for (int d = 0; d < maxSize; ++d) {
        baseField[d * numBaseNodes + ord] = field_data[d];
      }
    }

    for (int ord : desc.promotedNodeOrdinals) {
      const auto& isoParCoords = desc.nodeLocs.at(ord);
      meBase->interpolatePoint(maxSize, isoParCoords.data(), baseField.data(), interpolatedField.data());
      double* field_data = static_cast<double*>(stk::mesh::field_data(field, node_rels[ord]));
      for (int d = 0; d < maxSize; ++d) {
        field_data[d] = interpolatedField[d];
      }
    }
  });
}

} // namespace promotion
} // namespace naluUnit
} // namespace Sierra
