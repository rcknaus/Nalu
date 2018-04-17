#include <gtest/gtest.h>
#include <limits>

#include "master_element/MasterElement.h"
#include "UnitTestMeshFixtures.h"
#include "UnitTestUtils.h"

#include <stk_util/parallel/Parallel.hpp>
#include <stk_mesh/base/FieldParallel.hpp>
#include <stk_mesh/base/Bucket.hpp>
#include <stk_mesh/base/FieldBLAS.hpp>

namespace sierra {
namespace nalu {

typedef stk::mesh::Field<double>  ScalarFieldType;
typedef stk::mesh::Field<double, stk::mesh::Cartesian>  VectorFieldType;

template <typename FuncType> void loop_over_all_non_aura_nodes(const stk::mesh::BulkData& bulk, FuncType func)
{
  const stk::mesh::Selector not_aura = !stk::mesh::Selector(bulk.mesh_meta_data().aura_part());
   for (const auto* ib : bulk.get_buckets(stk::topology::NODE_RANK, not_aura)) {
     for (size_t k = 0; k < ib->size(); ++k) {
       func((*ib)[k]);
     }
   }
}

template <typename FuncType> void loop_over_locally_owned_nodes(const stk::mesh::BulkData& bulk, FuncType func)
{
   for (const auto* ib : bulk.get_buckets(stk::topology::NODE_RANK, bulk.mesh_meta_data().locally_owned_part())) {
     for (size_t k = 0; k < ib->size(); ++k) {
       func((*ib)[k]);
     }
   }
}

void compute_dual_nodal_volume(const stk::mesh::BulkData& bulk, ScalarFieldType& dnvField)
{
  auto& meta = bulk.mesh_meta_data();
  const int nDim = meta.spatial_dimension();
  const auto& coordField =*static_cast<const VectorFieldType*>(meta.coordinate_field());

  const stk::mesh::Selector selector = meta.locally_owned_part();
  const stk::mesh::BucketVector& elem_buckets = bulk.get_buckets(stk::topology::ELEM_RANK, selector);

  for (const auto* ib : elem_buckets) {
    const auto& b = *ib;
    MasterElement& meSCV = *MasterElementRepo::get_volume_master_element(b.topology());
    const int nodesPerElement = meSCV.nodesPerElement_;
    const int numScvIp = meSCV.numIntPoints_;
    const int* ipNodeMap = meSCV.ipNodeMap();

    std::vector<double> ws_scv_volume(numScvIp);
    std::vector<double> ws_coordinates(nodesPerElement * nDim);

    for (size_t k = 0u; k < b.size(); ++k) {
      stk::mesh::Entity const* node_rels = bulk.begin_nodes(b[k]);
      for (int ni = 0; ni < nodesPerElement; ++ni) {
        const double* const coords = stk::mesh::field_data(coordField, node_rels[ni]);
        const int offSet = ni * nDim;
        for (int j = 0; j < nDim; ++j) {
          ws_coordinates[offSet + j] = coords[j];
        }
      }

      // compute integration point volume
      double scv_error = 1.0;
      meSCV.determinant(1, ws_coordinates.data(), ws_scv_volume.data(), &scv_error);

      // assemble dual volume while scattering ip volume
      for (int ip = 0; ip < numScvIp; ++ip) {
        *stk::mesh::field_data(dnvField, node_rels[ipNodeMap[ip]]) += ws_scv_volume[ip];
      }
    }
  }
  stk::mesh::parallel_sum(bulk, {&dnvField});
}


void compute_projected_nodal_gradient_interior(
  const stk::mesh::BulkData& bulk,
  const ScalarFieldType& dnvField,
  const ScalarFieldType& qField,
  VectorFieldType& dqdxField)
{
  auto& meta = bulk.mesh_meta_data();
  const int nDim = meta.spatial_dimension();
  const auto& coordField =*static_cast<const VectorFieldType*>(meta.coordinate_field());

  const stk::mesh::Selector selector = meta.locally_owned_part();
  const stk::mesh::BucketVector& elem_buckets = bulk.get_buckets(stk::topology::ELEM_RANK, selector);

  for (const auto* ib : elem_buckets) {
    const auto& b = *ib;
    MasterElement& meSCS = *MasterElementRepo::get_surface_master_element(b.topology());
    const int nodesPerElement = meSCS.nodesPerElement_;
    const int numScsIp = meSCS.numIntPoints_;

    std::vector<double> ws_scalar(nodesPerElement);
    std::vector<double> ws_dualVolume(nodesPerElement);
    std::vector<double> ws_coords(nDim*nodesPerElement);
    std::vector<double> ws_areav(nDim*numScsIp);
    std::vector<double> ws_dqdx(nDim*numScsIp);
    const auto* lrscv = meSCS.adjacentNodes();
    std::vector<double> ws_shape_function(nodesPerElement*numScsIp);
    meSCS.shape_fcn(ws_shape_function.data());

    for (size_t k = 0u; k < b.size(); ++k) {
      const auto* node_rels = bulk.begin_nodes(b[k]);

      for (int ni = 0; ni < nodesPerElement; ++ni) {
        stk::mesh::Entity node = node_rels[ni];
        ws_scalar[ni]     = *stk::mesh::field_data(qField, node);
        ws_dualVolume[ni] = *stk::mesh::field_data(dnvField, node);

        const double * coords = stk::mesh::field_data(coordField, node);
        const int offSet = ni*nDim;
        for ( int j=0; j < nDim; ++j ) {
          ws_coords[offSet+j] = coords[j];
        }
      }

      double scs_error = 0.0;
      meSCS.determinant(1, ws_coords.data(), ws_areav.data(), &scs_error);

      for (int ip = 0; ip < numScsIp; ++ip) {
        const int il = lrscv[2*ip];
        const int ir = lrscv[2*ip+1];

        double* gradQL = stk::mesh::field_data(dqdxField, node_rels[il]);
        double* gradQR = stk::mesh::field_data(dqdxField, node_rels[ir]);

        double qIp = 0.0;
        const int offSet = ip*nodesPerElement;
        for (int ic = 0; ic < nodesPerElement; ++ic) {
          qIp += ws_shape_function[offSet+ic]*ws_scalar[ic];
        }

        double inv_volL = 1.0/ws_dualVolume[il];
        double inv_volR = 1.0/ws_dualVolume[ir];

        for ( int j = 0; j < nDim; ++j ) {
          double fac = qIp*ws_areav[ip*nDim+j];
          gradQL[j] += fac*inv_volL;
          gradQR[j] -= fac*inv_volR;
        }
      }
    }
  }
}

void compute_projected_nodal_gradient_boundary(
  const stk::mesh::BulkData& bulk,
  const ScalarFieldType& dnvField,
  const ScalarFieldType& qField,
  VectorFieldType& dqdxField)
{
  auto& meta = bulk.mesh_meta_data();
  const int nDim = meta.spatial_dimension();
  const auto& coordField =*static_cast<const VectorFieldType*>(meta.coordinate_field());

  const stk::mesh::Selector selector =  meta.locally_owned_part();
  const stk::mesh::BucketVector& face_buckets = bulk.get_buckets(meta.side_rank(), selector);

  for (const auto* ib : face_buckets) {
    const auto& b = *ib;
    MasterElement& meFC = *MasterElementRepo::get_surface_master_element(b.topology());
    const int nodesPerFace= meFC.nodesPerElement_;
    const int numScsIp = meFC.numIntPoints_;

    std::vector<double> ws_scalar(nodesPerFace);
    std::vector<double> ws_dualVolume(nodesPerFace);
    std::vector<double> ws_coords(nDim*nodesPerFace);
    std::vector<double> ws_areav(nDim*numScsIp);
    std::vector<double> ws_dqdx(nDim*numScsIp);
    const auto* ipNodeMap = meFC.ipNodeMap();

    std::vector<double> ws_shape_function(nodesPerFace*numScsIp);
    meFC.shape_fcn(ws_shape_function.data());

    for (size_t k = 0u; k < b.size(); ++k) {
      const auto* face_node_rels = bulk.begin_nodes(b[k]);
      for (int ni = 0; ni < nodesPerFace; ++ni) {
        stk::mesh::Entity node = face_node_rels[ni];

        ws_scalar[ni]     = *stk::mesh::field_data(qField, node);
        ws_dualVolume[ni] = *stk::mesh::field_data(dnvField, node);

        const double * coords = stk::mesh::field_data(coordField, node);
        const int offSet = ni*nDim;
        for ( int j=0; j < nDim; ++j ) {
          ws_coords[offSet+j] = coords[j];
        }
      }

      double scs_error = 0.0;
      meFC.determinant(1, ws_coords.data(), ws_areav.data(), &scs_error);

      for (int ip = 0; ip < numScsIp; ++ip) {
        const int nn = ipNodeMap[ip];

        stk::mesh::Entity nodeNN = face_node_rels[nn];

        // pointer to fields to assemble
        double *gradQNN = stk::mesh::field_data(dqdxField, nodeNN);
        double volNN = *stk::mesh::field_data(dnvField, nodeNN);

        // interpolate to scs point; operate on saved off ws_field
        double qIp = 0.0;
        const int offSet = ip*nodesPerFace;
        for ( int ic = 0; ic < nodesPerFace; ++ic ) {
          qIp += ws_shape_function[offSet+ic]*ws_scalar[ic];
        }

        // nearest node volume
        double inv_volNN = 1.0/volNN;

        // assemble to nearest node
        for ( int j = 0; j < nDim; ++j ) {
          double fac = qIp*ws_areav[ip*nDim+j];
          gradQNN[j] += fac*inv_volNN;
        }
      }
    }
  }
}

void compute_projected_nodal_gradient(
  const stk::mesh::BulkData& bulk,
  ScalarFieldType& dnvField, ScalarFieldType& qField,
  VectorFieldType& dqdxField)
{
  compute_dual_nodal_volume(bulk, dnvField);
  compute_projected_nodal_gradient_interior(bulk, dnvField, qField, dqdxField);
  compute_projected_nodal_gradient_boundary(bulk, dnvField, qField, dqdxField);
  stk::mesh::parallel_sum(bulk, {&dqdxField});
}


void check_dual_nodal_volume_sum(stk::topology topo)
{
  const int nprocs = stk::parallel_machine_size(MPI_COMM_WORLD);
  const int nx = 8*nprocs;

  auto fixture = unit_test_utils::MeshFixture::create(topo, nx);
  auto& bulk = *fixture->bulk_;
  auto& meta = bulk.mesh_meta_data();

  auto& dnvField = meta.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "dual_nodal_volume");
  stk::mesh::put_field(dnvField, meta.universal_part(), 1);
  fixture->commit();

  stk::mesh::field_fill(0.0, dnvField);

  compute_dual_nodal_volume(bulk, dnvField);

  double local_volume_sum = 0;
  loop_over_locally_owned_nodes(bulk, [&](stk::mesh::Entity node) {
      local_volume_sum += *stk::mesh::field_data(dnvField, node);
  });

  double volume_sum = 0;
  stk::all_reduce_sum(bulk.parallel(), &local_volume_sum, &volume_sum, 1);

  EXPECT_NEAR(volume_sum, 1.0, 1.0e-12);
}

void check_projected_nodal_gradient_linear_is_exact(stk::topology topo)
{
  const int nprocs = stk::parallel_machine_size(MPI_COMM_WORLD);
  const int nx = 8*nprocs;
  auto fixture = unit_test_utils::MeshFixture::create(topo, nx);
  auto& bulk = *fixture->bulk_;
  auto& meta = bulk.mesh_meta_data();
  const int dim = meta.spatial_dimension();

  auto& dnvField = meta.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "dual_nodal_volume");
  stk::mesh::put_field(dnvField, meta.universal_part(), 1);

  auto& qField = meta.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "q");
  stk::mesh::put_field(qField, meta.universal_part(), 1);

  auto& dqdxField = meta.declare_field<VectorFieldType>(stk::topology::NODE_RANK, "dqdx");
  stk::mesh::put_field(dqdxField, meta.universal_part(), dim);

  auto& dqdxExactField = meta.declare_field<VectorFieldType>(stk::topology::NODE_RANK, "dqdx_exact");
  stk::mesh::put_field(dqdxExactField, meta.universal_part(), dim);

  auto& dqdxErrorField = meta.declare_field<VectorFieldType>(stk::topology::NODE_RANK, "dqdx_error");
  stk::mesh::put_field(dqdxErrorField, meta.universal_part(), dim);

  fixture->commit();

  stk::mesh::field_fill(0.0, dnvField);
  stk::mesh::field_fill(0.0, dqdxField);


  auto& coordField = *static_cast<const VectorFieldType*>(meta.coordinate_field());

  loop_over_all_non_aura_nodes(bulk, [&](stk::mesh::Entity node) {
    const double* coords = stk::mesh::field_data(coordField, node);

    *stk::mesh::field_data(qField, node) = coords[0] + coords[1];
    if (dim == 3) {
      *stk::mesh::field_data(qField, node) += coords[2];
    }
    double* dqdxExact = stk::mesh::field_data(dqdxExactField, node);
    dqdxExact[0] = 1;
    dqdxExact[1] = 1;
    if (dim == 3) {
      dqdxExact[2] = 1;
    }
  });

  compute_projected_nodal_gradient(bulk, dnvField, qField, dqdxField);

  double local_max_error= -1;
  loop_over_locally_owned_nodes(bulk, [&](stk::mesh::Entity node) {
    double* dqdx = stk::mesh::field_data(dqdxField, node);
    double* dqdxExact = stk::mesh::field_data(dqdxExactField, node);
    double* dqdxError = stk::mesh::field_data(dqdxErrorField, node);

    for (int d = 0; d < dim; ++d) {
      dqdxError[d] = dqdx[d] - dqdxExact[d];
      local_max_error = std::max(local_max_error, std::abs(dqdxError[d]));
    }
  });

  double max_error = 0;
  stk::all_reduce_max(bulk.parallel(), &local_max_error, &max_error, 1);

  // just report on rank 0
  if (bulk.parallel_rank() == 0) {
    EXPECT_NEAR(max_error, 0, 1.0e-12);
  }

  const bool dump_mesh = max_error > 1.0e-12;
  if (dump_mesh) {
    const std::string name = "png_test_" + topo.name() + "_" + std::to_string(nx) + ".e";
    unit_test_utils::dump_mesh(bulk, {&dnvField, &qField, &dqdxField, &dqdxExactField, &dqdxErrorField}, name);
  }
}

#define TEST_ALL_P1_TOPOS(x, y) \
    TEST(x, tri##_##y)   { y(stk::topology::TRI_3_2D); }   \
    TEST(x, quad4##_##y)  { y(stk::topology::QUAD_4_2D); } \
    TEST(x, tet##_##y)   { y(stk::topology::TET_4); }      \
    TEST(x, wedge##_##y) { y(stk::topology::WEDGE_6); }    \
    TEST(x, pyr##_##y) { y(stk::topology::PYRAMID_5); }    \
    TEST(x, hex8##_##y)   { y(stk::topology::HEX_8); }

TEST_ALL_P1_TOPOS(projected_nodal_gradient, check_dual_nodal_volume_sum);
TEST_ALL_P1_TOPOS(projected_nodal_gradient, check_projected_nodal_gradient_linear_is_exact);

}
}
