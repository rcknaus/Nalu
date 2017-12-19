#include <gtest/gtest.h>
#include <limits>

#include <stk_util/parallel/Parallel.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Bucket.hpp>
#include <stk_mesh/base/CoordinateSystems.hpp>
#include <stk_mesh/base/FieldBase.hpp>
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/GetEntities.hpp>

#include <master_element/MasterElement.h>
#include <master_element/Quad42DCVFEM.h>
#include <master_element/TensorOps.h>

#include <memory>
#include <random>

#include "UnitTestUtils.h"
#include "kernels/UnitTestKernelUtils.h"

#include "master_element/TensorOps.h"



namespace {

using VectorFieldType = stk::mesh::Field<double, stk::mesh::Cartesian>;
//-------------------------------------------------------------------------
double linear_scalar_value(int dim, double a, const double* b, const double* x)
{
  if (dim == 2u) {
    return (a + b[0] * x[0] + b[1] * x[1]);
  }
  return (a + b[0] * x[0] + b[1] * x[1] + b[2] * x[2]);
}
//-------------------------------------------------------------------------
struct LinearField
{
  LinearField(int in_dim, double in_a, const double* in_b) : dim(in_dim), a(in_a) {
    b[0] = in_b[0];
    b[1] = in_b[1];
    if (dim == 3) b[2] = in_b[2];
  }

  double operator()(const double* x) { return linear_scalar_value(dim, a, b, x); }

  const int dim;
  const double a;
  double b[3];
};

LinearField make_random_linear_field(int dim, std::mt19937& rng)
{

  std::uniform_real_distribution<double> coeff(-1.0, 1.0);
  std::vector<double> coeffs(dim);

  double a = coeff(rng);
  for (int j = 0; j < dim; ++j) {
    coeffs[j] = coeff(rng);
  }
  return LinearField(dim, a, coeffs.data());
}

void randomly_perturb_element_coords(
  std::mt19937& rng,
  int npe,
  const stk::mesh::Entity* node_rels,
  const VectorFieldType& coordField)
{
  int dim = coordField.max_size(stk::topology::NODE_RANK);
  auto rot = unit_test_utils::random_rotation_matrix(dim, rng);

  std::uniform_real_distribution<double> random_perturb(0.125, 0.25);

  std::vector<double> coords_rt(npe * dim, 0.0);
  for (int n = 0; n < npe; ++n) {
    const double* coords =  stk::mesh::field_data(coordField, node_rels[n]);
    for (int i = 0; i < dim; ++i) {
      for (int j = 0; j < dim; ++j) {
        coords_rt[n*dim + i] += rot[i*dim+j]*coords[j];
      }
    }
  }

  for (int n = 0; n < npe; ++n) {
    double* coords =  stk::mesh::field_data(coordField, node_rels[n]);
    for (int d = 0; d < dim; ++d) {
      coords[d] = coords_rt[n*dim+d] + random_perturb(rng);
    }
  }
}

}


namespace {

  void vector_field_2d(const double* coords, double* v) {
    double x = coords[0];
    double y = coords[1];

    v[0] = -y;
    v[1] = +x;
  }


  void vector_field_3d(const double* coords, double* v) {
    double x = coords[0];
    double y = coords[1];
    double z = coords[2];

    v[0] = -y + 3 * y * y + z + x;
    v[1] = x - z + 3 * z * z  - 2 * y;
    v[2] = -x + 3 * x * x + y + z;
  }

  double analytical_div(const double* coords) {
    return 0;
  }

  void check_element_divergence(
    stk::mesh::BulkData& bulk,
    const stk::mesh::Entity* elem_node_rels,
    std::vector<stk::mesh::Entity> face_elements,
    const VectorFieldType& coordField,
    sierra::nalu::MasterElement& meSCS,
    std::vector<sierra::nalu::MasterElement*> meFCs,
    ScalarFieldType& dnv,
    VectorFieldType& v,
    ScalarFieldType& divv)
  {
    if (meSCS.nDim_ != 3) { return; }
    int dim = meSCS.nDim_;
    std::vector<double> ws_coords(dim*meSCS.nodesPerElement_);
    for (int n = 0; n < meSCS.nodesPerElement_; ++n) {
      auto node = elem_node_rels[n];
      const double* coords = stk::mesh::field_data(coordField, node);
      for (int d = 0; d < dim; ++d) {
        ws_coords[dim*n+ d] = coords[d];
      }
      if (dim == 3) {
        vector_field_3d(coords, stk::mesh::field_data(v,node));
      }
      else {
        vector_field_2d(coords, stk::mesh::field_data(v,node));
      }

      *stk::mesh::field_data(divv, node) = 0;
    }

    // interior contribution
    std::vector<double> elem_shp(meSCS.numIntPoints_*meSCS.nodesPerElement_, 0.0);
    meSCS.shape_fcn(elem_shp.data());

    double error = 0;
    std::vector<double> areav(dim * meSCS.numIntPoints_, 0.0);
    meSCS.determinant(1, ws_coords.data(), areav.data(), &error);

    for (int ip = 0; ip < meSCS.numIntPoints_; ++ip) {
      double vip[3] = { };
      for (int n = 0; n < meSCS.nodesPerElement_; ++n) {
        double r = elem_shp[ip * meSCS.nodesPerElement_ + n];
        double* vval = stk::mesh::field_data(v, elem_node_rels[n]);

        for (int d = 0; d < dim; ++d) {
          vip[d] += r * vval[d];
        }
      }
      double val = 0;
      for (int d = 0; d < dim; ++d) {
        val += vip[d]*areav[ip*dim+d];
      }
      auto lNode = elem_node_rels[meSCS.lrscv_[2*ip+0]];
      auto rNode = elem_node_rels[meSCS.lrscv_[2*ip+1]];
      *stk::mesh::field_data(divv,lNode) += val / (*stk::mesh::field_data(dnv, lNode));
      *stk::mesh::field_data(divv,rNode) -= val / (*stk::mesh::field_data(dnv, rNode));
    }

    for (int n = 0; n < meSCS.nodesPerElement_; ++n) {
      const double divval = *stk::mesh::field_data(divv, elem_node_rels[n]);
    }

    // boundary contributions
    for (unsigned side_ordinal = 0; side_ordinal < face_elements.size(); ++side_ordinal) {
      const auto* face_node_rels = bulk.begin_nodes(face_elements[side_ordinal]);
      auto& meFC = *meFCs[side_ordinal];

      std::vector<double> face_shp(meFC.numIntPoints_*meFC.nodesPerElement_, 0.0);
      meFC.shape_fcn(face_shp.data());

      std::vector<double> ws_face_coords(meFC.nodesPerElement_*3);
      for (int n = 0; n < meFC.nodesPerElement_; ++n) {
        auto node = face_node_rels[n];
        const double* coords = stk::mesh::field_data(coordField, node);
        for (int d = 0; d < dim; ++d) {
          ws_face_coords[dim*n+ d] = coords[d];
        }
      }

      std::vector<double> face_areav(dim * meFC.numIntPoints_, 0.0);
      meFC.determinant(1,ws_face_coords.data(), face_areav.data(), &error);
      for (int ip = 0; ip < meFC.numIntPoints_; ++ip) {
        double vip[3] = {};
        for (int n = 0; n < meFC.nodesPerElement_; ++n) {
          double r = face_shp[ip*meFC.nodesPerElement_ + n];
          const double* vval = stk::mesh::field_data(v, face_node_rels[n]);

          for (int d = 0; d < dim; ++d) {
            vip[d] += r * vval[d];
          }
        }

        double val = 0;
        for (int d = 0; d < dim; ++d) {
          val += vip[d]*face_areav[ip*dim+d];
        }
        const int nn = meFC.ipNodeMap_[ip];
        *stk::mesh::field_data(divv, face_node_rels[nn]) += val /(*stk::mesh::field_data(dnv, face_node_rels[nn]));
      }
    }


    for (int n = 0; n < meSCS.nodesPerElement_; ++n) {
      const double divval = *stk::mesh::field_data(divv, elem_node_rels[n]);
      const double expected_div = analytical_div(stk::mesh::field_data(coordField, elem_node_rels[n]));
      EXPECT_NEAR(divval, expected_div, 1.0e-12);
    }

  }

  void check_side_is_in_element(
    const stk::mesh::Entity* elem_node_rels,
    const stk::mesh::Entity* face_node_rels,
    const VectorFieldType& coordField,
    sierra::nalu::MasterElement& meSCS,
    sierra::nalu::MasterElement& meFC)
  {
    std::mt19937 rng;
    rng.seed(std::random_device()());
    std::uniform_real_distribution<double> random_pt(-0.1,1.1); // can be outside of element side
    std::vector<double> sideIsoPoint = { random_pt(rng), random_pt(rng) };
    randomly_perturb_element_coords(rng, meSCS.nodesPerElement_, elem_node_rels, coordField);

    int dim = meSCS.nDim_;
    std::vector<double> sideInterpWeights(meFC.numIntPoints_ * meFC.nodesPerElement_, 0.0);
    meFC.shape_fcn(sideInterpWeights.data());

    std::vector<double> faceCoords(dim * meFC.nodesPerElement_);
    for (int d = 0; d < dim; ++d) {
      for (int n = 0; n < meFC.nodesPerElement_; ++n) {
        stk::mesh::Entity face_node = face_node_rels[n];
        double* coords =  stk::mesh::field_data(coordField, face_node);
        faceCoords[d*meFC.nodesPerElement_ + n] = coords[d];
      }
    }

    // interpolat to the parametric coordinates
    std::vector<double> faceInterpCoords(dim, 0.0);
    meFC.interpolatePoint(dim, sideIsoPoint.data(), faceCoords.data(), faceInterpCoords.data());

    // check if isInElement can reproduce the original parametric coords
    std::vector<double> isoFacePoint(dim , 0.0);
    meFC.isInElement(faceCoords.data(), faceInterpCoords.data(), isoFacePoint.data());

    for (int d = 0; d < dim-1; ++d) {
      EXPECT_NEAR(isoFacePoint[d], sideIsoPoint[d], 1.0e-8);
    }
  }

  void check_side_is_not_in_element(
    const stk::mesh::Entity* elem_node_rels,
    const stk::mesh::Entity* face_node_rels,
    const VectorFieldType& coordField,
    sierra::nalu::MasterElement& meSCS,
    sierra::nalu::MasterElement& meFC)
  {
    std::mt19937 rng;
    rng.seed(std::random_device()());
    randomly_perturb_element_coords(rng, meSCS.nodesPerElement_, elem_node_rels, coordField);

    std::vector<double> point_coords = { 10.0, 10.0, 10.0 };

    int dim = meSCS.nDim_;
    std::vector<double> sideInterpWeights(meFC.numIntPoints_ * meFC.nodesPerElement_, 0.0);
    meFC.shape_fcn(sideInterpWeights.data());

    std::vector<double> faceCoords(dim * meFC.nodesPerElement_);
    for (int d = 0; d < dim; ++d) {
      for (int n = 0; n < meFC.nodesPerElement_; ++n) {
        stk::mesh::Entity face_node = face_node_rels[n];
        double* coords =  stk::mesh::field_data(coordField, face_node);
        faceCoords[d*meFC.nodesPerElement_ + n] = coords[d];
      }
    }

    // check if isInElement can reproduce the original parametric coords
    std::vector<double> isoFacePoint(dim , 0.0);
    double dist = meFC.isInElement(faceCoords.data(), point_coords.data(), isoFacePoint.data());

    EXPECT_GT(dist, 1.0+1.0e-6);
  }


}


class BoundaryMasterElement : public ::testing::Test
{
protected:
    BoundaryMasterElement() : comm(MPI_COMM_WORLD) {}

    void choose_topo(stk::topology topo)
    {
      bulk.reset();
      meta = std::unique_ptr<stk::mesh::MetaData>(new stk::mesh::MetaData(topo.dimension()));
      bulk = std::unique_ptr<stk::mesh::BulkData>(new stk::mesh::BulkData(*meta, comm));

      dnv = &meta->declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "dual_nodal_volume");
      stk::mesh::put_field(*dnv, meta->universal_part(), 1);

      v = &meta->declare_field<VectorFieldType>(stk::topology::NODE_RANK, "v");
      stk::mesh::put_field(*v, meta->universal_part(), meta->spatial_dimension());

      divv = &meta->declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "divv");
      stk::mesh::put_field(*divv, meta->universal_part(), 1);

      elem = unit_test_utils::create_one_reference_element(*bulk, topo);
      meSS = sierra::nalu::MasterElementRepo::get_surface_master_element(topo);
      meSV = sierra::nalu::MasterElementRepo::get_volume_master_element(topo);


      face_elements.resize(topo.num_sides());
      for (const auto* ib : bulk->get_buckets(meta->side_rank(), meta->universal_part())) {
        const auto& b = *ib;
        for (size_t k = 0u; k < b.size(); ++k) {
          auto side = b[k];
          const auto side_ordinal = bulk->begin_element_ordinals(side)[0];
          face_elements[side_ordinal] = side;
        }
      }

      meFCs.resize(topo.num_sides());
      for (unsigned side_ordinal = 0; side_ordinal < topo.num_sides(); ++side_ordinal) {
        auto side_topo = topo.side_topology(side_ordinal);
        meFCs[side_ordinal] = sierra::nalu::MasterElementRepo::get_surface_master_element(side_topo);
      }
    }

    void element_divergence(stk::topology topo) {
      choose_topo(topo);
      unit_test_kernel_utils::calc_dual_nodal_volume(*bulk, topo, coordinate_field(), *dnv);
      auto elem_node_rels = bulk->begin_nodes(elem);
      check_element_divergence(*bulk, elem_node_rels, face_elements, coordinate_field(), *meSS, meFCs, *dnv, *v, *divv);
    }

    void side_is_in_element(stk::topology topo) {
      for (unsigned side_ordinal = 0; side_ordinal < topo.num_sides(); ++side_ordinal) {
        choose_topo(topo);
        auto elem_node_rels = bulk->begin_nodes(elem);
        auto face_node_rels = bulk->begin_nodes(face_elements[side_ordinal]);
        check_side_is_in_element(elem_node_rels, face_node_rels, coordinate_field(), *meSS, *meFCs[side_ordinal]);
      }
    }

    void side_is_not_in_element(stk::topology topo) {
      for (unsigned side_ordinal = 0; side_ordinal < topo.num_sides(); ++side_ordinal) {
        choose_topo(topo);
        auto elem_node_rels = bulk->begin_nodes(elem);
        auto face_node_rels = bulk->begin_nodes(face_elements[side_ordinal]);
        check_side_is_in_element(elem_node_rels, face_node_rels, coordinate_field(), *meSS, *meFCs[side_ordinal]);
      }
    }

    const VectorFieldType& coordinate_field() const {
      return *static_cast<const VectorFieldType*>(meta->coordinate_field());
    }


    stk::ParallelMachine comm;
    std::unique_ptr<stk::mesh::MetaData> meta;
    std::unique_ptr<stk::mesh::BulkData> bulk;
    stk::mesh::Entity elem;
    sierra::nalu::MasterElement* meSS;
    sierra::nalu::MasterElement* meSV;
    std::vector<stk::mesh::Entity> face_elements;
    std::vector<sierra::nalu::MasterElement*> meFCs;

    ScalarFieldType* dnv;
    VectorFieldType* v;
    ScalarFieldType* divv;
};

TEST_F_ALL_TOPOS_NO_PYR(BoundaryMasterElement, element_divergence);
TEST_F_ALL_TOPOS_NO_PYR(BoundaryMasterElement, side_is_in_element);
TEST_F_ALL_TOPOS_NO_PYR(BoundaryMasterElement, side_is_not_in_element);

