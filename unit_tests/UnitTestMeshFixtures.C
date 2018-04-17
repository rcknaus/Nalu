#include "UnitTestMeshFixtures.h"
#include "UnitTestUtils.h"

#include <NaluEnv.h>
#include <nalu_make_unique.h>

#include <stk_unit_tests/stk_mesh_fixtures/CoordinateMapping.hpp>
#include <stk_unit_tests/stk_mesh_fixtures/HexFixture.hpp>
#include <stk_unit_tests/stk_mesh_fixtures/TetFixture.hpp>
#include <stk_unit_tests/stk_mesh_fixtures/QuadFixture.hpp>
#include <stk_unit_tests/stk_mesh_fixtures/WedgeFixture.hpp>
#include <stk_unit_tests/stk_mesh_fixtures/PyramidFixture.hpp>
#include <stk_unit_tests/stk_mesh_fixtures/TriFixture.hpp>
#include <stk_io/StkMeshIoBroker.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/CoordinateSystems.hpp>
#include <stk_mesh/base/FieldBase.hpp>
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/FieldBLAS.hpp>
#include <stk_mesh/base/SkinBoundary.hpp>

namespace {

  typedef stk::mesh::Field<double> ScalarFieldType;
  typedef stk::mesh::Field<double,stk::mesh::Cartesian> VectorFieldType;

  template <typename FixtureType> class MeshFixtureHelper3D : public unit_test_utils::MeshFixture
  {
  public:
    MeshFixtureHelper3D(int n) :
      MeshFixture(),
      n_(n),
      fixture_(sierra::nalu::NaluEnv::self().parallel_comm(), n, n, n, stk::mesh::BulkData::NO_AUTO_AURA)
    {
      bulk_ = &fixture_.m_bulk_data;
    }

    void commit() {
      auto& meta = fixture_.m_meta;
      const stk::mesh::PartVector surfaces = { &meta.declare_part("all_surfaces", meta.side_rank()) };
      stk::io::put_io_part_attribute(*surfaces.front());
      stk::io::put_io_part_attribute(*fixture_.m_elem_parts.front());
      meta.commit();

      fixture_.generate_mesh(stk::mesh::fixtures::FixedCartesianCoordinateMapping(n_,n_,n_,1.0,1.0,1.0));
      stk::mesh::create_exposed_block_boundary_sides(meta.mesh_bulk_data(), meta.universal_part(), surfaces);
    }

    const int n_;
    FixtureType fixture_;
  };

  template <typename FixtureType> class MeshFixtureHelper2D : public unit_test_utils::MeshFixture
  {
  public:
    MeshFixtureHelper2D(int n) :
      MeshFixture(),
      n_(n),
      fixture_(sierra::nalu::NaluEnv::self().parallel_comm(), n, n, stk::mesh::BulkData::NO_AUTO_AURA)
    {
      bulk_ = &fixture_.m_bulk_data;
    }

    void commit() {
      auto& meta  =fixture_.m_meta;
      const stk::mesh::PartVector surfaces = { &meta.declare_part("all_surfaces", meta.side_rank()) };
      stk::io::put_io_part_attribute(*surfaces.front());
      stk::io::put_io_part_attribute(*fixture_.m_elem_parts.front());
      meta.commit();
      fixture_.generate_mesh();
      const double scale = 1.0/static_cast<double>(n_);
      stk::mesh::field_scale(scale, *static_cast<const VectorFieldType*>(fixture_.m_meta.coordinate_field()));
      stk::mesh::create_exposed_block_boundary_sides(meta.mesh_bulk_data(), meta.universal_part(), surfaces);
    }

    const int n_;
    FixtureType fixture_;
  };
}

namespace unit_test_utils
{
  std::unique_ptr<MeshFixture> MeshFixture::create(stk::topology topo, int n)
  {
    if (topo == stk::topology::HEX_8) {
      return sierra::nalu::make_unique<MeshFixtureHelper3D<stk::mesh::fixtures::HexFixture>>(n);
    }
    else if (topo == stk::topology::TET_4) {
      return sierra::nalu::make_unique<MeshFixtureHelper3D<stk::mesh::fixtures::TetFixture>>(n);
    }
    else if (topo == stk::topology::WEDGE_6) {
      return sierra::nalu::make_unique<MeshFixtureHelper3D<stk::mesh::fixtures::WedgeFixture>>(n);
    }
    else if (topo == stk::topology::PYRAMID_5) {
      return sierra::nalu::make_unique<MeshFixtureHelper3D<stk::mesh::fixtures::PyramidFixture>>(n);
    }
    else if (topo == stk::topology::QUAD_4_2D) {
      return sierra::nalu::make_unique<MeshFixtureHelper2D<stk::mesh::fixtures::QuadFixture>>(n);
    }
    else if (topo == stk::topology::TRI_3_2D) {
      return sierra::nalu::make_unique<MeshFixtureHelper2D<stk::mesh::fixtures::TriFixture>>(n);
    }
    else {
      unit_test_utils::nalu_out() << "Invalid topology" << std::endl;
      return nullptr;
    }
  }
}

