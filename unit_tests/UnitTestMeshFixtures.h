#ifndef UNITTESTMESHFIXTURES_H_
#define UNITTESTMESHFIXTURES_H_

#include <memory>

namespace stk { namespace mesh  { class BulkData; } }
namespace stk { namespace mesh  { class MetaData; } }
namespace stk { class topology; }

namespace unit_test_utils {

class MeshFixture
{
public:
  static std::unique_ptr<MeshFixture> create(stk::topology topo, int n);
  virtual ~MeshFixture() = default;
  virtual void commit() = 0;
  stk::mesh::BulkData* bulk_{nullptr};
protected:
  MeshFixture() = default;
};

}

#endif
