/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef AlgTraits_h
#define AlgTraits_h

#include <stk_topology/topology.hpp>

namespace sierra {
namespace nalu {

// limited supported now (P=1 3D elements)
struct AlgTraitsHex8 {
  static constexpr int nDim_ = 3;
  static constexpr int nodesPerElement_ = 8;
  static constexpr int numScsIp_ = 12;
  static constexpr int numScvIp_ = 8;
  static constexpr int numGp_ = 8; // for FEM
  static constexpr stk::topology::topology_t topo_ = stk::topology::HEX_8;
};

struct AlgTraitsHex27 {
  static constexpr int nDim_ = 3;
  static constexpr int nodesPerElement_ = 27;
  static constexpr int numScsIp_ = 216;
  static constexpr int numScvIp_ = 216;
  static constexpr int numGp_ = 27; // for FEM (not supported)
  static constexpr stk::topology::topology_t topo_ = stk::topology::HEX_27;
};

struct AlgTraitsTet4 {
  static constexpr int nDim_ = 3;
  static constexpr int nodesPerElement_ = 4;
  static constexpr int numScsIp_ = 6;
  static constexpr int numScvIp_ = 4;
  static constexpr int numGp_ = 4; // for FEM (not supported)
  static constexpr stk::topology::topology_t topo_ = stk::topology::TET_4;
};

struct AlgTraitsPyr5 {
  static constexpr int nDim_ = 3;
  static constexpr int nodesPerElement_ = 5;
  static constexpr int numScsIp_ = 8;
  static constexpr int numScvIp_ = 5;
  static constexpr int numGp_ = 5; // for FEM (not supported)
  static constexpr stk::topology::topology_t topo_ = stk::topology::PYRAMID_5;
};

struct AlgTraitsWed6 {
  static constexpr int nDim_ = 3;
  static constexpr int nodesPerElement_ = 6;
  static constexpr int numScsIp_ = 9;
  static constexpr int numScvIp_ = 6;
  static constexpr int numGp_ = 6; // for FEM (not supported)
  static constexpr stk::topology::topology_t topo_ = stk::topology::WEDGE_6;
};


struct AlgTraitsQuad4_2D {
  static constexpr int nDim_ = 2;
  static constexpr int nodesPerElement_ = 4;
  static constexpr int numScsIp_ = 4;
  static constexpr int numScvIp_ = 4;
  static constexpr int numGp_ = 4; // for FEM (not supported)
  static constexpr stk::topology::topology_t topo_ = stk::topology::QUAD_4_2D;
};

struct AlgTraitsQuad9_2D {
  static constexpr int nDim_ = 2;
  static constexpr int nodesPerElement_ = 9;
  static constexpr int numScsIp_ = 24;
  static constexpr int numScvIp_ = 36;
  static constexpr int numGp_ = 9; // for FEM (not supported)
  static constexpr stk::topology::topology_t topo_ = stk::topology::QUAD_9_2D;
};

struct AlgTraitsTri3_2D {
  static constexpr int nDim_ = 2;
  static constexpr int nodesPerElement_ = 3;
  static constexpr int numScsIp_ = 3;
  static constexpr int numScvIp_ = 3;
  static constexpr int numGp_ = 3; // for FEM (not supported)
  static constexpr stk::topology::topology_t topo_ = stk::topology::TRI_3_2D;
};

template <int p> constexpr int nGL() { return (p % 2 == 0) ? p / 2 + 1 : (p + 1) / 2; }

template <int p>
struct AlgTraitsQuadGL {
  static constexpr int nDim_ = 3;
  static constexpr int nodesPerElement_ = (p+1) * (p+1);
  static constexpr int numScsIp_ = 2 * p * (p + 1) * nGL<p>();
  static constexpr int numScvIp_ = nodesPerElement_ * nGL<p>() * nGL<p>();
  static constexpr int numGp_ = nodesPerElement_; // for FEM (not supported)

  static constexpr stk::topology::topology_t topo_ = static_cast<stk::topology::topology_t>(
    nodesPerElement_ + stk::topology::SUPERELEMENT_START
  );
  static constexpr stk::topology::topology_t baseTopo_ = stk::topology::QUAD_4_2D;
};

template <int p>
struct AlgTraitsHexGL {
  static constexpr int nDim_ = 3;
  static constexpr int nodesPerElement_ = (p+1)*(p+1)*(p+1);
  static constexpr int numScsIp_ = 3 * p * (p+1) * (p+1) * nGL<p>() * nGL<p>();
  static constexpr int numScvIp_ = nodesPerElement_ * nGL<p>() * nGL<p>() * nGL<p>();
  static constexpr int numGp_ = nodesPerElement_; // for FEM (not supported)

  static constexpr stk::topology::topology_t topo_ = static_cast<stk::topology::topology_t>(
    nodesPerElement_ + stk::topology::SUPERELEMENT_START
  );
  static constexpr stk::topology::topology_t baseTopo_ = stk::topology::HEX_8;
};


template <int p>
struct AlgTraitsQuad
{
  static constexpr int nDim_ = 2;
  static constexpr int nodesPerElement_ = (p+1) * (p+1);
  static constexpr int numScsIp_ = 2 * p * (p+1);
  static constexpr int numScvIp_ = (p+1) * (p+1);
  static constexpr int numGp_ = (p+1) * (p+1);
  static constexpr stk::topology::topology_t topo_ =
      static_cast<stk::topology::topology_t>(stk::topology::SUPERELEMENT_START + nodesPerElement_);

  // some higher order information
  static constexpr int polyOrder_ = p;
  static constexpr int nodes1D_ = p + 1;
  static constexpr int nscs_ = p;
  static constexpr stk::topology::topology_t baseTopo_ = stk::topology::QUAD_4_2D;
};



//template <stk::topology::topology_t stk_topo> struct TopoTraits;
//template <> struct TopoTraits<stk::topology::TRIANGLE_3_2D> { using AlgTraits = AlgTraitsTri3_2D; };
//template <> struct TopoTraits<stk::topology::QUADRILATERAL_4_2D> { using AlgTraits = AlgTraitsQuad4_2D; };
//template <> struct TopoTraits<stk::topology::TET_4> { using AlgTraits = AlgTraitsTet4; };
//template <> struct TopoTraits<stk::topology::PYRAMID_5> { using AlgTraits = AlgTraitsPyr5; };
//template <> struct TopoTraits<stk::topology::WEDGE_6> { using AlgTraits = AlgTraitsWed6; };
//template <> struct TopoTraits<stk::topology::HEX_8> { using AlgTraits = AlgTraitsHex8; };
//template <stk::topology::topology_t stk_topo> using AlgTraits = typename TopoTraits<stk_topo>::AlgTraits;


} // namespace nalu
} // namespace Sierra

#endif
