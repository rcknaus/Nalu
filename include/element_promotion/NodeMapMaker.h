/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level NaluUnit      */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/
#ifndef NodeMapMaker_h
#define NodeMapMaker_h

#include <element_promotion/operators/HighOrderCoefficients.h>
#include <element_promotion/ElementDescription.h>

#include <BuildTemplates.h>

namespace sierra {
namespace nalu{

namespace internal
{
template <int p, stk::topology::topology_t topo> struct nmap_t { using type = void; };
template <int p> struct nmap_t<p, stk::topology::QUAD_4_2D> { using type = node_map_view<AlgTraitsQuad<p>>; };
template <int p, stk::topology::topology_t topo> struct node_map_maker { static typename nmap_t<p,topo>::type make(); };

template <int p> struct node_map_maker<p, stk::topology::QUAD_4_2D>
{
  static typename nmap_t<p, stk::topology::QUAD_4_2D>::type make()
  {
    using AlgTraits = AlgTraitsQuad<p>;

    node_map_view<AlgTraits> node_map{"node_map"};
    auto desc = ElementDescription::create(AlgTraits::nDim_, AlgTraits::polyOrder_);
    for (int j = 0; j < AlgTraits::nodes1D_; ++j) {
      for (int i = 0; i < AlgTraits::nodes1D_; ++i) {
        node_map(j*AlgTraits::nodes1D_+i) = desc->node_map(i,j);
      }
    }
    return node_map;
  }
};

template <int p> struct node_map_maker<p, stk::topology::HEX_8>
{
  static typename nmap_t<p, stk::topology::HEX_8>::type make()
  {
    using AlgTraits = AlgTraitsHex<p>;

    node_map_view<AlgTraits> node_map{"node_map"};
    auto desc = ElementDescription::create(AlgTraits::nDim_, AlgTraits::polyOrder_);
    for (int k = 0; k < AlgTraits::nodes1D_; ++k) {
      for (int j = 0; j < AlgTraits::nodes1D_; ++j) {
        for (int i = 0; i < AlgTraits::nodes1D_; ++i) {
          node_map(k*AlgTraits::nodes1D_*AlgTraits::nodes1D_ + j*AlgTraits::nodes1D_+i) = desc->node_map(i,j,k);
        }
      }
    }
    return node_map;
  }
};

}

template <int p, stk::topology::topology_t topo>
typename internal::nmap_t<p, topo>::type make_node_map() { return internal::node_map_maker<p, topo>::make(); }

inline Kokkos::View<int*> make_node_map(int p, stk::topology::topology_t topo)
{
  if (topo == stk::topology::QUAD_4_2D) {
    Kokkos::View<int*> dynamic_node_map("node_map", (p+1)*(p+1));
    switch (p)
    {
      case 2:
      {
        auto static_node_map = make_node_map<2, stk::topology::QUAD_4_2D>();
        Kokkos::deep_copy(dynamic_node_map, static_node_map);
        return dynamic_node_map;
      }
      case 3:
      {
        auto static_node_map = make_node_map<3, stk::topology::QUAD_4_2D>();
        Kokkos::deep_copy(dynamic_node_map, static_node_map);
        return dynamic_node_map;
      }
      case 4:
      {
        auto static_node_map = make_node_map<4, stk::topology::QUAD_4_2D>();
        Kokkos::deep_copy(dynamic_node_map, static_node_map);
        return dynamic_node_map;
      }
      case USER_POLY_ORDER:
      {

        auto static_node_map = make_node_map<USER_POLY_ORDER, stk::topology::QUAD_4_2D>();
        Kokkos::deep_copy(dynamic_node_map, static_node_map);
        return dynamic_node_map;
      }
    }
  }
  return Kokkos::View<int*>{""};
}


} // namespace naluUnit
} // namespace Sierra

#endif
