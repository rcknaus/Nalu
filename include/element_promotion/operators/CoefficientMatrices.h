/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level NaluUnit      */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/
#ifndef CoefficientMatrices_h
#define CoefficientMatrices_h

#include <element_promotion/operators/HighOrderCoefficients.h>
#include <element_promotion/ElementDescription.h>

namespace sierra {
namespace nalu{

template <int p>
struct CoefficientMatrices
{
  constexpr static int poly_order = p;

  CoefficientMatrices(const double* nodeLocs, const double* scsLocs)
  : scsDeriv(coefficients::scs_derivative_weights<poly_order>(nodeLocs, scsLocs)),
    scsInterp(coefficients::scs_interpolation_weights<poly_order>(nodeLocs, scsLocs)),
    nodalWeights(coefficients::nodal_integration_weights<poly_order>(nodeLocs, scsLocs)),
    nodalDeriv(coefficients::nodal_derivative_weights<poly_order>(nodeLocs)),
    linear_nodal_interp(coefficients::linear_nodal_interpolation_weights<poly_order>(nodeLocs)),
    linear_scs_interp(coefficients::linear_scs_interpolation_weights<poly_order>(scsLocs))
  {};

  CoefficientMatrices()
  : scsDeriv(coefficients::scs_derivative_weights<poly_order>()),
    scsInterp(coefficients::scs_interpolation_weights<poly_order>()),
    nodalWeights(coefficients::nodal_integration_weights<poly_order>()),
    nodalDeriv(coefficients::nodal_derivative_weights<poly_order>()),
    difference(coefficients::difference_matrix<poly_order>()),
    linear_nodal_interp(coefficients::linear_nodal_interpolation_weights<poly_order>()),
    linear_scs_interp(coefficients::linear_scs_interpolation_weights<poly_order>())
  {};

  const scs_matrix_view<p> scsDeriv;
  const scs_matrix_view<p> scsInterp;
  const nodal_matrix_view<p> nodalWeights;
  const nodal_matrix_view<p> nodalDeriv;
  const nodal_matrix_view<p> difference;
  const linear_nodal_matrix_view<p> linear_nodal_interp;
  const linear_scs_matrix_view<p> linear_scs_interp;
};

namespace internal
{
using topo_t = stk::topology::topology_t;
template <int p, topo_t topo> struct nmap_t { using type = void; };
template <int p> struct nmap_t<p, stk::topology::QUAD_4_2D> { using type = node_map_view<AlgTraitsQuad<p>>; };
template <int p, topo_t topo> struct node_map_maker { static typename nmap_t<p,topo>::type make(); };

template <int p> struct node_map_maker<p, stk::topology::QUAD_4_2D>
{
  static typename nmap_t<p, stk::topology::QUAD_4_2D>::type make()
  {
    node_map_view<AlgTraitsQuad<p>> node_map{"node_map"};
    auto desc = ElementDescription::create(AlgTraitsQuad<p>::nDim_, AlgTraitsQuad<p>::polyOrder_);
    for (int j = 0; j < AlgTraitsQuad<p>::nodes1D_; ++j) {
      for (int i = 0; i < AlgTraitsQuad<p>::nodes1D_; ++i) {
        node_map(j*AlgTraitsQuad<p>::nodes1D_+i) = desc->node_map(i,j);
      }
    }
    return node_map;
  }
};

}

template <int p, stk::topology::topology_t topo> typename internal::nmap_t<p, topo>::type
make_node_map()
{
  return internal::node_map_maker<p, topo>::make();
}

} // namespace naluUnit
} // namespace Sierra

#endif
