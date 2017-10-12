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

template <int p, typename Scalar = DoubleType>
struct CoefficientMatrices
{
  constexpr static int poly_order = p;

  CoefficientMatrices(const double* nodeLocs, const double* scsLocs)
  : scsDeriv(coefficients::scs_derivative_weights<p, Scalar>(nodeLocs, scsLocs)),
    scsInterp(coefficients::scs_interpolation_weights<p, Scalar>(nodeLocs, scsLocs)),
    nodalWeights(coefficients::nodal_integration_weights<p, Scalar>(nodeLocs, scsLocs)),
    nodalDeriv(coefficients::nodal_derivative_weights<p, Scalar>(nodeLocs)),
    linear_nodal_interp(coefficients::linear_nodal_interpolation_weights<p, Scalar>(nodeLocs)),
    linear_scs_interp(coefficients::linear_scs_interpolation_weights<p, Scalar>(scsLocs))
  {};

  CoefficientMatrices()
  : scsDeriv(coefficients::scs_derivative_weights<p, Scalar>()),
    scsInterp(coefficients::scs_interpolation_weights<p, Scalar>()),
    nodalWeights(coefficients::nodal_integration_weights<p, Scalar>()),
    nodalDeriv(coefficients::nodal_derivative_weights<p, Scalar>()),
    difference(coefficients::difference_matrix<p, Scalar>()),
    linear_nodal_interp(coefficients::linear_nodal_interpolation_weights<p, Scalar>()),
    linear_scs_interp(coefficients::linear_scs_interpolation_weights<p, Scalar>())
  {};

  const scs_matrix_view<p, Scalar> scsDeriv;
  const scs_matrix_view<p, Scalar> scsInterp;
  const nodal_matrix_view<p, Scalar> nodalWeights;
  const nodal_matrix_view<p, Scalar> nodalDeriv;
  const nodal_matrix_view<p, Scalar> difference;
  const linear_nodal_matrix_view<p, Scalar> linear_nodal_interp;
  const linear_scs_matrix_view<p, Scalar> linear_scs_interp;
};

namespace internal
{
template <int p, stk::topology::topology_t topo> struct nmap_t { using type = void; };
template <int p> struct nmap_t<p, stk::topology::QUAD_4_2D> { using type = node_map_view<AlgTraitsQuad<p>>; };
template <int p, stk::topology::topology_t topo> struct node_map_maker { static typename nmap_t<p,topo>::type make(); };

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

template <int p, stk::topology::topology_t topo>
typename internal::nmap_t<p, topo>::type make_node_map() { return internal::node_map_maker<p, topo>::make(); }

} // namespace naluUnit
} // namespace Sierra

#endif
