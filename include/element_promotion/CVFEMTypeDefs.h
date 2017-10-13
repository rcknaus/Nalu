/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef CVFEMTypeDefs_h
#define CVFEMTypeDefs_h

#include <SupplementalAlgorithm.h>
#include <AlgTraits.h>
#include <FieldTypeDef.h>

#include <stk_mesh/base/Entity.hpp>

#include <SimdInterface.h>
#include <KokkosInterface.h>

namespace sierra { namespace nalu {

  using default_float_type = DoubleType;

  template <typename ArrayType>
  using ViewType = SharedMemView<ArrayType>;

  template <typename ArrayType>
  using CoeffViewType = Kokkos::View<ArrayType>;

 //--------------------------------------------------------------------------
  template <typename AlgTraits, typename Scalar = default_float_type>
  using nodal_scalar_array = Scalar[AlgTraits::nodes1D_][AlgTraits::nodes1D_];
  
  template <typename AlgTraits, typename Scalar = default_float_type>
  using nodal_scalar_view = ViewType<nodal_scalar_array<AlgTraits,Scalar>>;
  //--------------------------------------------------------------------------
  template <typename AlgTraits, typename Scalar = default_float_type>
  using nodal_vector_array = Scalar[AlgTraits::nDim_][AlgTraits::nodes1D_][AlgTraits::nodes1D_];

  template <typename AlgTraits, typename Scalar = default_float_type>
  using nodal_vector_view = ViewType<nodal_vector_array<AlgTraits,Scalar>>;
  //--------------------------------------------------------------------------
  template <typename AlgTraits, typename Scalar = default_float_type>
  using nodal_tensor_array = Scalar[AlgTraits::nDim_][AlgTraits::nDim_][AlgTraits::nodes1D_][AlgTraits::nodes1D_];

  template <typename AlgTraits, typename Scalar = default_float_type>
  using nodal_tensor_view = ViewType<nodal_tensor_array<AlgTraits,Scalar>>;
  //--------------------------------------------------------------------------
  template <typename AlgTraits, typename Scalar = default_float_type>
  using scs_scalar_array = Scalar[AlgTraits::nscs_][AlgTraits::nodes1D_];

  template <typename AlgTraits, typename Scalar = default_float_type>
  using scs_scalar_view = ViewType<scs_scalar_array<AlgTraits,Scalar>>;
  //--------------------------------------------------------------------------
  template <typename AlgTraits, typename Scalar = default_float_type>
  using scs_vector_array = Scalar[AlgTraits::nDim_][AlgTraits::nscs_][AlgTraits::nodes1D_];

  template <typename AlgTraits, typename Scalar = default_float_type>
  using scs_vector_view = ViewType<scs_vector_array<AlgTraits,Scalar>>;
  //--------------------------------------------------------------------------
  template <typename AlgTraits, typename Scalar = default_float_type>
  using scs_tensor_array = Scalar[AlgTraits::nDim_][AlgTraits::nDim_][AlgTraits::nscs_][AlgTraits::nodes1D_];

  template <typename AlgTraits, typename Scalar = default_float_type>
  using scs_tensor_view = ViewType<scs_tensor_array<AlgTraits,Scalar>>;
  //--------------------------------------------------------------------------
  template <typename AlgTraits, typename Scalar = default_float_type>
  using matrix_array = Scalar[AlgTraits::nodesPerElement_][AlgTraits::nodesPerElement_];

  template <typename AlgTraits, typename Scalar = default_float_type>
  using matrix_view = ViewType<matrix_array<AlgTraits,Scalar>>;
  //--------------------------------------------------------------------------
  template <int poly_order, typename Scalar = default_float_type>
  using nodal_matrix_array = Scalar[poly_order+1][poly_order+1];

  template <int poly_order, typename Scalar = default_float_type>
  using nodal_matrix_view = CoeffViewType<nodal_matrix_array<poly_order, Scalar>>;
  //--------------------------------------------------------------------------
  template <int poly_order, typename Scalar = default_float_type>
  using scs_matrix_array = Scalar[poly_order+1][poly_order+1]; // always pad to be square

  template <int poly_order, typename Scalar = default_float_type>
  using scs_matrix_view = CoeffViewType<scs_matrix_array<poly_order,Scalar>>;
  //--------------------------------------------------------------------------
  template <int poly_order, typename Scalar = default_float_type>
  using linear_nodal_matrix_array = Scalar[2][poly_order+1];

  template <int poly_order, typename Scalar = default_float_type>
  using linear_nodal_matrix_view = CoeffViewType<linear_nodal_matrix_array<poly_order,Scalar>>;
  //--------------------------------------------------------------------------
  template <int poly_order, typename Scalar = default_float_type>
  using linear_scs_matrix_array = Scalar[2][poly_order];

  template <int poly_order, typename Scalar = default_float_type>
  using linear_scs_matrix_view = CoeffViewType<linear_scs_matrix_array<poly_order, Scalar>>;
  //--------------------------------------------------------------------------
  template <typename AlgTraits, typename IntegralScalar = int>
  using node_map_array = IntegralScalar[AlgTraits::nodesPerElement_];

  template <typename AlgTraits, typename IntegralScalar = int>
  using node_map_view = CoeffViewType<node_map_array<AlgTraits,IntegralScalar>>;

} // namespace nalu
} // namespace Sierra

#endif
