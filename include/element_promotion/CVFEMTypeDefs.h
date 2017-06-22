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

// Kokkos
#include <stk_simd_view/simd_view.hpp>
#include <KokkosInterface.h>

namespace sierra { namespace nalu {

  using default_float_type = double;

  template <typename ArrayType>
  using ViewType = Kokkos::View<ArrayType>;

 //--------------------------------------------------------------------------
  template <typename ElemTraits, typename Scalar = default_float_type>
  using nodal_scalar_array = Scalar[ElemTraits::nodes1D_][ElemTraits::nodes1D_];
  
  template <typename ElemTraits, typename Scalar = default_float_type>
  using nodal_scalar_view = ViewType<nodal_scalar_array<ElemTraits,Scalar>>;
  //--------------------------------------------------------------------------
  template <typename ElemTraits, typename Scalar = default_float_type>
  using nodal_vector_array = Scalar[ElemTraits::nDim_][ElemTraits::nodes1D_][ElemTraits::nodes1D_];

  template <typename ElemTraits, typename Scalar = default_float_type>
  using nodal_vector_view = ViewType<nodal_vector_array<ElemTraits,Scalar>>;
  //--------------------------------------------------------------------------
  template <typename ElemTraits, typename Scalar = default_float_type>
  using nodal_tensor_array = Scalar[ElemTraits::nDim_][ElemTraits::nDim_][ElemTraits::nodes1D_][ElemTraits::nodes1D_];

  template <typename ElemTraits, typename Scalar = default_float_type>
  using nodal_tensor_view = ViewType<nodal_tensor_array<ElemTraits,Scalar>>;
  //--------------------------------------------------------------------------
  template <typename ElemTraits, typename Scalar = default_float_type>
  using scs_scalar_array = Scalar[ElemTraits::nscs_][ElemTraits::nodes1D_];

  template <typename ElemTraits, typename Scalar = default_float_type>
  using scs_scalar_view = ViewType<scs_scalar_array<ElemTraits,Scalar>>;
  //--------------------------------------------------------------------------
  template <typename ElemTraits, typename Scalar = default_float_type>
  using scs_vector_array = Scalar[ElemTraits::nDim_][ElemTraits::nscs_][ElemTraits::nodes1D_];

  template <typename ElemTraits, typename Scalar = default_float_type>
  using scs_vector_view = ViewType<scs_vector_array<ElemTraits,Scalar>>;
  //--------------------------------------------------------------------------
  template <typename ElemTraits, typename Scalar = default_float_type>
  using scs_tensor_array = Scalar[ElemTraits::nDim_][ElemTraits::nDim_][ElemTraits::nscs_][ElemTraits::nodes1D_];

  template <typename ElemTraits, typename Scalar = default_float_type>
  using scs_tensor_view = ViewType<scs_tensor_array<ElemTraits,Scalar>>;
  //--------------------------------------------------------------------------
  template <typename ElemTraits, typename Scalar = default_float_type>
  using matrix_array = Scalar[ElemTraits::nodesPerElement_][ElemTraits::nodesPerElement_];

  template <typename ElemTraits, typename Scalar = default_float_type>
  using matrix_view = ViewType<matrix_array<ElemTraits,Scalar>>;
  //--------------------------------------------------------------------------
  template <int poly_order, typename Scalar = default_float_type>
  using nodal_matrix_array = Scalar[poly_order+1][poly_order+1];

  template <int poly_order, typename Scalar = default_float_type>
  using nodal_matrix_view = ViewType<nodal_matrix_array<poly_order, Scalar>>;
  //--------------------------------------------------------------------------
  template <int poly_order, typename Scalar = default_float_type>
  using scs_matrix_array = Scalar[poly_order+1][poly_order+1]; // always pad to be square

  template <int poly_order, typename Scalar = default_float_type>
  using scs_matrix_view = ViewType<scs_matrix_array<poly_order,Scalar>>;
  //--------------------------------------------------------------------------
  template <int poly_order, typename Scalar = default_float_type>
  using linear_nodal_matrix_array = Scalar[2][poly_order+1];

  template <int poly_order, typename Scalar = default_float_type>
  using linear_nodal_matrix_view = ViewType<linear_nodal_matrix_array<poly_order,Scalar>>;
  //--------------------------------------------------------------------------
  template <int poly_order, typename Scalar = default_float_type>
  using linear_scs_matrix_array = Scalar[2][poly_order];

  template <int poly_order, typename Scalar = default_float_type>
  using linear_scs_matrix_view = ViewType<linear_scs_matrix_array<poly_order, Scalar>>;
  //--------------------------------------------------------------------------
} // namespace nalu
} // namespace Sierra

#endif
