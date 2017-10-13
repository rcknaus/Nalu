/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level NaluUnit      */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/
#ifndef HighOrderOperatorsQuadInternal_h
#define HighOrderOperatorsQuadInternal_h

#include <element_promotion/CVFEMTypeDefs.h>
#include <KokkosInterface.h>
#include <SimdInterface.h>

#include <cmath>

namespace sierra {
namespace nalu {
namespace internal {

// Tensor contractions
template <int poly_order, typename MatViewType, typename ViewTypeIn, typename ViewTypeOut>
void apply_x(const MatViewType& coeffMatrix, const ViewTypeIn& in, ViewTypeOut& out)
{
  constexpr int n = poly_order+1;
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < n; ++i) {
      out(j,i) = 0.0;
    }

    for (int l = 0; l < n; ++l) {
      auto temp = in(j,l);
      for (int i = 0; i < n; ++i) {
        out(j,i) += temp*coeffMatrix(i,l);
      }
    }
  }
}
//--------------------------------------------------------------------------
template <int poly_order, typename MatViewType, typename ViewTypeIn, typename ViewTypeOut>
void apply_x(const MatViewType& coeffMatrix, const ViewTypeIn& in, ViewTypeOut& out, int component)
{
  constexpr int n = poly_order+1;
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < n; ++i) {
      out(component,j,i) = 0.0;
    }

    for (int l = 0; l < n; ++l) {
      auto temp = in(j,l);
      for (int i = 0; i < n; ++i) {
        out(component,j,i) += temp*coeffMatrix(i,l);
      }
    }
  }
}

//--------------------------------------------------------------------------
template <int poly_order, typename MatViewType, typename ViewTypeIn, typename ViewTypeOut>
void apply_x(const MatViewType& coeffMatrix, const ViewTypeIn& in, int in_component, ViewTypeOut& out, int out_component)
{
  constexpr int n = poly_order+1;
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < n; ++i) {
      out(out_component,j,i) = 0.0;
    }

    for (int l = 0; l < n; ++l) {
      auto temp = in(in_component, j,l);
      for (int i = 0; i < n; ++i) {
        out(out_component,j,i) += temp*coeffMatrix(i,l);
      }
    }
  }
}
//--------------------------------------------------------------------------
template <int poly_order, typename MatViewType, typename ViewTypeIn, typename ViewTypeOut>
void apply_y(const MatViewType& coeffMatrix, const ViewTypeIn& in, ViewTypeOut& out)
{
  constexpr int n = poly_order+1;
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < n; ++i) {
      out(j,i) = 0.0;
    }

    for (int l = 0; l < n; ++l) {
      auto temp = coeffMatrix(j,l);
      for (int i = 0; i < n; ++i) {
        out(j,i) += temp*in(l,i);
      }
    }
  }
}
//--------------------------------------------------------------------------
template <int poly_order, typename MatViewType, typename ViewTypeIn, typename ViewTypeOut>
void apply_y(const MatViewType& coeffMatrix, const ViewTypeIn& in, ViewTypeOut& out, int component)
{
  constexpr int n = poly_order+1;
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < n; ++i) {
      out(component, j,i) = 0.0;
    }

    for (int l = 0; l < n; ++l) {
      auto temp = coeffMatrix(j,l);
      for (int i = 0; i < n; ++i) {
        out(component, j,i) += temp*in(l,i);
      }
    }
  }
}

//--------------------------------------------------------------------------
template <int poly_order, typename MatViewType, typename ViewTypeIn, typename ViewTypeOut>
void apply_y(const MatViewType& coeffMatrix, const ViewTypeIn& in, int in_component,  ViewTypeOut& out, int out_component)
{
  constexpr int n = poly_order+1;
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < n; ++i) {
      out(out_component, j,i) = 0.0;
    }

    for (int l = 0; l < n; ++l) {
      auto temp = coeffMatrix(j,l);
      for (int i = 0; i < n; ++i) {
        out(j,i) += temp*in(in_component, l,i);
      }
    }
  }
}
//--------------------------------------------------------------------------
template <int poly_order, typename MatViewType, typename ViewTypeIn, typename ViewTypeOut>
void apply_yx(const MatViewType& coeffMatrix1, const MatViewType& coeffMatrix2, const ViewTypeIn& in,  ViewTypeOut& out)
{
  nodal_matrix_array<poly_order, typename MatViewType::value_type> scratch_data;

  constexpr int n = poly_order+1;
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < n; ++i) {
      scratch_data[j][i] = 0.0;
    }

    for (int l = 0; l < n; ++l) {
      auto temp = coeffMatrix1(j,l);
      for (int i = 0; i < n; ++i) {
        scratch_data[j][i] += temp*in(l,i);
      }
    }
  }


  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < n; ++i) {
      out(j,i) = 0.0;
    }

    for (int l = 0; l < n; ++l) {
      auto temp = scratch_data[j][l];
      for (int i = 0; i < n; ++i) {
        out(j,i) += temp*coeffMatrix2(i,l);
      }
    }
  }
}
//--------------------------------------------------------------------------
template <int poly_order, typename MatViewType, typename ViewTypeIn, typename ViewTypeOut>
void apply_yx(const MatViewType& coeffMatrix1, const MatViewType& coeffMatrix2, const ViewTypeIn& in,  ViewTypeOut& out, int component)
{
  nodal_matrix_array<poly_order, typename MatViewType::value_type> scratch_data;

  constexpr int n = poly_order+1;
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < n; ++i) {
      scratch_data[j][i] = 0.0;
    }

    for (int l = 0; l < n; ++l) {
      auto temp = coeffMatrix1(j,l);
      for (int i = 0; i < n; ++i) {
        scratch_data[j][i] += temp*in(l,i);
      }
    }
  }


  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < n; ++i) {
      out(component, j,i) = 0.0;
    }

    for (int l = 0; l < n; ++l) {
      auto temp = scratch_data[j][l];
      for (int i = 0; i < n; ++i) {
        out(component, j,i) += temp*coeffMatrix2(i,l);
      }
    }
  }
}
//--------------------------------------------------------------------------
template <int poly_order, typename MatViewType, typename ViewTypeIn, typename ViewTypeOut>
void apply_xy(const MatViewType& coeffMatrix1, const MatViewType& coeffMatrix2, const ViewTypeIn& in,  ViewTypeOut& out)
{
  nodal_matrix_array<poly_order, typename MatViewType::value_type> scratch_data;
  constexpr int n = poly_order+1;
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < n; ++i) {
      scratch_data[j][i] = 0.0;
    }

    for (int l = 0; l < n; ++l) {
      auto temp = in(j,l);
      for (int i = 0; i < n; ++i) {
        scratch_data[j][i] += temp*coeffMatrix1(i,l);
      }
    }
  }

  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < n; ++i) {
      out(j,i) = 0.0;
    }

    for (int l = 0; l < n; ++l) {
      auto temp = coeffMatrix2(j,l);
      for (int i = 0; i < n; ++i) {
        out(j,i) += temp*scratch_data[l][i];
      }
    }
  }
}
//--------------------------------------------------------------------------
template <int poly_order, typename MatViewType, typename ViewTypeIn, typename ViewTypeOut>
void apply_xy(const MatViewType& coeffMatrix1, const MatViewType& coeffMatrix2, const ViewTypeIn& in, ViewTypeOut& out, int component)
{
  nodal_matrix_array<poly_order, typename MatViewType::value_type> scratch_data;
  constexpr int n = poly_order+1;
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < n; ++i) {
      scratch_data[j][i] = 0.0;
    }

    for (int l = 0; l < n; ++l) {
      auto temp = in(j,l);
      for (int i = 0; i < n; ++i) {
        scratch_data[j][i] += temp*coeffMatrix1(i,l);
      }
    }
  }

  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < n; ++i) {
      out(component,j,i) = 0.0;
    }

    for (int l = 0; l < n; ++l) {
      auto temp = coeffMatrix2(j,l);
      for (int i = 0; i < n; ++i) {
        out(component, j,i) += temp*scratch_data[l][i];
      }
    }
  }
}
//--------------------------------------------------------------------------
template <int poly_order, typename MatViewType, typename ViewTypeIn, typename ViewTypeOut>
void apply_x_and_scatter(const MatViewType& coeffMatrix, const ViewTypeIn& in, ViewTypeOut& out)
{
  nodal_matrix_array<poly_order, typename MatViewType::value_type> scratch_data;

  constexpr int n = poly_order+1;
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < n; ++i) {
      scratch_data[j][i] = 0.0;
    }

    for (int l = 0; l < n; ++l) {
      auto temp = in(j,l);
      for (int i = 0; i < n; ++i) {
        scratch_data[j][i] += temp*coeffMatrix(i,l);
      }
    }
  }

  for (int m = 0; m < poly_order+1; ++m) {
    out(0,m) -= scratch_data[0][m];
    for (int p = 1; p < poly_order; ++p) {
      out(p,m) -= scratch_data[p][m] - scratch_data[p-1][m];
    }
    out(poly_order,m) += scratch_data[poly_order-1][m];
  }
}

//--------------------------------------------------------------------------
template <int poly_order, typename MatViewType, typename ViewTypeIn, typename ViewTypeOut>
void apply_y_and_scatter(const MatViewType& coeffMatrix, const ViewTypeIn& in, ViewTypeOut& out)
{
  nodal_matrix_array<poly_order, typename MatViewType::value_type> scratch_data;

  constexpr int n = poly_order+1;
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < n; ++i) {
      scratch_data[j][i] = 0.0;
    }

    for (int l = 0; l < n; ++l) {
      auto temp = coeffMatrix(j,l);
      for (int i = 0; i < n; ++i) {
        scratch_data[j][i] += temp*in(l,i);
      }
    }
  }

  for (int m = 0; m < poly_order+1; ++m) {
    out(m,0) -= scratch_data[m][0];
    for (int p = 1; p < poly_order; ++p) {
      out(m,p) -= scratch_data[m][p] - scratch_data[m][p - 1];
    }
    out(m, poly_order) += scratch_data[m][poly_order-1];
  }
}


}
} // namespace nalu
} // namespace Sierra

#endif

