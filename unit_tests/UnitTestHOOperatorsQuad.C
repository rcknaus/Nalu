#include <gtest/gtest.h>
#include <limits>
#include <random>
#include <stdexcept>

#include <stk_util/parallel/Parallel.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Bucket.hpp>
#include <stk_mesh/base/GetEntities.hpp>
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/FieldBase.hpp>

#include <element_promotion/ElementDescription.h>
#include <element_promotion/CVFEMTypeDefs.h>
#include <element_promotion/operators/HighOrderOperatorsQuad.h>

#include "UnitTestViewUtils.h"
#include "UnitTestUtils.h"

namespace {

double my_tol = 1.0e-10;

double poly_val(std::vector<double> coeffs, double x)
{
  double val = 0.0;
  for (unsigned j = 0; j < coeffs.size(); ++j) {
    val += coeffs[j]*std::pow(x,j);
  }
  return val;
}
//--------------------------------------------------------------------------
double poly_der(std::vector<double> coeffs, double x)
{
  double val = 0.0;
  for (unsigned j = 1; j < coeffs.size(); ++j) {
    val += coeffs[j]*std::pow(x,j-1)*j;
  }
  return val;
}
//--------------------------------------------------------------------------
double poly_int(std::vector<double> coeffs,
  double xlower, double xupper)
{
  double upper = 0.0; double lower = 0.0;
  for (unsigned j = 0; j < coeffs.size(); ++j) {
    upper += coeffs[j]*std::pow(xupper,j+1)/(j+1.0);
    lower += coeffs[j]*std::pow(xlower,j+1)/(j+1.0);
  }
  return (upper-lower);
}
//--------------------------------------------------------------------------
template <int p> void scs_interp_quad()
{
  auto elem = sierra::nalu::ElementDescription::create(2, p);
  std::mt19937 rng;
  rng.seed(0);
  std::uniform_real_distribution<double> coeff(-1.0, 1.0);
  std::vector<double> coeffsX = { coeff(rng), coeff(rng) };
  std::vector<double> coeffsY = { coeff(rng), coeff(rng) };

  using AlgTraits = sierra::nalu::AlgTraitsQuad<p>;
  auto ops = sierra::nalu::CVFEMOperators<p, double, AlgTraits::baseTopo_>();

  double s_nodalValues[AlgTraits::nodesPerElement_]  = {0.0};;
  sierra::nalu::nodal_scalar_view<AlgTraits,double> nodalValues(s_nodalValues);

  for (int j = 0; j < p +1; ++j) {
    double locy = elem->nodeLocs1D[j];
    for (int i = 0; i < p + 1; ++i) {
      double locx = elem->nodeLocs1D[i];
      nodalValues(j,i) =  poly_val(coeffsX, locx)  * poly_val(coeffsY, locy);
    }
  }

  const auto scsLocs = sierra::nalu::gauss_legendre_rule(p).first;

  double s_exact_xhat_interp[AlgTraits::nodesPerElement_] = {0.0};;
  sierra::nalu::nodal_scalar_view<AlgTraits,double> exact_xhat_interp(s_exact_xhat_interp);

  for (int j = 0; j < p + 1; ++j) {
    double locy = elem->nodeLocs1D[j];
    for (int i = 0; i < p; ++i) {
      double locx = scsLocs[i];
      exact_xhat_interp(j,i) =  poly_val(coeffsX, locx) * poly_val(coeffsY, locy);
    }
  }

  double s_operator_xhat_interp[AlgTraits::nodesPerElement_] = {0.0};
  sierra::nalu::nodal_scalar_view<AlgTraits,double> operator_xhat_interp(s_operator_xhat_interp);

  ops.scs_xhat_interp(nodalValues, operator_xhat_interp);
  EXPECT_VIEW_NEAR_2D(exact_xhat_interp, operator_xhat_interp, my_tol);

  double s_exact_yhat_interp[AlgTraits::nodesPerElement_] = {0.0};
  sierra::nalu::nodal_scalar_view<AlgTraits,double> exact_yhat_interp(s_exact_yhat_interp);
  for (int j = 0; j < p; ++j) {
    double locy = scsLocs[j];
    for (int i = 0; i < p+1; ++i) {
      double locx = elem->nodeLocs1D[i];
      exact_yhat_interp(j,i) =  poly_val(coeffsX, locx) * poly_val(coeffsY, locy);
    }
  }

  double s_operator_yhat_interp[AlgTraits::nodesPerElement_] = {0.0};
  sierra::nalu::nodal_scalar_view<AlgTraits,double> operator_yhat_interp(s_operator_yhat_interp);

  ops.scs_yhat_interp(nodalValues, operator_yhat_interp);
  EXPECT_VIEW_NEAR_2D(exact_yhat_interp, operator_yhat_interp, my_tol);
}
////--------------------------------------------------------------------------
template <int p> void scs_grad_quad()
{
  auto elem = sierra::nalu::ElementDescription::create(2, p);
  std::mt19937 rng;
  rng.seed(0);
  std::uniform_real_distribution<double> coeff(-1.0, 1.0);
  std::vector<double> coeffsX = { coeff(rng), coeff(rng) };
  std::vector<double> coeffsY = { coeff(rng), coeff(rng) };

  using AlgTraits = sierra::nalu::AlgTraitsQuad<p>;
  auto ops = sierra::nalu::CVFEMOperators<p, double, AlgTraits::baseTopo_>();

  double s_nodalValues[AlgTraits::nodesPerElement_] = {0.0};;
  sierra::nalu::nodal_scalar_view<AlgTraits,double> nodalValues(s_nodalValues);
  for (int j = 0; j < p +1; ++j) {
    double locy = elem->nodeLocs1D[j];
    for (int i = 0; i < p + 1; ++i) {
      double locx = elem->nodeLocs1D[i];
      nodalValues(j,i) = poly_val(coeffsX, locx)  * poly_val(coeffsY, locy);
    }
  }

 const auto scsLocs = sierra::nalu::gauss_legendre_rule(p).first;

  double s_exact_xhat_grad[AlgTraits::nDim_ * AlgTraits::nodesPerElement_] = {0.0};
  sierra::nalu::nodal_vector_view<AlgTraits,double> exact_xhat_grad(s_exact_xhat_grad);
  for (int j = 0; j < p + 1; ++j) {
    double locy = elem->nodeLocs1D[j];
    for (int i = 0; i < p; ++i) {
      double locx = scsLocs[i];
      exact_xhat_grad(0, j, i) =  poly_der(coeffsX, locx) * poly_val(coeffsY, locy);
      exact_xhat_grad(1, j, i) =  poly_val(coeffsX, locx) * poly_der(coeffsY, locy);
    }
  }

  double s_op_xhat_grad[AlgTraits::nDim_ * AlgTraits::nodesPerElement_] = {0.0};
  sierra::nalu::nodal_vector_view<AlgTraits,double> op_xhat_grad(s_op_xhat_grad);
  ops.scs_xhat_grad(nodalValues, op_xhat_grad);
  EXPECT_VIEW_NEAR_3D(exact_xhat_grad, op_xhat_grad, my_tol);

  double s_exact_yhat_grad[AlgTraits::nDim_ * AlgTraits::nodesPerElement_] = {0.0};
  sierra::nalu::nodal_vector_view<AlgTraits,double> exact_yhat_grad(s_exact_yhat_grad);
  for (int j = 0; j < p; ++j) {
    double locy = scsLocs[j];
    for (int i = 0; i < p+1; ++i) {
      double locx = elem->nodeLocs1D[i];
      exact_yhat_grad(0, j, i) =  poly_der(coeffsX, locx) * poly_val(coeffsY, locy);
      exact_yhat_grad(1, j, i) =  poly_val(coeffsX, locx) * poly_der(coeffsY, locy);
    }
  }

  double s_op_yhat_grad[AlgTraits::nDim_ * AlgTraits::nodesPerElement_] = {0.0};
  sierra::nalu::nodal_vector_view<AlgTraits, double> op_yhat_grad(s_op_yhat_grad);
  ops.scs_yhat_grad(nodalValues, op_yhat_grad);

  EXPECT_VIEW_NEAR_3D(exact_yhat_grad, op_yhat_grad, my_tol);
}
////--------------------------------------------------------------------------
template <int p> void nodal_grad_quad()
{
  auto elem = sierra::nalu::ElementDescription::create(2, p);
  std::mt19937 rng;
  rng.seed(0);
  std::uniform_real_distribution<double> coeff(-1.0, 1.0);
  std::vector<double> coeffsX = { coeff(rng), coeff(rng) };
  std::vector<double> coeffsY = { coeff(rng), coeff(rng) };

  using AlgTraits = sierra::nalu::AlgTraitsQuad<p>;
  auto ops = sierra::nalu::CVFEMOperators<p, double, AlgTraits::baseTopo_>();

  double s_nodalValues[AlgTraits::nodesPerElement_] = {0.0};
  sierra::nalu::nodal_scalar_view<AlgTraits,double> nodalValues(s_nodalValues);
  for (int j = 0; j < p +1; ++j) {
    double locy = elem->nodeLocs1D[j];
    for (int i = 0; i < p + 1; ++i) {
      double locx = elem->nodeLocs1D[i];
      nodalValues(j,i) =  poly_val(coeffsX, locx)  * poly_val(coeffsY, locy);
    }
  }

  const auto scsLocs = sierra::nalu::gauss_legendre_rule(p).first;

  double s_exact_nodal_grad[AlgTraits::nDim_ * AlgTraits::nodesPerElement_]  = {0.0};
  sierra::nalu::nodal_vector_view<AlgTraits,double> exact_nodal_grad(s_exact_nodal_grad);
  for (int j = 0; j < p + 1; ++j) {
    double locy = elem->nodeLocs1D[j];
    for (int i = 0; i < p+1; ++i) {
      double locx = elem->nodeLocs1D[i];
      exact_nodal_grad(0, j, i) =  poly_der(coeffsX, locx) * poly_val(coeffsY, locy);
      exact_nodal_grad(1, j, i) =  poly_val(coeffsX, locx) * poly_der(coeffsY, locy);
    }
  }

  double s_op_nodal_grad[AlgTraits::nDim_ * AlgTraits::nodesPerElement_] = {0.0};
  sierra::nalu::nodal_vector_view<AlgTraits,double> op_nodal_grad(s_op_nodal_grad);
  ops.nodal_grad(nodalValues, op_nodal_grad);
  EXPECT_VIEW_NEAR_3D(exact_nodal_grad, op_nodal_grad, my_tol);
}
////--------------------------------------------------------------------------
template <int p> void scs_integration_quad()
{
  auto elem = sierra::nalu::ElementDescription::create(2, p);
  std::mt19937 rng;
  rng.seed(0);
  std::uniform_real_distribution<double> coeff(-1.0, 1.0);
  std::vector<double> coeffsX = { coeff(rng), coeff(rng) };
  std::vector<double> coeffsY = { coeff(rng), coeff(rng) };

  using AlgTraits = sierra::nalu::AlgTraitsQuad<p>;
  auto ops = sierra::nalu::CVFEMOperators<p, double, AlgTraits::baseTopo_>();

  double s_nodalValues[AlgTraits::nodesPerElement_] = {0.0};
  sierra::nalu::nodal_scalar_view<AlgTraits,double> nodalValues(s_nodalValues);
  for (int j = 0; j < p +1; ++j) {
    double locy = elem->nodeLocs1D[j];
    for (int i = 0; i < p + 1; ++i) {
      double locx = elem->nodeLocs1D[i];
      nodalValues(j,i) =  poly_val(coeffsX, locx)  * poly_val(coeffsY, locy);
    }
  }

  const auto scsEndLoc = sierra::nalu::pad_end_points(sierra::nalu::gauss_legendre_rule(p).first);

  double s_exact_xhat_integral[AlgTraits::nodesPerElement_] = {0.0};
  sierra::nalu::nodal_scalar_view<AlgTraits,double> exact_xhat_integral(s_exact_xhat_integral);

  double s_exact_yhat_integral[AlgTraits::nodesPerElement_] = {0.0};
  sierra::nalu::nodal_scalar_view<AlgTraits,double> exact_yhat_integral(s_exact_yhat_integral);

  double s_exact_vol_integral[AlgTraits::nodesPerElement_] = {0.0};
  sierra::nalu::nodal_scalar_view<AlgTraits,double> exact_vol_integral(s_exact_vol_integral);

  for (unsigned j = 0; j < p + 1; ++j) {
    double yl = scsEndLoc[j + 0];
    double yr = scsEndLoc[j + 1];
    double locy = elem->nodeLocs1D[j];
    for (unsigned i = 0; i < p + 1; ++i) {
      double xl = scsEndLoc[i + 0];
      double xr = scsEndLoc[i + 1];
      double locx = elem->nodeLocs1D[i];
      exact_xhat_integral(j, i) = poly_int(coeffsX, xl, xr) * poly_val(coeffsY, locy);
      exact_yhat_integral(j, i) = poly_val(coeffsX, locx)   * poly_int(coeffsY, yl, yr);
      exact_vol_integral(j, i) =  poly_int(coeffsX, xl, xr) * poly_int(coeffsY, yl, yr);
    }
  }

  double s_op_xhat_integral[AlgTraits::nodesPerElement_] = {0.0};
  sierra::nalu::nodal_scalar_view<AlgTraits,double> op_xhat_integral(s_op_xhat_integral);

  double s_op_yhat_integral[AlgTraits::nodesPerElement_] = {0.0};
  sierra::nalu::nodal_scalar_view<AlgTraits,double> op_yhat_integral(s_op_yhat_integral);

  double s_op_vol_integral[AlgTraits::nodesPerElement_] = {0.0};
  sierra::nalu::nodal_scalar_view<AlgTraits,double> op_vol_integral(s_op_vol_integral);

  ops.volume_xhat(nodalValues, op_xhat_integral);
  EXPECT_VIEW_NEAR_2D(exact_xhat_integral, op_xhat_integral, my_tol);

  ops.volume_yhat(nodalValues, op_yhat_integral);
  EXPECT_VIEW_NEAR_2D(exact_yhat_integral, op_yhat_integral, my_tol);

  ops.volume_2D(nodalValues, op_vol_integral);
  EXPECT_VIEW_NEAR_2D(exact_vol_integral, op_vol_integral, my_tol);
}
//
}//namespace

TEST_POLY(HOOperators, scs_interp_quad, 5);
TEST_POLY(HOOperators, scs_grad_quad, 5);
TEST_POLY(HOOperators, nodal_grad_quad, 5);
TEST_POLY(HOOperators, scs_integration_quad, 5);
