#include <gtest/gtest.h>
#include <limits>

#include <element_promotion/operators/HighOrderOperatorsQuad.h>
#include <element_promotion/operators/HighOrderDiffusionQuad.h>
#include <element_promotion/operators/HighOrderGeometryQuadDiffusion.h>
#include <element_promotion/operators/HighOrderGeometryQuadVolume.h>
#include <element_promotion/operators/CoefficientMatrices.h>
#include <element_promotion/QuadratureRule.h>
#include <Kokkos_Core.hpp>
#include <Teuchos_BLAS.hpp>

#include <memory>
#include <tuple>
#include <random>

#include <element_promotion/ElementDescription.h>
#include "UnitTestViewUtils.h"
#include "UnitTestUtils.h"

namespace {
  //--------------------------------------------------------------
  template <typename Atype, typename Btype, typename Ctype>
  void kron(const Atype A, const Btype B, const Ctype C) {
    for (unsigned n = 0; n <  A.dimension_1(); ++n) {
      for (unsigned m = 0; m <  A.dimension_0(); ++m) {
        for (unsigned j = 0; j < A.dimension_1(); ++j) {
          for (unsigned i = 0; i < A.dimension_0(); ++i) {
            C(j*A.dimension_0()+i,n*A.dimension_0()+m) = A(n,j) * B(m,i);
          }
        }
      }
    }
  }
  //--------------------------------------------------------------
  template<typename T>
  T transpose(const T A)
  {
    T At(A.label()+"_t");

    EXPECT_EQ(A.dimension_0(), A.dimension_1());
    for (unsigned j = 0; j < A.dimension_0(); ++j) {
      for (unsigned i = 0; i < A.dimension_0(); ++i) {
        At(i,j) = A(j,i);
      }
    }
    return At;
  }
  //--------------------------------------------------------------
  template <int p> void check_volume_metric()
  {
    using AlgTraits = sierra::nalu::AlgTraitsQuad<p>;

    std::vector<double> s_exactScvVolume(AlgTraits::nodesPerElement_, 0.0);
    sierra::nalu::nodal_scalar_view<AlgTraits, double> exactScvVolume(s_exactScvVolume.data());
    std::vector<double> scsLocations1D = sierra::nalu::gauss_legendre_rule(p).first;
    std::vector<double> paddedScsLocations1D = sierra::nalu::pad_end_points(scsLocations1D,-1,+1); // add the element ends
    for (int j = 0; j < p+1; ++j) {
      double y_scsL = paddedScsLocations1D[j+0];
      double y_scsR = paddedScsLocations1D[j+1];
      for (int i = 0; i < p+1;++i) {
        double x_scsL = paddedScsLocations1D[i+0];
        double x_scsR = paddedScsLocations1D[i+1];
        exactScvVolume(j, i) = (x_scsR - x_scsL) * (y_scsR - y_scsL);
      }
    }

    std::vector<double> s_coords(AlgTraits::nDim_ * AlgTraits::nodesPerElement_,0.0);
    sierra::nalu::nodal_vector_view<AlgTraits, double> coords(s_coords.data());

    std::vector<double> coords1D = sierra::nalu::gauss_lobatto_legendre_rule(p+1).first;
    for (int j = 0; j < p+1; ++j) {
      double y = coords1D[j];
      for (int i = 0; i < p+1;++i) {
        double x = coords1D[i];
        coords(0,j,i) = x;
        coords(1,j,i) = y;
      }
    }

    auto ops = sierra::nalu::CVFEMOperators<p, double, stk::topology::QUAD_4_2D>();

    std::vector<double> s_detj(AlgTraits::nodesPerElement_, 0.0);
    sierra::nalu::nodal_scalar_view<AlgTraits, double> detj(s_detj.data());
    sierra::nalu::high_order_metrics::compute_volume_metric_linear(ops, coords, detj);

    std::vector<double> s_computedScvVolume(AlgTraits::nodesPerElement_, 0.0);
    sierra::nalu::nodal_scalar_view<AlgTraits, double> computedScvVolume(s_computedScvVolume.data());
    ops.volume_2D(detj, computedScvVolume);
    EXPECT_VIEW_NEAR_2D(exactScvVolume, computedScvVolume, 1.0e-10);
  }
  //--------------------------------------------------------------
  template <int p> Kokkos::View<double[(p+1)*(p+1)][(p+1)*(p+1)]>
  single_square_element_laplacian()
  {
    using AlgTraits = sierra::nalu::AlgTraitsQuad<p>;
    auto mat = sierra::nalu::CoefficientMatrices<p>();

    auto diff = sierra::nalu::coefficients::difference_matrix<p, double>();
    auto deriv = sierra::nalu::coefficients::scs_derivative_weights<p, double>();
    auto integ = sierra::nalu::coefficients::nodal_integration_weights<p, double>();
    auto identity = sierra::nalu::coefficients::identity_matrix<p, double>();

    std::vector<double> s_laplacian1D(AlgTraits::nodesPerElement_, 0.0);
    sierra::nalu::nodal_scalar_view<AlgTraits, double> laplacian1D(s_laplacian1D.data());
    Teuchos::BLAS<int, double>().GEMM(Teuchos::TRANS, Teuchos::TRANS,
      p + 1, p + 1, p + 1,
      1.0,
      diff.data(), p + 1, deriv.data(), p + 1,
      0.0,
      laplacian1D.data(),  p + 1);

    int ipiv[(p+1)*(p+1)] = {0};
    int info = 0;

    auto integ_t = transpose(integ);
    Teuchos::LAPACK<int,double>().GESV(p+1, p+1, integ_t.data(), p+1, ipiv, laplacian1D.data(), p+1, &info);
    EXPECT_EQ(info,0);

    std::vector<double> s_lhsx(AlgTraits::lhsSize_, 0.0);
    sierra::nalu::matrix_view<AlgTraits, double> lhsx(s_lhsx.data());
    kron(identity, laplacian1D, lhsx ); //dxx


    Kokkos::View<double[(p+1)*(p+1)][(p+1)*(p+1)]> lhs("laplacian");
    kron(laplacian1D, identity, lhs); //dyy

    for (unsigned j = 0; j < lhs.size(); ++j) {
      lhs.data()[j] += lhsx.data()[j]; // dxx + dyy
    }


    return lhs;
  }

  template <int p>
  void check_diffusion()
  {
    struct MMSFunction {
      MMSFunction() : k(1.0), pi(std::acos(-1.0)) {};

      double val(double x, double y) {return (0.25*(std::cos(1.0*k*pi*x) + std::cos(2.0*k*pi*y))); }

      double laplacian(double x, double y) const {
        return ( -(k*pi)*(k*pi) * (std::cos(1.0*k*pi*x)/4. + std::cos(2.0*k*pi*y)) );
      };

      double k;
      double pi;
    };
    MMSFunction mms;

    using AlgTraits = sierra::nalu::AlgTraitsQuad<p>;


    std::vector<double> s_coords(AlgTraits::nDim_ * AlgTraits::nodesPerElement_,0.0);
    sierra::nalu::nodal_vector_view<AlgTraits, double> coords(s_coords.data());

    std::vector<double> s_diffusivity(AlgTraits::nodesPerElement_,0.0);
    sierra::nalu::nodal_scalar_view<AlgTraits, double> diffusivity(s_diffusivity.data());

    std::vector<double> s_scalar(AlgTraits::nodesPerElement_,0.0);
    sierra::nalu::nodal_scalar_view<AlgTraits, double> scalar(s_scalar.data());

    std::vector<double> s_exact_laplacian(AlgTraits::nodesPerElement_,0.0);
    sierra::nalu::nodal_scalar_view<AlgTraits, double> exact_laplacian(s_exact_laplacian.data());

    std::vector<double> coords1D = sierra::nalu::gauss_lobatto_legendre_rule(p+1).first;
    for (int j = 0; j < p+1; ++j) {
      for (int i = 0; i < p+1;++i) {
        double x = coords1D[i];
        double y = coords1D[j];
        coords(0, j,i) = x;
        coords(1, j,i) = y;
        diffusivity(j,i) = 1.0;
        scalar(j,i) = mms.val(x,y);
        exact_laplacian(j,i) = mms.laplacian(x,y);
      }
    }
    constexpr int npe = (p+1)*(p+1);

    std::vector<double> s_mass(AlgTraits::lhsSize_,0.0);
    sierra::nalu::matrix_view<AlgTraits, double> mass(s_mass.data());

    std::vector<double> s_detj(AlgTraits::nodesPerElement_, 0.0);
    sierra::nalu::nodal_scalar_view<AlgTraits, double> detj(s_detj.data());


    auto ops = sierra::nalu::CVFEMOperators<p, double, stk::topology::QUAD_4_2D>();
    sierra::nalu::high_order_metrics::compute_volume_metric_linear(ops, coords, detj);
    kron(ops.mat_.nodalWeights, ops.mat_.nodalWeights, mass);

    Kokkos::View<double[p+1][p+1]> m_times_laplacian_of_scalar{"rhs_e"};
    Teuchos::BLAS<int,double>().GEMV(
      Teuchos::NO_TRANS,
      npe, npe,
      +1.0,
      mass.ptr_on_device(), npe,
      exact_laplacian.ptr_on_device(), 1,
      +0.0,
      m_times_laplacian_of_scalar.ptr_on_device(), 1
    );

    std::vector<double> s_AGJ(AlgTraits::metricSize_,0.0);
    sierra::nalu::scs_tensor_view<AlgTraits, double> AGJ(s_AGJ.data());
    sierra::nalu::high_order_metrics::compute_diffusion_metric_linear(ops, coords, diffusivity, AGJ);

    std::vector<double> s_lhs(AlgTraits::lhsSize_,0.0);
    sierra::nalu::matrix_view<AlgTraits, double> lhs(s_lhs.data());
    sierra::nalu::tensor_assembly::elemental_diffusion_jacobian(ops, AGJ, lhs);

    std::vector<double> s_rhs(AlgTraits::nodesPerElement_, 0.0);
    sierra::nalu::nodal_scalar_view<AlgTraits, double> rhs(s_rhs.data());
    Teuchos::BLAS<int,double>().GEMV(
      Teuchos::TRANS, // row v column
      npe, npe,
      -1.0,
      lhs.ptr_on_device(), npe,
      scalar.ptr_on_device(), 1,
      +1.0,
      rhs.ptr_on_device(), 1
    );
    EXPECT_VIEW_NEAR_2D(m_times_laplacian_of_scalar, rhs, 1.0e-8);

    std::vector<double> s_rhs_jf(AlgTraits::nodesPerElement_, 0.0);
    sierra::nalu::nodal_scalar_view<AlgTraits, double> rhs_jf(s_rhs_jf.data());
    sierra::nalu::tensor_assembly::elemental_diffusion_action(ops, AGJ, scalar, rhs_jf);
    EXPECT_VIEW_NEAR_2D(rhs, rhs_jf, 1.0e-8);

    auto laplacian_operator = single_square_element_laplacian<p>();

    std::vector<double> s_numerical_laplacian(AlgTraits::nodesPerElement_, 0.0);
    sierra::nalu::nodal_scalar_view<AlgTraits, double> numerical_laplacian(s_numerical_laplacian.data());
    Teuchos::BLAS<int,double>().GEMV(
      Teuchos::TRANS, // row v column
      npe, npe,
      -1.0,
      laplacian_operator.ptr_on_device(), npe,
      scalar.ptr_on_device(), 1,
      +0.0,
      numerical_laplacian.ptr_on_device(), 1
    );
    EXPECT_VIEW_NEAR_2D(exact_laplacian, numerical_laplacian, 1.0e-8);
  }
}
//--------------------------------------------------------------
TEST_POLY(QuadDiffusion, check_volume_metric, 42);
TEST_POLY(QuadDiffusion, check_diffusion, 42);

