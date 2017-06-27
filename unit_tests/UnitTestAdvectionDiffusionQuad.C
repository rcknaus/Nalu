#include <gtest/gtest.h>
#include <limits>

#include <element_promotion/operators/HighOrderOperatorsQuad.h>
#include <element_promotion/operators/HighOrderDiffusionQuad.h>
#include <element_promotion/operators/HighOrderGeometryQuadAdvection.h>
#include <element_promotion/operators/HighOrderGeometryQuadVolume.h>
#include <element_promotion/operators/CoefficientMatrices.h>
#include <element_promotion/QuadratureRule.h>
#include <Kokkos_Core.hpp>
#include <Teuchos_BLAS.hpp>

#include <element_promotion/computational_kernels/ScalarAdvDiffHOElemKernel.h>

#include <memory>
#include <tuple>
#include <random>

#include <element_promotion/ElementDescription.h>
#include "UnitTestViewUtils.h"
#include "UnitTestUtils.h"

// tests advection diffu

namespace {
  //--------------------------------------------------------------
  template <typename T, typename Y>
  void kron(const T A, const T B, const Y C) {
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
  template <int p> void check_advection_metric()
  {
    using AlgTraits = sierra::nalu::AlgTraitsQuad<p>;

    sierra::nalu::nodal_scalar_view<AlgTraits> exactScvVolume{"vol"};
    std::vector<double> scsLocations1D = sierra::nalu::gauss_legendre_rule(p).first;
    std::vector<double> paddedScsLocations1D = sierra::nalu::pad_end_points(scsLocations1D,-1,+1); // add the element ends

    sierra::nalu::nodal_vector_view<AlgTraits> coords{"coords"};
    sierra::nalu::nodal_vector_view<AlgTraits> rhou{"rhou"};
    std::vector<double> coords1D = sierra::nalu::gauss_lobatto_legendre_rule(p+1).first;

    std::mt19937 rng;
    rng.seed(0);
    std::uniform_real_distribution<double> angle_distribution(0, 2*std::acos(-1));

    double angle = angle_distribution(rng);
    double Q[2][2] = {
        { std::cos(angle), -std::sin(angle) },
        { std::sin(angle),  std::cos(angle) }
    };

    for (int j = 0; j < p+1; ++j) {
      double y = coords1D[j];
      for (int i = 0; i < p+1;++i) {
        double x = coords1D[i];
        coords(0,j,i) = Q[0][0]*x + Q[0][1]*y;
        coords(1,j,i) = Q[1][0]*x + Q[1][1]*y;

        double uvel = 0.5;
        double vvel = 0.5;
        rhou(0,j,i) = Q[0][0]*uvel + Q[0][1]*vvel;
        rhou(1,j,i) = Q[1][0]*uvel + Q[1][1]*vvel;
      }
    }

    auto ops = sierra::nalu::CVFEMOperatorsQuad<p>();
    sierra::nalu::scs_vector_view<AlgTraits> adv_metric{"rhouA"};
    sierra::nalu::high_order_metrics::compute_advection_metric_linear(ops, coords, rhou, adv_metric);


    sierra::nalu::scs_vector_view<AlgTraits> adv_metric_exact{"rhoA_exact"};
    Kokkos::deep_copy(adv_metric_exact, 1.0);

    EXPECT_VIEW_NEAR_3D(adv_metric, adv_metric_exact, 1.0e-12);
  }
}
//--------------------------------------------------------------
TEST_POLY(QuadAdvDiff, check_advection_metric, 10);

