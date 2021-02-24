/****************************************************************************
 * Copyright (c) 2021 by the Picasso authors                                *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Picasso library. Picasso is distributed under a *
 * BSD 3-clause license. For the licensing terms see the LICENSE file in    *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#ifndef PICASSO_BILINEARMESHMAPPING_HPP
#define PICASSO_BILINEARMESHMAPPING_HPP

#include <Picasso_CurvilinearMesh.hpp>
#include <Picasso_BatchedLinearAlgebra.hpp>
#include <Picasso_Types.hpp>

#include <Cajita.hpp>

#include <Kokkos_Core.hpp>

#include <type_traits>

namespace Picasso
{
//---------------------------------------------------------------------------//
/*!
  \class BilinearMeshMapping
  \brief Bilinear mesh mapping function.

  All cells in the mesh are described by physical node locations and a linear
  lagrange basis representing the space continuum between them.
 */
template<class MemorySpace, std::size_t NumSpaceDim>
class BilinearMeshMapping
{
  public:

    using memory_space = MemorySpace;

    static constexpr std::size_t num_space_dim = NumSpaceDim;

    using mesh_type =
        CurvilinearMesh<BilinearMeshMapping<MemorySpace,NumSpaceDim>>;

    using coord_view_type = std::conditional_t<
        3 == num_space_dim, Kokkos::View<value_type****, MemorySpace>,
        std::conditional_t<2 == num_space_dim,
                           Kokkos::View<value_type***, MemorySpace>, void>>;

    // Construct a bilinear mapping.
    BilinearMeshMapping( const Kokkos::array<int,NumSpaceDim>& global_num_cell,
                         const Kokkos::array<bool,NumSpaceDim>& periodic )
        : _global_num_cell( global_num_cell )
        , _periodic( periodic )
    {}

    // Set the local node coordinates.
    void setLocalNodeCoordinates( const coord_view_type& local_node_coords )
    {
        _local_node_coords = local_node_coords;
    }

  public:
    Kokkos::array<int,NumSpaceDim> _global_num_cell;
    Kokkos::array<bool,NumSpaceDim> _periodic;
    coord_view_type _local_node_coords;
};

//---------------------------------------------------------------------------//
// Template interface implementation. 3D specialization
template <class MemorySpace>
class CurvilinearMeshMapping<BilinearMeshMapping<MemorySpace,3>>
{
  public:
    using mesh_mapping = BilinearMeshMapping<MemorySpace,3>;
    static constexpr std::size_t num_space_dim = 3;

    // Get the global number of cells in given logical dimension that construct
    // the mapping.
    static int globalNumCell( const mesh_mapping& mapping, const int dim )
    {
        return mapping._global_num_cell[dim];
    }

    // Get the periodicity of a given logical dimension of the mapping.
    static bool periodic( const mesh_mapping& mapping, const int dim )
    {
        return mapping._periodic[dim];
    }

    // Forward mapping. Given coordinates in the reference frame compute the
    // coordinates in the physical frame.
    template<class ReferenceCoords, class PhysicalCoords>
    static KOKKOS_INLINE_FUNCTION void
    mapToPhysicalFrame( const mesh_mapping& mapping,
                        const ReferenceCoords& reference_coords,
                        PhysicalCoords& physical_coords )
    {
        using value_type = typename ReferenceCoords::value_type;

        int i = static_cast<int>( reference_coords(Dim::I) );
        int j = static_cast<int>( reference_coords(Dim::J) );
        int k = static_cast<int>( reference_coords(Dim::K) );

        value_type xi = reference_coords(Dim::I) - i;
        value_type xj = reference_coords(Dim::J) - j;
        value_type xk = reference_coords(Dim::K) - k;

        value_type w[3][2] = { {1.0 - xi, xi}, {1.0 - xj, xj}, {1.0 - xk, xk} };

        physical_coords = 0.0;

        for ( int ni = 0; ni < 2; ++ni )
            for ( int nj = 0; nj < 2; ++nj )
                for ( int nk = 0; nk < 2; ++nk )
                    for ( int d = 0; d < 3; ++d )
                        physical_coords(d) +=
                            w[Dim::I][ni] * w[Dim::J][nj] * w[Dim::K][nk] *
                            mapping._local_node_coords(i+ni,j+nj,k+nk,d);
    }

    // Given coordinates in the local reference frame compute the grid
    // transformation metrics. This is the of jacobian of the forward mapping,
    // its determinant, and inverse.
    template<class ReferenceCoords>
    static KOKKOS_INLINE_FUNCTION void
    transformationMetrics(
        const Mapping& mapping,
        const ReferenceCoords& local_ref_coords,
        LinearAlgebra::Matrix<typename ReferenceCoords::value_type,3,3>& jacobian,
        typename ReferenceCoordinates::value_type& jacobian_det,
        LinearAlgebra::Matrix<typename ReferenceCoords::value_type,3,3>& jacobian_inv )
    {
        using value_type = typename ReferenceCoords::value_type;

        int i = static_cast<int>( reference_coords(Dim::I) );
        int j = static_cast<int>( reference_coords(Dim::J) );
        int k = static_cast<int>( reference_coords(Dim::K) );

        value_type xi = reference_coords(Dim::I) - i;
        value_type xj = reference_coords(Dim::J) - j;
        value_type xk = reference_coords(Dim::K) - k;

        value_type w[3][2] = { {1.0 - xi, xi}, {1.0 - xj, xj}, {1.0 - xk, xk} };
        value_type g[3][2] = { {-1.0,1.0}, {-1.0,1.0}, {-1.0,1.0} };

        jacobian = 0.0;

        for ( int ni = 0; ni < 2; ++ni )
            for ( int nj = 0; nj < 2; ++nj )
                for ( int nk = 0; nk < 2; ++nk )
                    for ( int d = 0; d < 3; ++d )
                    {
                        jacobian(d,Dim::I) +=
                            g[Dim::I][ni] * w[Dim::J][nj] * w[Dim::K][nk] *
                            mapping._local_node(i+ni,j+nj,k+nk,d);

                        jacobian(d,Dim::J) +=
                            w[Dim::I][ni] * g[Dim::J][nj] * w[Dim::K][nk] *
                            mapping._local_node(i+ni,j+nj,k+nk,d);

                        jacobian(d,Dim::K) +=
                            w[Dim::I][ni] * w[Dim::J][nj] * g[Dim::K][nk] *
                            mapping._local_node_coords(i+ni,j+nj,k+nk,d);
                    }

        jacobian_det = !jacobian;

        jacobian_inv = LinearAlgebra::inverse( jacobian, jacobian_det );
    }

    // Reverse mapping. Given coordinates in the physical frame compute the
    // coordinates in the local reference frame of the given cell. The
    // contents of local_ref_coords will be used as the initial guess. Return
    // whether or not the mapping succeeded.
    template<class PhysicalCoords, class ReferenceCoords>
    static KOKKOS_INLINE_FUNCTION bool
    mapToReferenceFrame( const mesh_mapping& mapping,
                         const PhysicalCoords& physical_coords,
                         const int ijk[3],
                         ReferenceCoords& reference_coords )
    {
        return DefaultCurvilinearMeshMapping<
            BilinearMeshMapping<MemorySpace,3>>::mapToReferenceFrame(
                mapping, physical_coords, ijk, reference_coords );
    }
};

//---------------------------------------------------------------------------//
// Template interface implementation. 2D specialization
template <class MemorySpace>
struct CurvilinearMeshMapping<BilinearMeshMapping<MemorySpace,2>>
{
    using mesh_mapping = BilinearMeshMapping<MemorySpace,2>;
    static constexpr std::size_t num_space_dim = 2;

    // Get the global number of cells in given logical dimension that construct
    // the mapping.
    static int globalNumCell( const mesh_mapping& mapping, const int dim )
    {
        return mapping._global_num_cell[dim];
    }

    // Get the periodicity of a given logical dimension of the mapping.
    static bool periodic( const mesh_mapping& mapping, const int dim )
    {
        return mapping._periodic[dim];
    }

    // Forward mapping. Given coordinates in the reference frame compute the
    // coordinates in the physical frame.
    template<class ReferenceCoords, class PhysicalCoords>
    static KOKKOS_INLINE_FUNCTION void
    mapToPhysicalFrame( const mesh_mapping& mapping,
                        const ReferenceCoords& reference_coords,
                        PhysicalCoords& physical_coords )
    {
        using value_type = typename ReferenceCoords::value_type;

        int i = static_cast<int>( reference_coords(Dim::I) );
        int j = static_cast<int>( reference_coords(Dim::J) );

        value_type xi = reference_coords(Dim::I) - i;
        value_type xj = reference_coords(Dim::J) - j;

        value_type w[2][2] = { {1.0 - xi, xi}, {1.0 - xj, xj} };

        physical_coords = 0.0;

        for ( int ni = 0; ni < 2; ++ni )
            for ( int nj = 0; nj < 2; ++nj )
                for ( int d = 0; d < 2; ++d )
                    physical_coords(d) +=
                        w[Dim::I][ni] * w[Dim::J][nj] *
                        mapping._local_node_coords(i+ni,j+nj,d);
    }

    // Given coordinates in the local reference frame compute the grid
    // transformation metrics. This is the of jacobian of the forward mapping,
    // its determinant, and inverse.
    template<class ReferenceCoords>
    static KOKKOS_INLINE_FUNCTION void
    transformationMetrics(
        const Mapping& mapping,
        const ReferenceCoords& local_ref_coords,
        LinearAlgebra::Matrix<typename ReferenceCoords::value_type,2,2>& jacobian,
        typename ReferenceCoordinates::value_type& jacobian_det,
        LinearAlgebra::Matrix<typename ReferenceCoords::value_type,2,2>& jacobian_inv )
    {
        using value_type = typename ReferenceCoords::value_type;

        int i = static_cast<int>( reference_coords(Dim::I) );
        int j = static_cast<int>( reference_coords(Dim::J) );

        value_type xi = reference_coords(Dim::I) - i;
        value_type xj = reference_coords(Dim::J) - j;

        value_type w[2][2] = { {1.0 - xi, xi}, {1.0 - xj, xj} };
        value_type g[2][2] = { {-1.0,1.0}, {-1.0,1.0} };

        jacobian = 0.0;

        for ( int ni = 0; ni < 2; ++ni )
            for ( int nj = 0; nj < 2; ++nj )
                for ( int d = 0; d < 2; ++d )
                {
                    jacobian(d,Dim::I) +=
                        g[Dim::I][ni] * w[Dim::J][nj] *
                        mapping._local_node_coords(i+ni,j+nj,d);

                    jacobian(d,Dim::J) +=
                        w[Dim::I][ni] * g[Dim::J][nj] *
                        mapping._local_node_coords(i+ni,j+nj,d);
                }

        jacobian_det = !jacobian;

        jacobian_inv = LinearAlgebra::inverse( jacobian, jacobian_det );
    }

    // Reverse mapping. Given coordinates in the physical frame compute the
    // coordinates in the local reference frame of the given cell. The
    // contents of local_ref_coords will be used as the initial guess. Return
    // whether or not the mapping succeeded.
    template<class PhysicalCoords, class ReferenceCoords>
    static KOKKOS_INLINE_FUNCTION bool
    mapToReferenceFrame( const mesh_mapping& mapping,
                         const PhysicalCoords& physical_coords,
                         const int ijk[2],
                         ReferenceCoords& reference_coords )
    {
        // NOTE: I am using the default procedure here but it is possible that
        // the procedure outlined on page 329 of the 1986 Brackbill FLIP paper
        // could be more efficient here as it is a direct solve. This default
        // version could be used if that one fails.
        return DefaultCurvilinearMeshMapping<
            BilinearMeshMapping<MemorySpace,2>>::mapToReferenceFrame(
                mapping, physical_coords, ijk, reference_coords );
    }
};

//---------------------------------------------------------------------------//

} // end namespace Picasso

#endif // end PICASSO_BILINEARMESHMAPPING_HPP
