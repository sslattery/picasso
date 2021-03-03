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

#ifndef PICASSO_UNIFORMCARTESIANMESHMAPPING_HPP
#define PICASSO_UNIFORMCARTESIANMESHMAPPING_HPP

#include <Picasso_CurvilinearMesh.hpp>
#include <Picasso_BatchedLinearAlgebra.hpp>
#include <Picasso_Types.hpp>

#include <Cajita.hpp>

#include <Kokkos_Core.hpp>

#include <mpi.h>

#include <type_traits>
#include <cmath>

namespace Picasso
{
//---------------------------------------------------------------------------//
/*!
  \class UniformCartesianMeshMapping
  \brief Uniform Cartesian mesh mapping function.
 */
template<class MemorySpace, std::size_t NumSpaceDim>
class UniformCartesianMeshMapping
{
    double _cell_size;
    double _inv_cell_size;
    double _cell_measure;
    double _inv_cell_measure;
    Kokkos::Array<int,NumSpaceDim> _global_num_cell;
    Kokkos::Array<bool,NumSpaceDim> _periodic;
    Kokkos::Array<double,NumSpaceDim> _local_min;
    Kokkos::Array<double,NumSpaceDim> _local_max;
};

//---------------------------------------------------------------------------//
// Template interface implementation.
template <class MemorySpace,std::size_t NumSpaceDim>
class CurvilinearMeshMapping<UniformCartesianMeshMapping<MemorySpace,NumSpaceDim>>
{
  public:
    using memory_space = MemorySpace;
    using mesh_mapping = UniformCartesianMeshMapping<NumSpaceDim>;
    static constexpr std::size_t num_space_dim = NumSpaceDim;

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
        for ( std::size_t d = 0; d < num_space_dim; ++d )
            physical_coords =
                mapping._local_min[d] + reference_coords(d) * mapping._cell_size;
    }

    // Given coordinates in the local reference frame compute the grid
    // transformation metrics. This is the of jacobian of the forward mapping,
    // its determinant, and inverse.
    template<class ReferenceCoords>
    static KOKKOS_INLINE_FUNCTION void
    transformationMetrics(
        const mesh_mapping& mapping,
        const ReferenceCoords& local_ref_coords,
        LinearAlgebra::Matrix<typename ReferenceCoords::value_type,3,3>& jacobian,
        typename ReferenceCoordinates::value_type& jacobian_det,
        LinearAlgebra::Matrix<typename ReferenceCoords::value_type,3,3>& jacobian_inv )
    {
        for ( std::size_t i = 0; i < num_space_dim; ++i )
            for ( std::size_t j = 0; j < num_space_dim; ++j )
                jacobian(i,j) = (i==j) ? mapping._cell_size : 0.0;

        jacobian_det = mapping._inv_cell_measure;

        for ( std::size_t i = 0; i < num_space_dim; ++i )
            for ( std::size_t j = 0; j < num_space_dim; ++j )
                jacobian_inv(i,j) = (i==j) ? mapping._inv_cell_size : 0.0;
    }

    // Reverse mapping. Given coordinates in the physical frame compute the
    // coordinates in the local reference frame. Return whether or not the
    // mapping succeeded.
    template<class PhysicalCoords, class ReferenceCoords>
    static KOKKOS_INLINE_FUNCTION bool
    mapToReferenceFrame( const mesh_mapping& mapping,
                         const PhysicalCoords& physical_coords,
                         ReferenceCoords& reference_coords )
    {
        for ( std::size_t d = 0; d < 3; ++d )
            reference_coords(d) =
                (physical_coords(d) - mapping._local_min[d]) *
                mapping._inv_cell_size;
    }
};

//---------------------------------------------------------------------------//
// Create a uniform mesh. Creates a mapping, a mesh, a field manager, a
// coordinate array in the field manager, and assigns the local mesh bounds to
// the mapping. A field manager containing the mesh is returned.
template<class MemorySpace, std::size_t NumSpaceDim>
auto createUniformCartesianMesh(
    MemorySpace,
    const double cell_size,
    const Kokkos::Array<double,2*NumSpaceDim>& global_bounding_box,
    const Kokkos::Array<bool,NumSpaceDim>& periodic,
    const int base_halo,
    const int extended_halo,
    MPI_Comm comm,
    const std::array<int,NumSpaceDim>& ranks_per_dim )
{
    // Create the mapping.
    UniformCartesianMeshMapping<MemorySpace,NumSpaceDim> mapping;
    mapping._cell_size = cell_size;
    mapping._inv_cell_size = 1.0 / cell_size;
    mapping._cell_measure = pow( cell_size, NumSpaceDim );
    mapping._inv_cell_measure = 1.0 / mapping._cell_measure;
    mapping._periodic = periodic;
    for ( std::size_t d = 0; d < num_space_dim; ++d )
    {
        mapping._global_num_cell[d] =
            ( global_bounding_box[2*d+1] -
              global_bounding_box[2*d] ) / cell_size;
    }

    // Create mesh.
    auto mesh = createCurvilinearMesh( mapping, base_halo, extended_halo,
                                       comm, ranks_per_dim );

    // Create field manager.
    auto manager = createFieldManager( mesh );

    // Get the local bounds.
    auto local_mesh =
        Cajita::createLocalMesh<Kokkos::HostSpace>( *(mesh->localGrid()) );
    for ( std::size_t d = 0; d < num_space_dim; ++d )
    {
        mapping._local_min[d] = local_mesh.lowCorner( Cajita::Ghost(), d );
        mapping._local_max[d] = local_mesh.highCorner( Cajita::Ghost(), d );
    }

    return manager;
}

//---------------------------------------------------------------------------//

} // end namespace Picasso

#endif // end PICASSO_UNIFORMCARTESIANMESHMAPPING_HPP
