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

#ifndef PICASSO_CURVILINEARMESH_HPP
#define PICASSO_CURVILINEARMESH_HPP

#include <Picasso_Types.hpp>
#include <Picasso_BatchedLinearAlgebra.hpp>

#include <Cajita.hpp>

#include <Kokkos_Core.hpp>

#include <cmath>
#include <array>
#include <type_traits>
#include <memory>

#include <mpi.h>

namespace Picasso
{
//---------------------------------------------------------------------------//
// Curvilinear mesh mapping function template interface. Describes the
// physical frame and physical-to-reference mapping for a curvilinear mesh.
//
// Physical frame - the physical representation in the chosen mapping
// coordinate system.
//
// Reference frame - the logical representation in the local grid reference
// frame. Combined with global mesh information, the mesh mapping
// implementation can convert this to a logical representation in the global
// grid reference frame.
//
// The global reference frame spans from [0,globalNumCell(dim)] in each
// dimension. The mesh data structure created from this mapping will apply
// domain decomposition to the global reference frame and create a local
// reference frame. Note that the global reference frame will be padded in
// non-periodic dimensions by the base halo width when the mesh is constructed
// to allow for stencil operations outside of the domain. Therefore in
// practice the global reference frame spans from
// [-base_halo,globalNumCell(dim)+base_halo] in each dimension.
//
template <class Mapping>
struct CurvilinearMeshMapping
{
    // Memory space.
    using memory_space = typename Mapping::memory_space;

    // Spatial dimension.
    static constexpr std::size_t num_space_dim = Mapping::num_space_dim;

    // Get the global number of cells in given logical dimension that construct
    // the mapping.
    static int globalNumCell( const Mapping& mapping, const int dim );

    // Get the periodicity of a given logical dimension of the mapping.
    static bool periodic( const Mapping& mapping, const int dim );

    // Forward mapping. Given coordinates in the local reference frame compute
    // the coordinates in the physical frame.
    template<class ReferenceCoords, class PhysicalCoords>
    static KOKKOS_INLINE_FUNCTION void
    mapToPhysicalFrame( const Mapping& mapping,
                        const ReferenceCoords& local_ref_coords,
                        PhysicalCoords& physical_coords );

    // Given coordinates in the local reference frame compute the grid
    // transformation metrics. This is the jacobian of the forward mapping,
    // its determinant, and inverse.
    template<class ReferenceCoords>
    static KOKKOS_INLINE_FUNCTION void
    transformationMetrics(
        const Mapping& mapping,
        const ReferenceCoords& local_ref_coords,
        LinearAlgebra::Matrix<typename ReferenceCoords::value_type,
        num_space_dim,num_space_dim>& jacobian,
        typename ReferenceCoords::value_type& jacobian_det,
        LinearAlgebra::Matrix<typename ReferenceCoords::value_type,
        num_space_dim,num_space_dim>& jacobian_inv );

    // Reverse mapping. Given coordinates in the physical frame compute the
    // coordinates in the local reference frame. The data in local_ref_coords
    // will be used as the initial guess. Return whether or not the mapping
    // succeeded.
    template<class PhysicalCoords, class ReferenceCoords>
    static KOKKOS_INLINE_FUNCTION bool
    mapToReferenceFrame( const Mapping& mapping,
                         const PhysicalCoords& physical_coords,
                         ReferenceCoords& local_ref_coords );
};

//---------------------------------------------------------------------------//
// Default implementations for mesh mapping functions.
template<class Mapping>
struct DefaultCurvilinearMeshMapping
{
    static constexpr std::size_t num_space_dim = Mapping::num_space_dim;

    // NOTE: Automatic differentiation of mapToPhysicalFrame could be used as
    // a default implementation of the transformation metrics.

    // Reverse mapping. Given coordinates in the physical frame compute the
    // coordinates in the local reference frame. The data in local_ref_coords
    // will be used as the initial guess. Return whether or not the mapping
    // succeeded.
    template<class PhysicalCoords, class ReferenceCoords>
    static KOKKOS_INLINE_FUNCTION bool
    mapToReferenceFrame( const Mapping& mapping,
                         const PhysicalCoords& physical_coords,
                         ReferenceCoords& local_ref_coords )
    {
        using value_type = typename PhysicalCoords::value_type;

        // Newton iteration tolerance.
        double tol = 1.0e-12;
        double tol2 = tol * tol;

        // Maximum number of Newton iterations.
        int max_iter = 15;

        // Iteration data.
        LinearAlgebra::Vector<value_type,num_space_dim> x_ref_old;
        LinearAlgebra::Vector<value_type,num_space_dim> x_phys_new;
        LinearAlgebra::Matrix<value_type,num_space_dim,num_space_dim> jacobian;
        value_type jacobian_det;
        LinearAlgebra::Matrix<value_type,num_space_dim,num_space_dim> jacobian_inv;

        // Newton iterations.
        value_type error;
        for ( int n = 0; n < max_iter; ++n )
        {
            // Update iteration.
            x_ref_old = local_ref_coords;

            // Compute jacobian.
            CurvilinearMeshMapping<Mapping>::transformationMetrics(
                mapping, x_ref_old, jacobian, jacobian_det, jacobian_inv );

            // Compute residual.
            CurvilinearMeshMapping<Mapping>::mapToPhysicalFrame(
                mapping, x_ref_old, x_phys_new );

            // Update solution.
            local_ref_coords =
                jacobian_inv * ( physical_coords - x_phys_new ) + x_ref_old;

            // Check for convergence.
            error = ~(local_ref_coords - x_ref_old) * (local_ref_coords - x_ref_old);

            // Return true if we converged.
            if ( error < tol2 )
                return true;
        }

        // Return false if we failed to converge.
        return false;
    }
};

//---------------------------------------------------------------------------//
/*!
  \class CurvilinearMesh
  \brief Logically rectilinear curvilinear mesh based on a given mapping.
 */
template <class Mapping>
class CurvilinearMesh
{
  public:

    using mesh_mapping = Mapping;

    using memory_space =
        typename CurvilinearMeshMapping<Mapping>::memory_space;

    static constexpr std::size_t num_space_dim =
        CurvilinearMeshMapping<Mapping>::num_space_dim;

    using cajita_mesh = Cajita::UniformMesh<double,num_space_dim>;

    using local_grid = Cajita::LocalGrid<cajita_mesh>;

    /*!
      \brief Constructor.
      \param mapping The mapping with which to build the curvilinear mesh.
      \param base_halo The number of cells required for grid-based halo
      operations.
      \param extended_halo The number of cells to use for extended halo
      (e.g. particle) operations. The extended halo and the base halo can be
      the same. If the extended halo is smaller than the base halo the base
      halo will be used.
      \param comm The MPI comm to use for the mesh.
      \param ranks_per_dim The number of MPI ranks to assign to each logical
      dimension in the partitioning.
    */
    CurvilinearMesh( const std::shared_ptr<Mapping>& mapping,
                     const int base_halo,
                     const int extended_halo,
                     MPI_Comm comm,
                     const std::array<int,num_space_dim>& ranks_per_dim )
        : _mapping( mapping )
        , _base_halo( base_halo )
    {
        // Extended halo must be at least as big as the base halo.
        _extended_halo = std::max( base_halo, extended_halo );

        // The logical grid is uniform with unit cell size.
        std::array<int, num_space_dim> global_num_cell;
        std::array<bool, num_space_dim> periodic;
        std::array<double, num_space_dim> global_low_corner;
        std::array<double, num_space_dim> global_high_corner;
        for ( std::size_t d = 0; d < num_space_dim; ++d )
        {
            global_num_cell[d] =
                CurvilinearMeshMapping<Mapping>::globalNumCell( *_mapping, d );
            periodic[d] =
                CurvilinearMeshMapping<Mapping>::periodic( *_mapping, d );
            global_low_corner[d] = 0.0;
            global_high_corner[d] = static_cast<double>(global_num_cell[d]);
        }

        // For dimensions that are not periodic we pad by the baso halo
        // cell width to allow for projections outside of the domain.
        for ( std::size_t d = 0; d < num_space_dim; ++d )
        {
            if ( !periodic[d] )
            {
                global_num_cell[d] += 2 * _base_halo;
                global_low_corner[d] -= _base_halo;
                global_high_corner[d] += _base_halo;
            }
        }

        // Create the global mesh.
        auto global_mesh = Cajita::createUniformGlobalMesh(
            global_low_corner, global_high_corner, global_num_cell );

        // Build the global grid.
        auto global_grid = Cajita::createGlobalGrid(
            comm, global_mesh, periodic,
            Cajita::ManualBlockPartitioner<num_space_dim>(ranks_per_dim) );

        // Build the local grid.
        _local_grid = Cajita::createLocalGrid( global_grid, _extended_halo );
    }

    // Get the number of cells in the base halo.
    int baseHalo() const { return _base_halo; }

    // Get the number of cells in the extended halo.
    int extendedHalo() const { return _extended_halo; }

    // Get the mesh mapping.
    const mesh_mapping& mapping() const { return *_mapping; }

    // Get the local grid.
    std::shared_ptr<local_grid> localGrid() const { return _local_grid; }

  public:
    std::shared_ptr<mesh_mapping> _mapping;
    int _base_halo;
    int _extended_halo;
    std::shared_ptr<local_grid> _local_grid;
};

//---------------------------------------------------------------------------//
// Static type checker.
template <class>
struct is_curvilinear_mesh_impl : public std::false_type
{
};

template <class Mapping>
struct is_curvilinear_mesh_impl<CurvilinearMesh<Mapping>> : public std::true_type
{
};

template <class T>
struct is_curvilinear_mesh
    : public is_curvilinear_mesh_impl<typename std::remove_cv<T>::type>::type
{
};

//---------------------------------------------------------------------------//
// Creation function.
template<class Mapping>
auto createCurvilinearMesh(
    const std::shared_ptr<Mapping>& mapping,
    const int base_halo,
    const int extended_halo,
    MPI_Comm comm,
    const std::array<int,Mapping::num_space_dim>& ranks_per_dim )
{
    return std::make_shared<CurvilinearMesh<Mapping>>(
        mapping, base_halo, extended_halo, comm, ranks_per_dim );
}

//---------------------------------------------------------------------------//

} // end namespace Picasso

#endif // end PICASSO_CURVILINEARMESH_HPP
