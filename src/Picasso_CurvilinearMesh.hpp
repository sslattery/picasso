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

#include <array>
#include <type_traits>
#include <memory>

#include <mpi.h>

namespace Picasso
{
//---------------------------------------------------------------------------//
/*!
  \class CurvilinearMesh
  \brief Logically rectilinear curvilinear mesh.
 */
template <class MemorySpace, std::size_t NumSpaceDim>
class CurvilinearMesh
{
  public:
    using memory_space = MemorySpace;

    static constexpr std::size_t num_space_dim = NumSpaceDim;

    using cajita_mesh = Cajita::UniformMesh<double,NumSpaceDim>;

    using local_grid = Cajita::LocalGrid<cajita_mesh>;

    /*!
      \brief Constructor.
      \param global_num_cell The global number of cells to put in each logical
      dimension.
      \param periodic The peridocity of each logical dimension.
      \param halo_width The number of cells to use for the domain
      decomposition halo.
      \param boundary_padding The number of cells to pad non-periodic physical
      boundaries with.
      \param comm The MPI comm to use for the mesh.
      \param ranks_per_dim The number of MPI ranks to assign to each logical
      dimension in the partitioning.
    */
    template <class ExecutionSpace>
    CurvilinearMesh( const std::array<int,NumSpaceDim>& global_num_cell,
                     const std::array<bool,NumSpaceDim>& periodic,
                     const int boundary_padding,
                     const int halo_width,
                     MPI_Comm comm,
                     const std::array<int,NumSpaceDim>& ranks_per_dim )
        : _boundary_padding( minimum_halo_cell_width )
    {
        // The logical grid is uniform with unit cell size.
        std::array<double, NumSpaceDim> global_low_corner;
        std::array<double, NumSpaceDim> global_high_corner;
        for ( std::size_t d = 0; d < NumSpaceDim; ++d )
        {
            global_low_corner[d] = 0.0;
            global_high_corner[d] = static_cast<double>(global_num_cell[d]);
        }

        // For dimensions that are not periodic we pad by the minimum halo
        // cell width to allow for projections outside of the domain.
        for ( std::size_t d = 0; d < NumSpaceDim; ++d )
        {
            if ( !periodic[d] )
            {
                global_num_cell[d] += 2 * _boundary_padding;
                global_low_corner[d] -= _boundary_padding;
                global_high_corner[d] += _boundary_padding;
            }
        }

        // Create the global mesh.
        auto global_mesh = Cajita::createUniformGlobalMesh(
            global_low_corner, global_high_corner, global_num_cell );

        // Build the global grid.
        auto global_grid = Cajita::createGlobalGrid(
            comm, global_mesh, periodic,
            Cajita::ManualBlockPartitioner<NumSpaceDim>(ranks_per_dim) );

        // Build the local grid.
        _local_grid = Cajita::createLocalGrid( global_grid, halo_width );
    }

    // Get the physical boundary padding.
    int boundaryPadding() const { return _boundary_padding; }

    // Get the local grid.
    std::shared_ptr<local_grid> localGrid() const { return _local_grid; }

  public:
    int _boundary_padding;
    std::shared_ptr<local_grid> _local_grid;
};

//---------------------------------------------------------------------------//
// Static type checker.
template <class>
struct is_curvilinear_mesh_impl : public std::false_type
{
};

template <class MemorySpace, std::size_t NumSpaceDim>
struct is_curvilinear_mesh_impl<CurvilinearMesh<MemorySpace,NumSpaceDim>> : public std::true_type
{
};

template <class T>
struct is_curvilinear_mesh
    : public is_curvilinear_mesh_impl<typename std::remove_cv<T>::type>::type
{
};

//---------------------------------------------------------------------------//
// Creation function.
template<class MemorySpace,std::size_t NumSpaceDim>
auto createCurvilinearMesh(
    MemorySpace,
    SpaceDim<NumSpaceDim>,
    const std::array<int,NumSpaceDim>& global_num_cell,
    const std::array<bool,NumSpaceDim>& periodic,
    const int boundary_padding,
    const int halo_width,
    MPI_Comm comm,
    const std::array<int,NumSpaceDim>& ranks_per_dim )
{
    return std::make_shared<CurvilinearMesh<MemorySpace,NumSpaceDim>>(
        global_num_cell, periodic, boundary_padding, halo_width,
        comm, ranks_per_dim );
}

//---------------------------------------------------------------------------//
// Curvilinear mesh mapping function template interface.
//
// Physical frame - the physical representation in the chosen mapping
// coordinate system.
//
// Reference frame - the logical representation in the local grid reference
// frame. Combined with global mesh information, the mesh mapping
// implementation can convert this to a logical representation in the global
// grid reference frame.
//
template <class Mapping>
class CurvilinearMeshMapping
{
  public:

    // Spatial dimension.
    static constexpr std::size_t num_space_dim = Mapping::num_space_dim;

    // Forward mapping. Given coordinates in the local reference frame compute the
    // coordinates in the physical frame.
    template<class ReferenceCoords, class PhysicalCoords>
    static KOKKOS_INLINE_FUNCTION void
    mapToPhysicalFrame( const ReferenceCoords& local_ref_coords,
                        PhysicalCoords& physical_coords );

    // Reverse mapping. Given coordinates in the physical frame compute the
    // coordinates in the local reference frame.
    template<class PhysicalCoords, class ReferenceCoords>
    static KOKKOS_INLINE_FUNCTION void
    mapToReferenceFrame( const PhysicalCoords& physical_coords,
                         ReferenceCoords& local_ref_coords );

    // Given coordinates in the local reference frame compute the Jacobian of
    // the forward mapping.
    template<class ReferenceCoords>
    static KOKKOS_INLINE_FUNCTION void
    jacobian( const ReferenceCoords& local_ref_coords,
              LinearAlgebra::Matrix<typename ReferenceCoords::value_type,
              num_space_dim,num_space_dim>& jacobian );

    // Given coordinates in the local reference frame compute the determinant of
    // the Jacobian of the forward mapping.
    template<class ReferenceCoords>
    static KOKKOS_INLINE_FUNCTION void
    jacobianDeterminant( const ReferenceCoords& local_ref_coords,
                         typename ReferenceCoordinates::value_type& jacobian_det );

    // Given coordinates in the local reference frame compute the inverse of
    // the Jacobian of the forward mapping.
    template<class ReferenceCoords>
    static KOKKOS_INLINE_FUNCTION void
    jacobianInverse( const ReferenceCoords& local_ref_coords,
                     LinearAlgebra::Matrix<typename ReferenceCoords::value_type,
                     num_space_dim,num_space_dim>& jacobian_inv );
};

//---------------------------------------------------------------------------//

} // end namespace Picasso

#endif // end PICASSO_CURVILINEARMESH_HPP
