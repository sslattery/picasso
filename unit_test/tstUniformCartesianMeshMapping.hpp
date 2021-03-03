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

#include <Picasso_Types.hpp>
#include <Picasso_UniformCartesianMeshMapping.hpp>
#include <Picasso_CurvilinearMesh.hpp>

#include <Kokkos_Core.hpp>

#include <cmath>

#include <gtest/gtest.h>

using namespace Picasso;

namespace Test
{
//---------------------------------------------------------------------------//
void constructionTest()
{
    // Global parameters.
    Kokkos::Array<double, 6> global_box = { -10.0, -10.0, -10.0,
                                            10.0,  10.0,  10.0 };
    double cell_size = 0.5;
    int num_cell = 20.0 / cell_size;
    int base_halo = 1;
    int extended_halo = 0;
    Kokkos::Array<bool,3> periodic = { true, false, true };

    // Partition only in the x direction.
    std::array<int,3> ranks_per_dim = { 1, 1, 1 };
    MPI_Comm_size( MPI_COMM_WORLD, &ranks_per_dim[0] );

    // Create the mesh and field manager.
    auto manager =
        createUniformCartesianMesh( TEST_MEMSPACE{}, cell_size,
                                    global_box, periodic,
                                    base_halo, extended_halo,
                                    MPI_COMM_WORLD, ranks_per_dim );

    // Check the mapping.
    auto mapping = manager->mesh().mapping();
    EXPECT_EQ( cell_size, mapping._cell_size );
    EXPECT_EQ( 1.0 / cell_size, mapping._inv_cell_size );
    EXPECT_EQ( cell_size * cell_size * cell_size, mapping._cell_measure );
    EXPECT_EQ( 1.0/ (cell_size * cell_size * cell_size), mapping._inv_cell_measure );
    for ( int d = 0; d < 3; ++d )
    {
        EXPECT_EQ( num_cell, mapping._global_num_cell[d] );
        EXPECT_EQ( periodic[d], mapping._periodic[d] );
    }

    // Check mesh.
    EXPECT_EQ( base_halo, manager->mesh().baseHalo() );
    EXPECT_EQ( base_halo, manager->mesh().extendedHalo() );

    // Check grid.
    const auto& global_grid = manager->mesh().localGrid()->globalGrid();
    const auto& global_mesh = global_grid.globalMesh();

    EXPECT_EQ( global_mesh.lowCorner( 0 ), global_box[0] );
    EXPECT_EQ( global_mesh.lowCorner( 1 ), global_box[1] - cell_size );
    EXPECT_EQ( global_mesh.lowCorner( 2 ), global_box[2] );

    EXPECT_EQ( global_mesh.highCorner( 0 ), global_box[3] );
    EXPECT_EQ( global_mesh.highCorner( 1 ), global_box[4] + cell_size );
    EXPECT_EQ( global_mesh.highCorner( 2 ), global_box[5] );

    EXPECT_EQ( global_grid.globalNumEntity( Cajita::Cell(), 0 ), num_cell );
    EXPECT_EQ( global_grid.globalNumEntity( Cajita::Cell(), 1 ),
               num_cell + 2 );
    EXPECT_EQ( global_grid.globalNumEntity( Cajita::Cell(), 2 ), num_cell );

    EXPECT_TRUE( global_grid.isPeriodic( 0 ) );
    EXPECT_FALSE( global_grid.isPeriodic( 1 ) );
    EXPECT_TRUE( global_grid.isPeriodic( 2 ) );

    EXPECT_EQ( manager->mesh().localGrid()->haloCellWidth(), 1 );
}

//---------------------------------------------------------------------------//
// RUN TESTS
//---------------------------------------------------------------------------//
TEST( TEST_CATEGORY, construction_test ) { constructionTest(); }

//---------------------------------------------------------------------------//

} // end namespace Test
