#ifndef HARLOW_GRIDBLOCK_HPP
#define HARLOW_GRIDBLOCK_HPP

#include <vector>

namespace Harlow
{
//---------------------------------------------------------------------------//
// Local Cartesian grid block representation.
//
// Note that a block always has a halo - even if it is on a physical
// boundary that is not periodic. We do this to facilitate particle deposition
// at boundaries.
//
// As a result, the boundary node/cell on a physical boundary that is not
// periodic is the first/last local node/cell depending on whether the low or
// high boundary is chosen.
//---------------------------------------------------------------------------//
class GridBlock
{
  public:

    // Default constructor.
    GridBlock();

    /*!
      \brief Constructor.
      \param local_low_corner The low corner of the space locally owned by the
      block.
      \param local_num_cell The number of cells in each dimension locally
      owned by the block.
      \param boundary_location Boolean indicating if the block is on any of
      the 6 physical boundaries {-x,+x,-y,+y,-z,+z}.
      \param is_dim_periodic Whether each logical dimension is periodic.
      \param cell_size The size of the cells in the mesh. The cells are cubes.
      \param halo_cell_width The number of halo cells surrounding the locally
      owned cells.
    */
    GridBlock( const std::vector<double>& local_low_corner,
               const std::vector<int>& local_num_cell,
               const std::vector<bool>& boundary_location,
               const std::vector<bool>& is_dim_periodic,
               const double cell_size,
               const int halo_cell_width );

    // Assign the local state of a cartesian grid block with a new halo size.
    void assign( const GridBlock& rhs, const int halo_cell_width );

    // Get the physical coordinates of the low corner of the grid in a given
    // dimension. This low corner includes the halo region.
    double lowCorner( const int dim ) const;

    // Given a physical boundary id return if this grid is on that boundary.
    bool onBoundary( const int boundary_id ) const;

    // Get whether a given logical dimension is periodic.
    bool isPeriodic( const int dim ) const;

    // Given a boundary id return if this has a halo on that boundary. This
    // will be true when this block is not on the domain boundary or, if it
    // is, that boundary is periodic.
    bool hasHalo( const int boundary_id ) const;

    // Get the cell size.
    double cellSize() const;

    // Get the inverse cell size.
    double inverseCellSize() const;

    // Get the halo size.
    int haloSize() const;

    // Get the total number of cells in a given dimension including the halo.
    int numCell( const int dim ) const;

    // Get the total number of nodes in a given dimension including the halo.
    int numNode( const int dim ) const;

    // Get the beginning local cell index in a given direction. The local
    // cells do not include the halo.
    int localCellBegin( const int dim ) const;

    // Get the ending local cell index in a given direction. The local cells
    // do not include the halo.
    int localCellEnd( const int dim ) const;

    // Get the beginning local node index in a given direction. The local
    // nodes do not include the halo. A local grid block always "owns" the
    // node on the negative logical boundary.
    int localNodeBegin( const int dim ) const;

    // Get the ending local node index in a given direction. The local nodes
    // do not include the halo. The local grid block does not "own" the node
    // on the high logical boundary unless the high logical boundary is also a
    // physical boundary that is not periodic.
    int localNodeEnd( const int dim ) const;

    // Get the local number of cells in a given dimension.
    int localNumCell( const int dim ) const;

    // Get the local number of nodes in a given dimension.
    int localNumNode( const int dim ) const;

  private:

    // Set the halo.
    void setHalo();

  private:

    std::vector<double> _local_low_corner;
    std::vector<double> _low_corner;
    std::vector<int> _local_num_cell;
    std::vector<int> _total_num_cell;
    std::vector<bool> _boundary_location;
    std::vector<bool> _periodic;
    double _cell_size;
    int _halo_cell_width;
};

//---------------------------------------------------------------------------//

} // end namespace Harlow

#endif // end HARLOW_GRIDBLOCK_HPP
