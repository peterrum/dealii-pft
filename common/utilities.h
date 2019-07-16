#ifndef PARALLEL_FULLY_DISTRIBUTED_MESH_UTIL
#define PARALLEL_FULLY_DISTRIBUTED_MESH_UTIL

#include <deal.II/base/geometry_info.h>
#include <deal.II/base/mpi.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_dgq.h>

using namespace dealii;


namespace dealii
{
namespace parallel
{
namespace fullydistributed
{
namespace Utilities
{
template<int dim>
void
copy_from_serial_triangulation(const dealii::Triangulation<dim> &   tria,
                               const MPI_Comm                       comm,
                               std::vector<CellData<dim>> &         cells,
                               std::vector<Point<dim>> &            vertices,
                               std::vector<int> &                   boundary_ids,
                               std::map<int, std::pair<int, int>> & coarse_lid_to_gid,
                               std::vector<Part> &                  parts)
{
  // 1) enumerate cells of original triangulation globally uniquelly
  FE_DGQ<dim>     fe(0);
  DoFHandler<dim> dof_handler(tria);
  dof_handler.distribute_dofs(fe);

  const unsigned int my_rank = dealii::Utilities::MPI::this_mpi_process(comm);

  // 2) collect vertices of active locally owned cells
  std::set<unsigned int> vertices_owned_by_loclly_owned_cells;
  for(auto cell : dof_handler.cell_iterators())
    if(cell->active() && (cell->is_locally_owned() && cell->subdomain_id() == my_rank))
      for(unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; v++)
        vertices_owned_by_loclly_owned_cells.insert(cell->vertex_index(v));

  // helper function to determine if cell is locally relevant
  auto is_ghost = [&](auto & cell) {
    for(unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; v++)
      if(vertices_owned_by_loclly_owned_cells.find(cell->vertex_index(v)) !=
         vertices_owned_by_loclly_owned_cells.end())
        return true;
    return false;
  };

  // 3) process all local and ghost cells: setup needed data structures and
  //    collect all locally relevant vertices for second sweep
  std::map<unsigned int, unsigned int> vertices_locally_relevant;
  parts.push_back(Part());
  Part & part = parts[0];

  unsigned int cell_counter = 0;
  for(auto cell : dof_handler.cell_iterators())
    if(cell->active() &&
       ((cell->is_locally_owned() && cell->subdomain_id() == my_rank) || is_ghost(cell)))
    {
      // a) extract cell definition (with old numbering of vertices)
      CellData<dim> cell_data;
      cell_data.material_id = cell->material_id();
      for(unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; v++)
        cell_data.vertices[v] = cell->vertex_index(v);
      cells.push_back(cell_data);

      // b) save indices of each vertex of this cell
      for(unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; v++)
        vertices_locally_relevant[cell->vertex_index(v)] = numbers::invalid_unsigned_int;

      // c) save boundary_ids of each face of this cell
      for(unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; f++)
        boundary_ids.push_back(cell->face(f)->boundary_id());

      // d) determine gid of this cell
      std::vector<types::global_dof_index> indices(1);
      cell->get_dof_indices(indices);

      // e) save translation for corase grid: lid -> gid
      coarse_lid_to_gid[cell_counter] = {indices[0], cell->subdomain_id()};

      // f) save level information
      if((cell->is_locally_owned() && cell->subdomain_id() == my_rank))
      {
        // gid of local cells
        part.local.push_back(indices[0]);
      }
      else
      {
        // gid, subdomain_id, and leve_subdomain_id of ghost cells
        part.ghost.push_back(indices[0]);
        part.ghost_rank.push_back(cell->subdomain_id());
        part.ghost_rank_mg.push_back(cell->level_subdomain_id());
      }
      cell_counter++;
    }

  // 4) enumerate locally relevant
  unsigned int vertex_counter = 0;
  for(auto & vertex : vertices_locally_relevant)
  {
    vertices.push_back(tria.get_vertices()[vertex.first]);
    vertex.second = vertex_counter++;
  }

  // 5) correct vertices of cells (make them local)
  for(auto & cell : cells)
    for(unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; v++)
      cell.vertices[v] = vertices_locally_relevant[cell.vertices[v]];
}



template<int dim>
void
copy_from_distributed_triangulation(const parallel::distributed::Triangulation<dim> & tria,
                                    std::vector<CellData<dim>> &                      cells,
                                    std::vector<Point<dim>> &                         vertices,
                                    std::vector<int> &                                boundary_ids,
                                    std::map<int, std::pair<int, int>> & coarse_lid_to_gid,
                                    std::vector<Part> &                  parts)
{
  // 1) enumerate cells of original triangulation globally uniquelly
  FE_DGQ<dim>     fe(0);
  DoFHandler<dim> dof_handler(tria);
  dof_handler.distribute_dofs(fe);

  // 2) collect vertices of active locally owned cells
  std::set<unsigned int> vertices_owned_by_loclly_owned_cells;
  for(auto cell : dof_handler.cell_iterators())
    if(cell->active() && cell->is_locally_owned())
      for(unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; v++)
        vertices_owned_by_loclly_owned_cells.insert(cell->vertex_index(v));

  // helper function to determine if cell is locally relevant
  auto is_ghost = [&](auto & cell) {
    for(unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; v++)
      if(vertices_owned_by_loclly_owned_cells.find(cell->vertex_index(v)) !=
         vertices_owned_by_loclly_owned_cells.end())
        return true;
    return false;
  };

  // 3) process all local and ghost cells: setup needed data structures and
  //    collect all locally relevant vertices for second sweep
  std::map<unsigned int, unsigned int> vertices_locally_relevant;
  parts.push_back(Part());
  Part & part = parts[0];

  unsigned int cell_counter = 0;
  for(auto cell : dof_handler.cell_iterators())
    if(cell->active() && (cell->is_locally_owned() || is_ghost(cell)))
    {
      // a) extract cell definition (with old numbering of vertices)
      CellData<dim> cell_data;
      cell_data.material_id = cell->material_id();
      for(unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; v++)
        cell_data.vertices[v] = cell->vertex_index(v);
      cells.push_back(cell_data);

      // b) save indices of each vertex of this cell
      for(unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; v++)
        vertices_locally_relevant[cell->vertex_index(v)] = numbers::invalid_unsigned_int;

      // c) save boundary_ids of each face of this cell
      for(unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; f++)
        boundary_ids.push_back(cell->face(f)->boundary_id());

      // d) determine gid of this cell
      std::vector<types::global_dof_index> indices(1);
      cell->get_dof_indices(indices);

      // e) save translation for corase grid: lid -> gid
      coarse_lid_to_gid[cell_counter] = {indices[0], cell->subdomain_id()};

      // f) save level information
      if(cell->is_locally_owned())
      {
        // gid of local cells
        part.local.push_back(indices[0]);
      }
      else
      {
        // gid, subdomain_id, and leve_subdomain_id of ghost cells
        part.ghost.push_back(indices[0]);
        part.ghost_rank.push_back(cell->subdomain_id());
        part.ghost_rank_mg.push_back(cell->level_subdomain_id());
      }
      cell_counter++;
    }

  // 4) enumerate locally relevant
  unsigned int vertex_counter = 0;
  for(auto & vertex : vertices_locally_relevant)
  {
    vertices.push_back(tria.get_vertices()[vertex.first]);
    vertex.second = vertex_counter++;
  }

  // 5) correct vertices of cells (make them local)
  for(auto & cell : cells)
    for(unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; v++)
      cell.vertices[v] = vertices_locally_relevant[cell.vertices[v]];
}



template<typename CELL>
void
set_flag_reverse(CELL cell)
{
  cell->set_user_flag();
  if(cell->level() != 0)
    set_flag_reverse(cell->parent());
}

template<int dim>
void
copy_from_serial_triangulation_mg(const dealii::Triangulation<dim> &   tria,
                                  const MPI_Comm                       comm,
                                  std::vector<CellData<dim>> &         cells,
                                  std::vector<Point<dim>> &            vertices,
                                  std::vector<int> &                   boundary_ids,
                                  std::map<int, std::pair<int, int>> & coarse_lid_to_gid,
                                  std::vector<Part> &                  parts)
{
  // 1) enumerate cells of original triangulation globally uniquelly
  FE_DGQ<dim>     fe(0);
  DoFHandler<dim> dof_handler(tria);
  dof_handler.distribute_dofs(fe);
  dof_handler.distribute_mg_dofs();

  const unsigned int my_rank = dealii::Utilities::MPI::this_mpi_process(comm);

  // 2) collect vertices of cells on level 0
  std::map<unsigned int, unsigned int> vertices_locally_relevant;

  for(auto cell : dof_handler.cell_iterators_on_level(0))
    cell->recursively_clear_user_flag();

  for(unsigned int level = dof_handler.get_triangulation().n_global_levels() - 1; level != 0;
      level--)
  {
    std::set<unsigned int> vertices_owned_by_loclly_owned_cells;
    for(auto cell : dof_handler.cell_iterators_on_level(level))
      if(cell->level_subdomain_id() == my_rank ||
         (cell->active() && cell->subdomain_id() == my_rank))
        for(unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; v++)
          vertices_owned_by_loclly_owned_cells.insert(cell->vertex_index(v));

    // helper function to determine if cell is locally relevant
    auto is_ghost = [&](auto & cell) {
      for(unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; v++)
        if(vertices_owned_by_loclly_owned_cells.find(cell->vertex_index(v)) !=
           vertices_owned_by_loclly_owned_cells.end())
          return true;
      return false;
    };

    for(auto cell : dof_handler.cell_iterators_on_level(level))
      if(is_ghost(cell))
        set_flag_reverse(cell);
  }

  unsigned int cell_counter = 0;
  for(auto cell : dof_handler.cell_iterators_on_level(0))
  {
    // a) determine gid of this cell
    std::vector<types::global_dof_index> indices(1);
    cell->get_mg_dof_indices(indices);

    if(!cell->user_flag_set())
      continue;

    // b) extract cell definition (with old numbering of vertices)
    CellData<dim> cell_data;
    cell_data.material_id = cell->material_id();
    for(unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; v++)
      cell_data.vertices[v] = cell->vertex_index(v);
    cells.push_back(cell_data);

    // c) save indices of each vertex of this cell
    for(unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; v++)
      vertices_locally_relevant[cell->vertex_index(v)] = numbers::invalid_unsigned_int;

    // d) save boundary_ids of each face of this cell
    for(unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; f++)
      boundary_ids.push_back(cell->face(f)->boundary_id());

    // e) save translation for corase grid: lid -> gid
    coarse_lid_to_gid[cell_counter] = {indices[0], cell->level_subdomain_id()};

    cell_counter++;
  }

  // 4) enumerate locally relevant
  unsigned int vertex_counter = 0;
  for(auto & vertex : vertices_locally_relevant)
  {
    vertices.push_back(tria.get_vertices()[vertex.first]);
    vertex.second = vertex_counter++;
  }

  // 5) correct vertices of cells (make them local)
  for(auto & cell : cells)
    for(unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; v++)
      cell.vertices[v] = vertices_locally_relevant[cell.vertices[v]];


  auto convert_binary_to_gid = [](const std::array<unsigned int, 4> binary_representation) {
    const unsigned int coarse_cell_id = binary_representation[0];

    const unsigned int n_child_indices = binary_representation[1] >> 2;

    const unsigned int children_per_value = sizeof(CellId::binary_type::value_type) * 8 / dim;
    unsigned int       child_level        = 0;
    unsigned int       binary_entry       = 2;

    std::vector<unsigned int> cell_indices;

    // Loop until all child indices have been written
    while(child_level < n_child_indices)
    {
      Assert(binary_entry < binary_representation.size(), ExcInternalError());

      for(unsigned int j = 0; j < children_per_value; ++j)
      {
        // const unsigned int child_index =
        //  static_cast<unsigned int>(child_indices[child_level]);
        // Shift the child index to its position in the unsigned int and store
        // it
        unsigned int cell_index = (((binary_representation[binary_entry] >> (j * dim))) &
                                   (GeometryInfo<dim>::max_children_per_cell - 1));
        cell_indices.push_back(cell_index);
        ++child_level;
        if(child_level == n_child_indices)
          break;
      }
      ++binary_entry;
    }

    unsigned int temp = coarse_cell_id;
    for(auto i : cell_indices)
    {
      temp = temp * GeometryInfo<dim>::max_children_per_cell + i;
    }

    return temp;
  };

  for(unsigned int level = 0; level < dof_handler.get_triangulation().n_global_levels(); level++)
  {
    parts.push_back(Part());
    Part & part = parts.back();
    for(auto cell : dof_handler.cell_iterators_on_level(level))
    {
      // a) determine gid of this cell
      std::vector<types::global_dof_index> indices(1);
      cell->get_mg_dof_indices(indices);

      if(!cell->user_flag_set())
        continue;

      const unsigned int index = convert_binary_to_gid(cell->id().template to_binary<dim>());

      if(cell->is_locally_owned_on_level())
      {
        // gid of local cells
        part.local.push_back(index);
      }
      else
      {
        // gid, subdomain_id, and leve_subdomain_id of ghost cells
        part.ghost.push_back(index);

        if(cell->active())
          part.ghost_rank.push_back(cell->subdomain_id());
        else
          part.ghost_rank.push_back(cell->level_subdomain_id());

        part.ghost_rank_mg.push_back(cell->level_subdomain_id());
      }
    }
  }
}



template<int dim>
void
copy_from_distributed_triangulation_mg(const parallel::distributed::Triangulation<dim> & tria,
                                       std::vector<CellData<dim>> &                      cells,
                                       std::vector<Point<dim>> &                         vertices,
                                       std::vector<int> &                   boundary_ids,
                                       std::map<int, std::pair<int, int>> & coarse_lid_to_gid,
                                       std::vector<Part> &                  parts)
{
  // 1) enumerate cells of original triangulation globally uniquelly
  FE_DGQ<dim>     fe(0);
  DoFHandler<dim> dof_handler(tria);
  dof_handler.distribute_dofs(fe);
  dof_handler.distribute_mg_dofs();

  // 2) collect vertices of cells on level 0
  std::map<unsigned int, unsigned int> vertices_locally_relevant;

  unsigned int cell_counter = 0;
  for(auto cell : dof_handler.cell_iterators_on_level(0))
  {
    // a) determine gid of this cell
    std::vector<types::global_dof_index> indices(1);
    cell->get_mg_dof_indices(indices);

    if(indices[0] == numbers::invalid_dof_index)
      continue;

    // b) extract cell definition (with old numbering of vertices)
    CellData<dim> cell_data;
    cell_data.material_id = cell->material_id();
    for(unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; v++)
      cell_data.vertices[v] = cell->vertex_index(v);
    cells.push_back(cell_data);

    // c) save indices of each vertex of this cell
    for(unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; v++)
      vertices_locally_relevant[cell->vertex_index(v)] = numbers::invalid_unsigned_int;

    // d) save boundary_ids of each face of this cell
    for(unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; f++)
      boundary_ids.push_back(cell->face(f)->boundary_id());

    // e) save translation for corase grid: lid -> gid
    coarse_lid_to_gid[cell_counter] = {indices[0], cell->level_subdomain_id()};

    cell_counter++;
  }

  // 4) enumerate locally relevant
  unsigned int vertex_counter = 0;
  for(auto & vertex : vertices_locally_relevant)
  {
    vertices.push_back(tria.get_vertices()[vertex.first]);
    vertex.second = vertex_counter++;
  }

  // 5) correct vertices of cells (make them local)
  for(auto & cell : cells)
    for(unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; v++)
      cell.vertices[v] = vertices_locally_relevant[cell.vertices[v]];


  auto convert_binary_to_gid = [](const std::array<unsigned int, 4> binary_representation) {
    const unsigned int coarse_cell_id = binary_representation[0];

    const unsigned int n_child_indices = binary_representation[1] >> 2;

    const unsigned int children_per_value = sizeof(CellId::binary_type::value_type) * 8 / dim;
    unsigned int       child_level        = 0;
    unsigned int       binary_entry       = 2;

    std::vector<unsigned int> cell_indices;

    // Loop until all child indices have been written
    while(child_level < n_child_indices)
    {
      Assert(binary_entry < binary_representation.size(), ExcInternalError());

      for(unsigned int j = 0; j < children_per_value; ++j)
      {
        // const unsigned int child_index =
        //  static_cast<unsigned int>(child_indices[child_level]);
        // Shift the child index to its position in the unsigned int and store
        // it
        unsigned int cell_index = (((binary_representation[binary_entry] >> (j * dim))) &
                                   (GeometryInfo<dim>::max_children_per_cell - 1));
        cell_indices.push_back(cell_index);
        ++child_level;
        if(child_level == n_child_indices)
          break;
      }
      ++binary_entry;
    }

    unsigned int temp = coarse_cell_id;
    for(auto i : cell_indices)
    {
      temp = temp * GeometryInfo<dim>::max_children_per_cell + i;
    }

    return temp;
  };

  for(unsigned int level = 0; level < dof_handler.get_triangulation().n_global_levels(); level++)
  {
    parts.push_back(Part());
    Part & part = parts.back();
    for(auto cell : dof_handler.cell_iterators_on_level(level))
    {
      // a) determine gid of this cell
      std::vector<types::global_dof_index> indices(1);
      cell->get_mg_dof_indices(indices);

      if(indices[0] == numbers::invalid_dof_index)
        continue;

      const unsigned int index = convert_binary_to_gid(cell->id().template to_binary<dim>());

      if(cell->is_locally_owned_on_level())
      {
        // gid of local cells
        part.local.push_back(index);
      }
      else
      {
        // gid, subdomain_id, and leve_subdomain_id of ghost cells
        part.ghost.push_back(index);

        if(cell->active())
          part.ghost_rank.push_back(cell->subdomain_id());
        else
          part.ghost_rank.push_back(cell->level_subdomain_id());

        part.ghost_rank_mg.push_back(cell->level_subdomain_id());
      }
    }
  }
}

} // namespace Utilities
} // namespace fullydistributed
} // namespace parallel
} // namespace dealii

#endif