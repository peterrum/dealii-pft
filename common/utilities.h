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
template<typename CELL>
void
set_flag_reverse(CELL cell)
{
  cell->set_user_flag();
  if(cell->level() != 0)
    set_flag_reverse(cell->parent());
}


template<int dim>
unsigned int
convert_binary_to_gid(const std::array<unsigned int, 4> binary_representation)
{
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
}

template<int dim, int spacedim = dim>
ConstructionData<dim, spacedim>
copy_from_serial_triangulation(const dealii::Triangulation<dim, spacedim> & tria,
                               const Triangulation<dim, spacedim> &         tria_pft)
{
  const auto                      comm = tria_pft.get_communicator();
  ConstructionData<dim, spacedim> cd;

  auto & cells             = cd.cells;
  auto & vertices          = cd.vertices;
  auto & boundary_ids      = cd.boundary_ids;
  auto & coarse_lid_to_gid = cd.coarse_lid_to_gid;
  auto & parts             = cd.parts;

  if(!tria_pft.do_construct_multigrid_hierarchy())
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
  else
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
      coarse_lid_to_gid[cell_counter] = {convert_binary_to_gid<dim>(
                                           cell->id().template to_binary<dim>()),
                                         cell->level_subdomain_id()};

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

    std::map<int, int> coarse_gid_to_lid;
    for(auto i : coarse_lid_to_gid)
      coarse_gid_to_lid[i.second.first] = i.first;

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

        auto id = cell->id().template to_binary<dim>();
        id[0]   = coarse_gid_to_lid[id[0]];

        if(cell->active())
          part.cells.emplace_back(id, cell->subdomain_id(), cell->level_subdomain_id());
        else
          part.cells.emplace_back(id, numbers::invalid_subdomain_id, cell->level_subdomain_id());
      }

      std::sort(part.cells.begin(), part.cells.end(), [](auto a, auto b) {
        return convert_binary_to_gid<dim>(a.index) < convert_binary_to_gid<dim>(b.index);
      });
    }
  }

  return cd;
}



template<int dim, int spacedim = dim>
ConstructionData<dim, spacedim>
copy_from_distributed_triangulation(
  const parallel::distributed::Triangulation<dim, spacedim> & tria,
  const Triangulation<dim, spacedim> &                        tria_pft)
{
  ConstructionData<dim, spacedim> cd;

  auto & cells             = cd.cells;
  auto & vertices          = cd.vertices;
  auto & boundary_ids      = cd.boundary_ids;
  auto & coarse_lid_to_gid = cd.coarse_lid_to_gid;
  auto & parts             = cd.parts;

  if(!tria_pft.do_construct_multigrid_hierarchy())
  {
    // 2) collect vertices of active locally owned cells
    std::set<unsigned int> vertices_owned_by_loclly_owned_cells;
    for(auto cell : tria.cell_iterators())
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
    for(auto cell : tria.cell_iterators())
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

        // e) save translation for corase grid: lid -> gid
        coarse_lid_to_gid[cell_counter] = {convert_binary_to_gid<dim>(
                                             cell->id().template to_binary<dim>()),
                                           numbers::invalid_subdomain_id};

        CellId::binary_type id;
        id.fill(0);
        id[0] = cell_counter;
        id[1] = dim;
        id[2] = 0;
        id[3] = 0;

        part.cells.emplace_back(id, cell->subdomain_id(), numbers::invalid_subdomain_id);

        cell_counter++;
      }

    std::sort(part.cells.begin(), part.cells.end(), [](auto a, auto b) {
      return convert_binary_to_gid<dim>(a.index) < convert_binary_to_gid<dim>(b.index);
    });

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
  else
  {
    MPI_Comm           comm    = MPI_COMM_WORLD;
    const unsigned int my_rank = dealii::Utilities::MPI::this_mpi_process(comm);

    for(auto cell : tria.cell_iterators_on_level(0))
      cell->recursively_clear_user_flag();

    for(unsigned int level = tria.get_triangulation().n_global_levels() - 1;
        level != numbers::invalid_unsigned_int;
        level--)
    {
      std::set<unsigned int> vertices_owned_by_loclly_owned_cells;
      for(auto cell : tria.cell_iterators_on_level(level))
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

      for(auto cell : tria.cell_iterators_on_level(level))
        if(is_ghost(cell))
          set_flag_reverse(cell);
    }



    // 2) collect vertices of cells on level 0
    std::map<unsigned int, unsigned int> vertices_locally_relevant;

    unsigned int cell_counter = 0;
    for(auto cell : tria.cell_iterators_on_level(0))
    {
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
      coarse_lid_to_gid[cell_counter] = {convert_binary_to_gid<dim>(
                                           cell->id().template to_binary<dim>()),
                                         cell->level_subdomain_id()};

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


    std::map<int, int> coarse_gid_to_lid;
    for(auto i : coarse_lid_to_gid)
      coarse_gid_to_lid[i.second.first] = i.first;

    for(unsigned int level = 0; level < tria.get_triangulation().n_global_levels(); level++)
    {
      std::set<unsigned int> vertices_owned_by_loclly_owned_cells;
      for(auto cell : tria.cell_iterators_on_level(level))
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



      parts.push_back(Part());
      Part & part = parts.back();
      for(auto cell : tria.cell_iterators_on_level(level))
      {
        if(!is_ghost(cell))
          continue;

        auto id = cell->id().template to_binary<dim>();
        id[0]   = coarse_gid_to_lid[id[0]];

        if(cell->active())
          part.cells.emplace_back(id, cell->subdomain_id(), cell->level_subdomain_id());
        else
          part.cells.emplace_back(id, numbers::invalid_subdomain_id, cell->level_subdomain_id());
      }

      std::sort(part.cells.begin(), part.cells.end(), [](auto a, auto b) {
        return convert_binary_to_gid<dim>(a.index) < convert_binary_to_gid<dim>(b.index);
      });
    }
  }

  return cd;
}

} // namespace Utilities
} // namespace fullydistributed
} // namespace parallel
} // namespace dealii

#endif