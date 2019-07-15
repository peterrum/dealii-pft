#include <deal.II/base/geometry_info.h>
#include <deal.II/base/mpi.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>

const MPI_Comm comm = MPI_COMM_WORLD;

using namespace dealii;

template<int dim>
void
extract_info(const Triangulation<dim> &           tria,
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

  const unsigned int my_rank = Utilities::MPI::this_mpi_process(comm);

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
test(int n_refinements, MPI_Comm comm)
{
  // create pdt
  Triangulation<dim> basetria;
  GridGenerator::hyper_L(basetria);
  basetria.refine_global(n_refinements);

  GridTools::partition_triangulation(Utilities::MPI::n_mpi_processes(comm),
                                     basetria,
                                     SparsityTools::Partitioner::metis);
  GridTools::partition_multigrid_levels(basetria);

  // extract relevant information from pdt to be able to create pft
  std::vector<CellData<dim>>         cells;
  std::vector<Point<dim>>            vertices;
  std::vector<int>                   boundary_ids;
  std::map<int, std::pair<int, int>> coarse_lid_to_gid;
  std::vector<Part>                  parts;

  extract_info(basetria, comm, cells, vertices, boundary_ids, coarse_lid_to_gid, parts);

  // create pft
  parallel::fullydistributed::Triangulation<dim> tria_pft(comm);
  tria_pft.reinit(cells, vertices, boundary_ids, coarse_lid_to_gid, parts);

  // output meshes as VTU
  GridOut grid_out;
  grid_out.write_mesh_per_processor_as_vtu(basetria, "trid_pdt", true, true);
  grid_out.write_mesh_per_processor_as_vtu(tria_pft, "trid_pft", true, true);
}



int
main(int argc, char ** argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

  AssertThrow(argc > 2, ExcMessage("You have not provided two command-line arguments."));

  const int dim           = atoi(argv[1]);
  const int n_refinements = atoi(argv[2]);

  if(dim == 2)
    test<2>(n_refinements, comm);
  else if(dim == 3)
    test<3>(n_refinements, comm);
  else
    AssertThrow(false, ExcMessage("Only working for dimensions 2 and 3!"));
}