#include <deal.II/base/geometry_info.h>
#include <deal.II/base/mpi.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>

const MPI_Comm comm = MPI_COMM_WORLD;

using namespace dealii;

template<int dim>
void
extract_info(const parallel::distributed::Triangulation<dim> & tria,
             std::vector<CellData<dim>> &                      cells,
             std::vector<Point<dim>> &                         vertices,
             std::vector<int> &                                boundary_ids,
             std::map<int, std::pair<int, int>> &              coarse_lid_to_gid,
             std::vector<Part> &                               parts)
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



template<int dim>
void
test(int n_refinements, const int n_subdivisions, MPI_Comm comm)
{
  // create pdt
  parallel::distributed::Triangulation<dim> tria_pdt(
    comm,
    dealii::Triangulation<dim>::none,
    parallel::distributed::Triangulation<dim>::construct_multigrid_hierarchy);
  GridGenerator::subdivided_hyper_cube(tria_pdt, n_subdivisions);
  tria_pdt.refine_global(n_refinements);

  // extract relevant information from pdt to be able to create pft
  std::vector<CellData<dim>>         cells;
  std::vector<Point<dim>>            vertices;
  std::vector<int>                   boundary_ids;
  std::map<int, std::pair<int, int>> coarse_lid_to_gid;
  std::vector<Part>                  parts;

  extract_info(tria_pdt, cells, vertices, boundary_ids, coarse_lid_to_gid, parts);

  // create pft
  parallel::fullydistributed::Triangulation<dim> tria_pft(comm);
  tria_pft.reinit(cells, vertices, boundary_ids, coarse_lid_to_gid, parts, n_refinements);

  // output meshes as VTU
  GridOut grid_out;
  grid_out.write_mesh_per_processor_as_vtu(tria_pdt, "trid_pdt", true, true);
  grid_out.write_mesh_per_processor_as_vtu(tria_pft, "trid_pft", true, true);
}



int
main(int argc, char ** argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

  AssertThrow(argc > 3, ExcMessage("You have not provided two command-line arguments."));

  const int dim            = atoi(argv[1]);
  const int n_refinements  = atoi(argv[2]);
  const int n_subdivisions = atoi(argv[3]);

  if(dim == 2)
    test<2>(n_refinements, n_subdivisions, comm);
  else if(dim == 3)
    test<3>(n_refinements, n_subdivisions, comm);
  else
    AssertThrow(false, ExcMessage("Only working for dimensions 2 and 3!"));
}