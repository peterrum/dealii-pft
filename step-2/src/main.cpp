#include <deal.II/base/mpi.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/manifold_lib.h>

#include "../../common/utilities.h"

const MPI_Comm comm = MPI_COMM_WORLD;

using namespace dealii;



template<int dim>
void
test(int n_refinements, const int n_subdivisions, MPI_Comm comm)
{
  // create pdt
  parallel::distributed::Triangulation<dim> tria_pdt(
    comm,
    dealii::Triangulation<dim>::none,
    parallel::distributed::Triangulation<dim>::construct_multigrid_hierarchy);


  const Point<dim> center(1, 0);
  const double     inner_radius = 0.5, outer_radius = 1.0;
  GridGenerator::hyper_shell(tria_pdt, center, inner_radius, outer_radius, n_subdivisions);
  // tria_pdt.reset_all_manifolds ();
  for(int step = 0; step < n_refinements; ++step)
  {
    for(auto & cell : tria_pdt.active_cell_iterators())
    {
      // if(cell->is_locally_owned ())
      for(unsigned int v = 0; v < GeometryInfo<2>::vertices_per_cell; ++v)
      {
        const double distance_from_center = center.distance(cell->vertex(v));
        if(std::fabs(distance_from_center - inner_radius) < 1e-10)
        {
          cell->set_refine_flag();
          break;
        }
      }
    }
    tria_pdt.execute_coarsening_and_refinement();
  }

  // create instance of pft
  parallel::fullydistributed::Triangulation<dim> tria_pft(
    comm, parallel::fullydistributed::Triangulation<dim>::construct_multigrid_hierarchy);
  
  tria_pft.set_manifold(0, SphericalManifold<dim>(center)); 

  // extract relevant information form pdt
  auto construction_data =
    parallel::fullydistributed::Utilities::copy_from_triangulation(tria_pdt, tria_pft);

  // actually create triangulation
  tria_pft.reinit(construction_data);

  // test triangulation
  FE_Q<dim>       fe(2);
  DoFHandler<dim> dof_handler(tria_pft);
  dof_handler.distribute_dofs(fe);
  dof_handler.distribute_mg_dofs();

  // output meshes as VTU
  GridOut grid_out;
  grid_out.write_mesh_per_processor_as_vtu(tria_pdt, "trid_pdt", false, true);
  grid_out.write_mesh_per_processor_as_vtu(tria_pft, "trid_pft", false, true);
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