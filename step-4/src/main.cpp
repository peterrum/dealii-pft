#include <deal.II/base/mpi.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/manifold_lib.h>

#include <deal.II/distributed/fullydistributed_tria.h>
#include <deal.II/distributed/fullydistributed_tria_util.h>

const MPI_Comm comm = MPI_COMM_WORLD;

using namespace dealii;



template<int dim>
void
test(int n_refinements, const int n_subdivisions, MPI_Comm comm)
{
  // create pdt
  Triangulation<dim> basetria(Triangulation<dim>::limit_level_difference_at_vertices);


  const Point<dim> center(1, 0);
  const double     inner_radius = 0.5, outer_radius = 1.0;
  GridGenerator::hyper_shell(basetria, center, inner_radius, outer_radius, n_subdivisions);
  // basetria.reset_all_manifolds ();
  for(int step = 0; step < n_refinements; ++step)
  {
    for(auto & cell : basetria.active_cell_iterators())
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
    basetria.execute_coarsening_and_refinement();
  }

  GridTools::partition_triangulation(Utilities::MPI::n_mpi_processes(comm),
                                     basetria,
                                     SparsityTools::Partitioner::metis);
  // if(n_refinements!=0)
  GridTools::partition_multigrid_levels(basetria);

  // for(auto cell : basetria){
  //    std::cout << cell.id().to_string();
  //    std::cout  << " " << cell.level_subdomain_id();
  //    if(cell.active())
  //      std::cout << " " << cell.subdomain_id();
  //
  //    std::cout << std::endl;
  //}

  // else
  //  for(auto cell : basetria)
  //      cell.set_level_subdomain_id(numbers::artificial_subdomain_id);

  GridOut grid_out;
  grid_out.write_mesh_per_processor_as_vtu(basetria, "trid_pdt", false, true);

  // create instance of pft
  parallel::fullydistributed::Triangulation<dim> tria_pft(
    comm, parallel::fullydistributed::Triangulation<dim>::construct_multigrid_hierarchy);

  tria_pft.set_manifold(0, SphericalManifold<dim>(center));

  // extract relevant information form pdt
  auto construction_data =
    parallel::fullydistributed::Utilities::copy_from_triangulation(basetria, tria_pft);

  // actually create triangulation
  tria_pft.reinit(construction_data);


  // test triangulation
  FE_Q<dim>       fe(2);
  DoFHandler<dim> dof_handler(tria_pft);
  dof_handler.distribute_dofs(fe);
  dof_handler.distribute_mg_dofs();

  // output meshes as VTU
  grid_out.write_mesh_per_processor_as_vtu(tria_pft, "trid_pft", false, false);
}



int
main(int argc, char ** argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

  AssertThrow(argc > 3, ExcMessage("You have not provided two command-line arguments."));

  const int dim            = atoi(argv[1]);
  const int n_refinements  = atoi(argv[2]);
  const int n_subdivisions = atoi(argv[3]);

  ConditionalOStream pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(comm) == 0);

  try
  {
    // clang-format off
    pcout << "Run step-4:"
          << " p=" << std::setw(2) << dealii::Utilities::MPI::n_mpi_processes(comm)
          << " d=" << std::setw(2) << dim
          << " r=" << std::setw(2) << n_refinements
          << " s=" << std::setw(2) << n_subdivisions
          << ":";
    // clang-format on

    if(dim == 2)
      test<2>(n_refinements, n_subdivisions, comm);
    else if(dim == 3)
      test<3>(n_refinements, n_subdivisions, comm);
    else
      AssertThrow(false, ExcMessage("Only working for dimensions 2 and 3!"));
    pcout << " success...." << std::endl;
  }
  catch(...)
  {
    pcout << " failed...." << std::endl;
  }
}
