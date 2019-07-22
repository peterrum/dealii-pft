#ifndef PARALLEL_FULLY_DISTRIBUTED_GRID_TOOLS
#define PARALLEL_FULLY_DISTRIBUTED_GRID_TOOLS

#include <ostream>

#ifdef DEAL_II_WITH_METIS
#  include <metis.h>
#endif

namespace dealii
{
namespace GridTools
{
class Graph
{
public:
  std::vector<int> xadj;
  std::vector<int> adjncy;
  std::vector<int> weights;
  int              elements;
  std::vector<int> parts;

  void
  print(std::ostream & out);
};

class PartitioningAlgorithm
{
public:
  virtual void
  mesh_to_dual(std::vector<int> & eptr_in,
               std::vector<int> & eind_in,
               int                ncommon_in,
               Graph &            graph_out);

  virtual void
  partition(Graph & graph, int n_partitions, bool is_prepartitioned = false);
};

class MetisPartitioningAlgorithm : public PartitioningAlgorithm
{
public:
  void
  mesh_to_dual(std::vector<int> & eptr_in,
               std::vector<int> & eind_in,
               int                ncommon_in,
               Graph &            graph_out) override;

  void
  partition(Graph & graph, int n_partitions, bool is_prepartitioned = false) override;
};

enum PartitioningType
{
  metis
};
enum PartitioningGroup
{
  single,
  fixed,
  shared
};

struct AdditionalData
{
  AdditionalData() : partition_type(metis), partition_group(fixed), partition_group_size(4)
  {
  }

  PartitioningType  partition_type;
  PartitioningGroup partition_group;
  unsigned int      partition_group_size;

  unsigned int size_all    = 1;
  unsigned int size_groups = 1;
  unsigned int size_node   = 1;
  unsigned int size_coarse = 1;
};

template<int dim, int spacedim>
void
shared_partition_triangulation(dealii::Triangulation<dim, spacedim> & tria,
                               AdditionalData additional_data = AdditionalData())
{
  const unsigned int size_all       = additional_data.size_all;
  const unsigned int size_groups    = additional_data.size_groups;
  const unsigned int size_node      = additional_data.size_node;
  const unsigned int size_coarse    = additional_data.size_coarse;
  const bool         do_repartition = size_coarse != size_all;

  // retrieve number of refinements from triangulation and send number
  // of refinements
  unsigned int refinements = tria.n_global_levels() - 1;

  // create a sparse graph ...
  std::vector<Graph> graphs(refinements + 1);
  // ... for each refinement level
  for(unsigned int ref = 0; ref <= refinements; ref++)
  {
    std::vector<int> eind, eptr;
    eptr.push_back(0);

    // Step 2: extract mesh from triangulation
    std::map<unsigned int, unsigned int> periodic_map;
    for(auto cell : tria.cell_iterators_on_level(ref))
    {
      for(unsigned int i = 0; i < GeometryInfo<dim>::faces_per_cell; i++)
      {
        if(cell->has_periodic_neighbor(i))
        {
          auto face_t = cell->face(i);
          auto face_n = cell->periodic_neighbor(i)->face(cell->periodic_neighbor_face_no(i));
          for(unsigned int j = 0; j < GeometryInfo<dim>::vertices_per_face; j++)
          {
            auto         v_t  = face_t->vertex_index(j);
            auto         v_n  = face_n->vertex_index(j);
            unsigned int temp = std::min(v_t, v_n);
            {
              auto it = periodic_map.find(v_t);
              if(it != periodic_map.end())
                temp = std::min(temp, it->second);
            }
            {
              auto it = periodic_map.find(v_n);
              if(it != periodic_map.end())
                temp = std::min(temp, it->second);
            }
            periodic_map[v_t] = temp;
            periodic_map[v_n] = temp;
          }
        }
      }
    }

    for(auto p : periodic_map)
    {
      if(p.first == p.second)
        continue;
      auto pp = periodic_map.find(p.second);
      if(pp->first == pp->second)
        continue;
      AssertThrow(false, ExcMessage("Map has to be compressed!"));
    }

    std::map<unsigned int, unsigned int> temp_map;
    for(auto cell : tria.cell_iterators_on_level(ref))
      for(unsigned int i = 0; i < GeometryInfo<dim>::vertices_per_cell; i++)
      {
        auto pp = periodic_map.find(cell->vertex_index(i));
        if(pp != periodic_map.end() && (pp->first != pp->second))
          continue;
        temp_map[cell->vertex_index(i)] = -1;
      }

    int c = 0;
    for(auto & m : temp_map)
      m.second = c++;

    for(auto cell : tria.cell_iterators_on_level(ref))
    {
      for(unsigned int i = 0; i < GeometryInfo<dim>::vertices_per_cell; i++)
      {
        if(temp_map.find(cell->vertex_index(i)) != temp_map.end())
          eind.push_back(temp_map[cell->vertex_index(i)]);
        else
        {
          eind.push_back(temp_map[periodic_map[cell->vertex_index(i)]]);
        }
      }
      eptr.push_back(GeometryInfo<dim>::vertices_per_cell + eptr.back());
    }

    auto & graph_vertex = graphs[ref];

    // select a partitioner
    std::shared_ptr<PartitioningAlgorithm> partitioner;

    if(additional_data.partition_type == PartitioningType::metis)
      partitioner.reset(new MetisPartitioningAlgorithm());
    else
      AssertThrow(false, ExcMessage("No partitioner has been selected."));

    // Step 3: create dual graph with connectivity on vertices
    partitioner->mesh_to_dual(eptr, eind, 1, graph_vertex);

    if(ref == refinements)
    {
      // Step 4a: perform partitioning on finest level
      // create dual graph with connectivity on faces
      Graph graph_face;
      partitioner->mesh_to_dual(eptr, eind, GeometryInfo<dim>::vertices_per_face, graph_face);
      // perform pre-partitioning such that groups are kept together
      // partitioner->partition(graph_face, size_node, false);
      // partitioner->partition(graph_face, size_groups, true);
      // use pre-partitioning result as weight for actual
      // partitioning
      partitioner->partition(graph_face, size_all, false);

      auto compress = [](const Graph & graph_in) {
        Graph graph_out;

        auto n_parts = *std::max_element(graph_in.parts.begin(), graph_in.parts.end()) + 1;
        std::vector<std::map<unsigned int, unsigned int>> temp(n_parts);

        for(unsigned int i = 0; i < graph_in.xadj.size() - 1; i++)
        {
          for(int j = graph_in.xadj[i]; j < graph_in.xadj[i + 1]; j++)
            temp[graph_in.parts[i]][graph_in.parts[graph_in.adjncy[j]]] = 0;
        }

        for(unsigned int i = 0; i < graph_in.xadj.size() - 1; i++)
        {
          for(int j = graph_in.xadj[i]; j < graph_in.xadj[i + 1]; j++)
            if(graph_in.weights.size() == 0)
              temp[graph_in.parts[i]][graph_in.parts[graph_in.adjncy[j]]]++;
            else
              temp[graph_in.parts[i]][graph_in.parts[graph_in.adjncy[j]]] += graph_in.weights[j];
        }

        graph_out.elements = n_parts;

        graph_out.xadj.push_back(0);

        for(auto & t : temp)
        {
          graph_out.xadj.push_back(graph_out.xadj.back() + t.size());
          for(auto & tt : t)
          {
            graph_out.adjncy.push_back(tt.first);
            graph_out.weights.push_back(tt.second);
          }
        }

        return graph_out;
      };

      auto g1 = compress(graph_face);
      partitioner->partition(g1, size_groups, false);
      // std::cout << "G-";
      // g1.print(std::cout);

      auto g2 = compress(g1);
      partitioner->partition(g2, size_node, false);
      // std::cout << "N-";
      // g2.print(std::cout);


      // re-numerate ranks
      std::map<std::pair<unsigned int, unsigned int>, std::set<unsigned int>> sets;
      for(unsigned int i = 0; i < g1.parts.size(); i++)
        sets[{g2.parts[g1.parts[i]], g1.parts[i]}] = std::set<unsigned int>();

      for(unsigned int i = 0; i < g1.parts.size(); i++)
        sets[{g2.parts[g1.parts[i]], g1.parts[i]}].insert(i);

      unsigned int     k = 0;
      std::vector<int> re_order(g1.parts.size());
      for(auto & set : sets)
        for(auto s : set.second)
          re_order[s] = k++;

      // printf("OO:  ");
      // for (auto i : re_order)
      //  printf("%3d  ", i);
      // printf("\n");

      for(unsigned int i = 0; i < graph_face.parts.size(); i++)
        graph_face.parts[i] = re_order[graph_face.parts[i]];

      // save partitioning
      graph_vertex.parts = graph_face.parts;
    }
    else if(ref == 0 && do_repartition)
    {
      // Step 4b: perform partitioning on coarest level
      // create dual graph with connectivity on faces
      Graph graph_face;
      partitioner->mesh_to_dual(eptr, eind, GeometryInfo<dim>::vertices_per_face, graph_face);
      // perform partitioning on this dual graph
      partitioner->partition(graph_face, size_coarse);
      // save partitioning
      graph_vertex.parts = graph_face.parts;
    }
  }
}

} // namespace GridTools
} // namespace dealii

#include "grid_tools.cpp"

#endif
