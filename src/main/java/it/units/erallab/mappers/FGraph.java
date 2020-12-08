package it.units.erallab.mappers;

import it.units.erallab.RealFunction;
import it.units.malelab.jgea.representation.graph.Graph;
import it.units.malelab.jgea.representation.graph.Node;
import it.units.malelab.jgea.representation.graph.numeric.functiongraph.FunctionGraph;
import it.units.malelab.jgea.representation.graph.numeric.functiongraph.ShallowSparseFactory;

import java.util.Random;

/**
 * @author eric
 * @created 2020/12/08
 * @project VSREvolution
 */
public class FGraph implements ReversableFunction<Graph<Node, Double>, RealFunction> {
  @Override
  public Graph<Node, Double> example(RealFunction function) {
    return new ShallowSparseFactory(0d, 0d, 1d, function.getInputDim(), function.getOutputDim()).build(new Random(0));
  }

  @Override
  public RealFunction apply(Graph<Node, Double> graph) {
    return RealFunction.from(
        0, //TODO fix
        0, //TODO fix
        FunctionGraph.builder().apply(graph)
    );
  }
}
