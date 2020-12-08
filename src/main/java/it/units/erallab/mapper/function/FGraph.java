package it.units.erallab.mapper.function;

import it.units.erallab.RealFunction;
import it.units.erallab.mapper.PrototypedFunctionBuilder;
import it.units.malelab.jgea.representation.graph.Graph;
import it.units.malelab.jgea.representation.graph.Node;
import it.units.malelab.jgea.representation.graph.numeric.functiongraph.FunctionGraph;
import it.units.malelab.jgea.representation.graph.numeric.functiongraph.ShallowSparseFactory;

import java.util.Random;
import java.util.function.Function;

/**
 * @author eric
 * @created 2020/12/08
 * @project VSREvolution
 */
public class FGraph implements PrototypedFunctionBuilder<Graph<Node, Double>, RealFunction> {

  @Override
  public Function<Graph<Node, Double>, RealFunction> buildFor(RealFunction function) {
    return graph -> RealFunction.from(
        function.getNOfInputs(),
        function.getNOfOutputs(),
        FunctionGraph.builder().apply(graph)
    );
  }

  @Override
  public Graph<Node, Double> exampleFor(RealFunction function) {
    return new ShallowSparseFactory(0d, 0d, 1d, function.getNOfInputs(), function.getNOfOutputs()).build(new Random(0));
  }
}
