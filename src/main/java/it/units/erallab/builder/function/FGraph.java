package it.units.erallab.builder.function;

import it.units.erallab.builder.NamedProvider;
import it.units.erallab.builder.PrototypedFunctionBuilder;
import it.units.erallab.hmsrobots.core.controllers.RealFunction;
import it.units.erallab.hmsrobots.util.SerializableFunction;
import it.units.malelab.jgea.representation.graph.Graph;
import it.units.malelab.jgea.representation.graph.Node;
import it.units.malelab.jgea.representation.graph.numeric.functiongraph.FunctionGraph;
import it.units.malelab.jgea.representation.graph.numeric.functiongraph.ShallowSparseFactory;

import java.util.Map;
import java.util.Random;
import java.util.function.Function;

/**
 * @author eric
 */
public class FGraph implements NamedProvider<PrototypedFunctionBuilder<Graph<Node, Double>, RealFunction>> {

  @Override
  public PrototypedFunctionBuilder<Graph<Node, Double>, RealFunction> build(Map<String, String> params) {
    return new PrototypedFunctionBuilder<>() {
      @Override
      public Function<Graph<Node, Double>, RealFunction> buildFor(RealFunction function) {
        return graph -> {
          FunctionGraph functionGraph = FunctionGraph.builder().apply(graph);
          return RealFunction.build(
              (SerializableFunction<double[], double[]>) functionGraph::apply,
              function.getInputDimension(),
              function.getOutputDimension()
          );
        };
      }

      @Override
      public Graph<Node, Double> exampleFor(RealFunction function) {
        return new ShallowSparseFactory(0d, 0d, 1d, function.getInputDimension(), function.getOutputDimension()).build(
            new Random(0));
      }
    };
  }

}
