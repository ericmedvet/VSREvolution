package it.units.erallab.mapper.phenotype;

import it.units.erallab.RealFunction;
import it.units.erallab.hmsrobots.core.controllers.MultiLayerPerceptron;
import it.units.erallab.mapper.PrototypedFunctionBuilder;

import java.util.Collections;
import java.util.List;
import java.util.function.Function;

/**
 * @author eric
 * @created 2020/12/08
 * @project VSREvolution
 */
public class MLP implements PrototypedFunctionBuilder<List<Double>, RealFunction> {

  private final double innerLayerRatio;
  private final int nOfInnerLayers;

  public MLP(double innerLayerRatio, int nOfInnerLayers) {
    this.innerLayerRatio = innerLayerRatio;
    this.nOfInnerLayers = nOfInnerLayers;
  }

  private int[] innerNeurons(int nOfInputs, int nOfOutputs) {
    int[] innerNeurons = new int[nOfInnerLayers];
    int previousSize = nOfInputs;
    for (int i = 0; i < innerNeurons.length; i++) {
      innerNeurons[i] = (int) Math.max(Math.round((double) previousSize * innerLayerRatio), 2);
      previousSize = innerNeurons[i];
    }
    return innerNeurons;
  }

  @Override
  public Function<List<Double>, RealFunction> buildFor(RealFunction function) {
    return values -> RealFunction.from(
        function.getNOfInputs(),
        function.getNOfOutputs(),
        new MultiLayerPerceptron(
            MultiLayerPerceptron.ActivationFunction.TANH,
            function.getNOfInputs(),
            innerNeurons(function.getNOfInputs(), function.getNOfOutputs()),
            function.getNOfOutputs(),
            values.stream().mapToDouble(d -> d).toArray()
        ));
  }

  @Override
  public List<Double> exampleFor(RealFunction function) {
    return Collections.nCopies(
        MultiLayerPerceptron.countWeights(
            MultiLayerPerceptron.countNeurons(
                function.getNOfInputs(),
                innerNeurons(function.getNOfInputs(), function.getNOfOutputs()),
                function.getNOfOutputs())
        ),
        0d
    );
  }
}
