package it.units.erallab.builder.phenotype;

import it.units.erallab.RealFunction;
import it.units.erallab.builder.PrototypedFunctionBuilder;
import it.units.erallab.hmsrobots.core.controllers.MultiLayerPerceptron;

import java.util.Collections;
import java.util.List;
import java.util.function.Function;

/**
 * @author eric
 */
public class MLP implements PrototypedFunctionBuilder<List<Double>, RealFunction> {

  private final double innerLayerRatio;
  private final int nOfInnerLayers;
  private final MultiLayerPerceptron.ActivationFunction activationFunction;

  public MLP(double innerLayerRatio, int nOfInnerLayers) {
    this(innerLayerRatio, nOfInnerLayers, MultiLayerPerceptron.ActivationFunction.TANH);
  }

  public MLP(double innerLayerRatio, int nOfInnerLayers, MultiLayerPerceptron.ActivationFunction activationFunction) {
    this.innerLayerRatio = innerLayerRatio;
    this.nOfInnerLayers = nOfInnerLayers;
    this.activationFunction = activationFunction;
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
    return values -> {
      int nOfInputs = function.getNOfInputs();
      int nOfOutputs = function.getNOfOutputs();
      int[] innerNeurons = innerNeurons(nOfInputs, nOfOutputs);
      int nOfWeights = MultiLayerPerceptron.countWeights(nOfInputs, innerNeurons, nOfOutputs);
      if (nOfWeights != values.size()) {
        throw new IllegalArgumentException(String.format(
            "Wrong number of values for weights: %d expected, %d found",
            nOfWeights,
            values.size()
        ));
      }
      return RealFunction.from(
          nOfInputs,
          nOfOutputs,
          new MultiLayerPerceptron(
              activationFunction,
              nOfInputs,
              innerNeurons,
              nOfOutputs,
              values.stream().mapToDouble(d -> d).toArray()
          ));
    };
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
