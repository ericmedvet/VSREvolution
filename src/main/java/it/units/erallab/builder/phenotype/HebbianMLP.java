package it.units.erallab.builder.phenotype;

import it.units.erallab.builder.PrototypedFunctionBuilder;
import it.units.erallab.hmsrobots.core.controllers.HebbianMultiLayerPerceptron;
import it.units.erallab.hmsrobots.core.controllers.MultiLayerPerceptron;
import it.units.erallab.hmsrobots.core.controllers.TimedRealFunction;

import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.function.Function;

public class HebbianMLP implements PrototypedFunctionBuilder<List<Double>, TimedRealFunction> {

  private final double innerLayerRatio;
  private final int nOfInnerLayers;
  private final MultiLayerPerceptron.ActivationFunction activationFunction;
  private final double eta;
  private final Random random;
  private final boolean weightsNormalization;

  public HebbianMLP(double innerLayerRatio, int nOfInnerLayers, MultiLayerPerceptron.ActivationFunction activationFunction, double eta, Random random, boolean weightsNormalization) {
    this.innerLayerRatio = innerLayerRatio;
    this.nOfInnerLayers = nOfInnerLayers;
    this.activationFunction = activationFunction;
    this.eta = eta;
    this.random = random;
    this.weightsNormalization = weightsNormalization;
  }

  private int[] innerNeurons(int nOfInputs, int nOfOutputs) {
    int[] innerNeurons = new int[nOfInnerLayers];
    int centerSize = (int) Math.max(2, Math.round(nOfInputs * innerLayerRatio));
    if (nOfInnerLayers > 1) {
      for (int i = 0; i < nOfInnerLayers / 2; i++) {
        innerNeurons[i] = nOfInputs + (centerSize - nOfInputs) / (nOfInnerLayers / 2 + 1) * (i + 1);
      }
      for (int i = nOfInnerLayers / 2; i < nOfInnerLayers; i++) {
        innerNeurons[i] = centerSize + (nOfOutputs - centerSize) / (nOfInnerLayers / 2 + 1) * (i - nOfInnerLayers / 2);
      }
    } else if (nOfInnerLayers > 0) {
      innerNeurons[0] = centerSize;
    }
    return innerNeurons;
  }

  @Override
  public Function<List<Double>, TimedRealFunction> buildFor(TimedRealFunction function) {
    return values -> {
      int nOfInputs = function.getInputDimension();
      int nOfOutputs = function.getOutputDimension();
      int[] innerNeurons = innerNeurons(nOfInputs, nOfOutputs);
      int nOfHebbianCoefficients = HebbianMultiLayerPerceptron.countHebbianCoefficients(nOfInputs, innerNeurons, nOfOutputs);
      if (nOfHebbianCoefficients != values.size()) {
        throw new IllegalArgumentException(String.format(
            "Wrong number of values for Hebbian coefficient: %d expected, %d found",
            nOfHebbianCoefficients,
            values.size()
        ));
      }
      return new HebbianMultiLayerPerceptron(
          activationFunction,
          nOfInputs,
          innerNeurons,
          nOfOutputs,
          values.stream().mapToDouble(d -> d).toArray(),
          eta,
          random,
          weightsNormalization);
    };
  }

  @Override
  public List<Double> exampleFor(TimedRealFunction function) {
    return Collections.nCopies(
        HebbianMultiLayerPerceptron.countHebbianCoefficients(
            function.getInputDimension(),
            innerNeurons(function.getInputDimension(), function.getOutputDimension()),
            function.getOutputDimension()),
        0d
    );
  }

}
