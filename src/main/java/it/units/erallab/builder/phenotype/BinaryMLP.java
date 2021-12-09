package it.units.erallab.builder.phenotype;

import it.units.erallab.builder.PrototypedFunctionBuilder;
import it.units.erallab.hmsrobots.core.controllers.MultiLayerPerceptron;
import it.units.erallab.hmsrobots.core.controllers.TimedRealFunction;
import it.units.malelab.jgea.representation.sequence.bit.BitString;

import java.util.function.Function;

/**
 * @author eric
 */
public class BinaryMLP implements PrototypedFunctionBuilder<BitString, TimedRealFunction> {

  private final double innerLayerRatio;
  private final int nOfInnerLayers;
  private final MultiLayerPerceptron.ActivationFunction activationFunction;
  private final double weight;

  public BinaryMLP(double innerLayerRatio, int nOfInnerLayers, double weight) {
    this(innerLayerRatio, nOfInnerLayers, MultiLayerPerceptron.ActivationFunction.TANH, weight);
  }

  public BinaryMLP(double innerLayerRatio, int nOfInnerLayers, MultiLayerPerceptron.ActivationFunction activationFunction, double weight) {
    this.innerLayerRatio = innerLayerRatio;
    this.nOfInnerLayers = nOfInnerLayers;
    this.activationFunction = activationFunction;
    this.weight = weight;
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
  public Function<BitString, TimedRealFunction> buildFor(TimedRealFunction function) {
    return values -> {
      int nOfInputs = function.getInputDimension();
      int nOfOutputs = function.getOutputDimension();
      int[] innerNeurons = innerNeurons(nOfInputs, nOfOutputs);
      int nOfWeights = MultiLayerPerceptron.countWeights(nOfInputs, innerNeurons, nOfOutputs);
      if (nOfWeights != values.size()) {
        throw new IllegalArgumentException(String.format(
            "Wrong number of values for weights: %d expected, %d found",
            nOfWeights,
            values.size()
        ));
      }
      return new MultiLayerPerceptron(
          activationFunction,
          nOfInputs,
          innerNeurons,
          nOfOutputs,
          values.stream().mapToDouble(b -> b ? weight : -weight).toArray()
      );
    };
  }

  @Override
  public BitString exampleFor(TimedRealFunction function) {
    return new BitString(
        MultiLayerPerceptron.countWeights(
            MultiLayerPerceptron.countNeurons(
                function.getInputDimension(),
                innerNeurons(function.getInputDimension(), function.getOutputDimension()),
                function.getOutputDimension())
        )
    );
  }

}
