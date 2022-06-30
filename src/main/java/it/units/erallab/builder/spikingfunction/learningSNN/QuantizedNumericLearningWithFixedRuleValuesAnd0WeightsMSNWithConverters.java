package it.units.erallab.builder.spikingfunction.learningSNN;

import it.units.erallab.builder.PrototypedFunctionBuilder;
import it.units.erallab.hmsrobots.core.controllers.MultiLayerPerceptron;
import it.units.erallab.hmsrobots.core.controllers.TimedRealFunction;
import it.units.erallab.hmsrobots.core.controllers.snn.learning.STDPLearningRule;
import it.units.erallab.hmsrobots.core.controllers.snndiscr.QuantizedLearningMultilayerSpikingNetwork;
import it.units.erallab.hmsrobots.core.controllers.snndiscr.QuantizedMultilayerSpikingNetwork;
import it.units.erallab.hmsrobots.core.controllers.snndiscr.QuantizedMultilayerSpikingNetworkWithConverters;
import it.units.erallab.hmsrobots.core.controllers.snndiscr.QuantizedSpikingFunction;
import it.units.erallab.hmsrobots.core.controllers.snndiscr.converters.stv.QuantizedSpikeTrainToValueConverter;
import it.units.erallab.hmsrobots.core.controllers.snndiscr.converters.vts.QuantizedValueToSpikeTrainConverter;

import java.util.Collections;
import java.util.List;
import java.util.function.BiFunction;
import java.util.function.Function;
import java.util.stream.IntStream;

public class QuantizedNumericLearningWithFixedRuleValuesAnd0WeightsMSNWithConverters implements PrototypedFunctionBuilder<List<Double>, TimedRealFunction> {

  private final double innerLayerRatio;
  private final int nOfInnerLayers;
  private final BiFunction<Integer, Integer, QuantizedSpikingFunction> neuronBuilder;
  private final QuantizedValueToSpikeTrainConverter valueToSpikeTrainConverter;
  private final QuantizedSpikeTrainToValueConverter spikeTrainToValueConverter;
  private final double[] defaultSymmetricParams;
  private final double[] defaultAsymmetricParams;

  public QuantizedNumericLearningWithFixedRuleValuesAnd0WeightsMSNWithConverters(double innerLayerRatio, int nOfInnerLayers, BiFunction<Integer, Integer, QuantizedSpikingFunction> neuronBuilder, QuantizedValueToSpikeTrainConverter valueToSpikeTrainConverter, QuantizedSpikeTrainToValueConverter spikeTrainToValueConverter, double[] defaultSymmetricParams, double[] defaultAsymmetricParams) {
    this.innerLayerRatio = innerLayerRatio;
    this.nOfInnerLayers = nOfInnerLayers;
    this.neuronBuilder = neuronBuilder;
    this.valueToSpikeTrainConverter = valueToSpikeTrainConverter;
    this.spikeTrainToValueConverter = spikeTrainToValueConverter;
    this.defaultSymmetricParams = defaultSymmetricParams;
    this.defaultAsymmetricParams = defaultAsymmetricParams;
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
      int nOfWeights = QuantizedMultilayerSpikingNetwork.countWeights(nOfInputs, innerNeurons, nOfOutputs);
      if (2 * nOfWeights != values.size()) {
        throw new IllegalArgumentException(String.format(
            "Wrong number of values for learning rules: %d expected, %d found",
            2 * nOfWeights,
            values.size()
        ));
      }
      double[] weights = new double[nOfWeights];
      double[][] rulesGenerator = IntStream.range(0, values.size() / 2).map(i -> 2 * i).mapToObj(i -> new double[]{values.get(i), values.get(i + 1)}).toArray(double[][]::new);
      STDPLearningRule[] flatLearningRules = STDPLearningRule.createLearningRules(rulesGenerator, defaultSymmetricParams, defaultAsymmetricParams);
      QuantizedLearningMultilayerSpikingNetwork quantizedLearningMultilayerSpikingNetwork = new QuantizedLearningMultilayerSpikingNetwork(
          nOfInputs, innerNeurons, nOfOutputs, weights, flatLearningRules, neuronBuilder, spikeTrainToValueConverter);
      return new QuantizedMultilayerSpikingNetworkWithConverters<>(
          quantizedLearningMultilayerSpikingNetwork,
          valueToSpikeTrainConverter,
          spikeTrainToValueConverter
      );
    };
  }

  @Override
  public List<Double> exampleFor(TimedRealFunction function) {
    return Collections.nCopies(
        2 * QuantizedMultilayerSpikingNetwork.countWeights(
            MultiLayerPerceptron.countNeurons(
                function.getInputDimension(),
                innerNeurons(function.getInputDimension(), function.getOutputDimension()),
                function.getOutputDimension())
        ),
        0d
    );
  }

}
