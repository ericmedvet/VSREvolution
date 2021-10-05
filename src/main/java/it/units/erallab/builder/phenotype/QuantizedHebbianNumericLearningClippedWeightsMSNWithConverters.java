package it.units.erallab.builder.phenotype;

import it.units.erallab.builder.PrototypedFunctionBuilder;
import it.units.erallab.hmsrobots.core.controllers.MultiLayerPerceptron;
import it.units.erallab.hmsrobots.core.controllers.TimedRealFunction;
import it.units.erallab.hmsrobots.core.controllers.snn.learning.AsymmetricHebbianLearningRule;
import it.units.erallab.hmsrobots.core.controllers.snn.learning.AsymmetricSTDPLearningRule;
import it.units.erallab.hmsrobots.core.controllers.snn.learning.STDPLearningRule;
import it.units.erallab.hmsrobots.core.controllers.snndiscr.QuantizedLearningMultilayerSpikingNetwork;
import it.units.erallab.hmsrobots.core.controllers.snndiscr.QuantizedMultilayerSpikingNetwork;
import it.units.erallab.hmsrobots.core.controllers.snndiscr.QuantizedMultilayerSpikingNetworkWithConverters;
import it.units.erallab.hmsrobots.core.controllers.snndiscr.QuantizedSpikingFunction;
import it.units.erallab.hmsrobots.core.controllers.snndiscr.converters.stv.QuantizedSpikeTrainToValueConverter;
import it.units.erallab.hmsrobots.core.controllers.snndiscr.converters.vts.QuantizedValueToSpikeTrainConverter;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.function.BiFunction;
import java.util.function.Function;

public class QuantizedHebbianNumericLearningClippedWeightsMSNWithConverters implements PrototypedFunctionBuilder<List<Double>, TimedRealFunction> {

  private final double innerLayerRatio;
  private final int nOfInnerLayers;
  private final BiFunction<Integer, Integer, QuantizedSpikingFunction> neuronBuilder;
  private final double weightsMaxValue;
  private final QuantizedValueToSpikeTrainConverter valueToSpikeTrainConverter;
  private final QuantizedSpikeTrainToValueConverter spikeTrainToValueConverter;

  public QuantizedHebbianNumericLearningClippedWeightsMSNWithConverters(double innerLayerRatio, int nOfInnerLayers, BiFunction<Integer, Integer, QuantizedSpikingFunction> neuronBuilder, double weightsMaxValue, QuantizedValueToSpikeTrainConverter valueToSpikeTrainConverter, QuantizedSpikeTrainToValueConverter spikeTrainToValueConverter) {
    this.innerLayerRatio = innerLayerRatio;
    this.nOfInnerLayers = nOfInnerLayers;
    this.neuronBuilder = neuronBuilder;
    this.weightsMaxValue = weightsMaxValue;
    this.valueToSpikeTrainConverter = valueToSpikeTrainConverter;
    this.spikeTrainToValueConverter = spikeTrainToValueConverter;
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
      if (5 * nOfWeights != values.size()) {
        throw new IllegalArgumentException(String.format(
            "Wrong number of values for learning rules: %d expected, %d found",
            5 * nOfWeights,
            values.size()
        ));
      }
      double[] weights = values.subList(0, nOfWeights).stream().mapToDouble(d -> d).toArray();
      values = values.subList(nOfWeights, values.size());
      double[][] rulesGenerator = new double[values.size() / 4][4];
      int j = 0;
      for (int i = 0; i < values.size(); i++) {
        int pos = i % 4;
        rulesGenerator[j][pos] = values.get(i);
        if (pos == 3) {
          j++;
        }
      }
      STDPLearningRule[] learningRules = Arrays.stream(rulesGenerator).map(params -> {
        AsymmetricHebbianLearningRule rule = new AsymmetricHebbianLearningRule();
        rule.setParams(AsymmetricSTDPLearningRule.scaleParameters(params));
        return rule;
      }).toArray(AsymmetricHebbianLearningRule[]::new);
      QuantizedLearningMultilayerSpikingNetwork quantizedLearningMultilayerSpikingNetwork = new QuantizedLearningMultilayerSpikingNetwork(
          nOfInputs, innerNeurons, nOfOutputs, weights, learningRules, neuronBuilder, spikeTrainToValueConverter);
      quantizedLearningMultilayerSpikingNetwork.enableWeightsClipping(weightsMaxValue);
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
        5 * QuantizedMultilayerSpikingNetwork.countWeights(
            MultiLayerPerceptron.countNeurons(
                function.getInputDimension(),
                innerNeurons(function.getInputDimension(), function.getOutputDimension()),
                function.getOutputDimension())
        ),
        0d
    );
  }

}
