package it.units.erallab.builder.spikingfunction.learningSNN;

import it.units.erallab.builder.PrototypedFunctionBuilder;
import it.units.erallab.hmsrobots.core.controllers.MultiLayerPerceptron;
import it.units.erallab.hmsrobots.core.controllers.TimedRealFunction;
import it.units.erallab.hmsrobots.core.controllers.snn.learning.STDPLearningRule;
import it.units.erallab.hmsrobots.core.controllers.snndiscr.*;
import it.units.erallab.hmsrobots.core.controllers.snndiscr.converters.stv.QuantizedSpikeTrainToValueConverter;
import it.units.erallab.hmsrobots.core.controllers.snndiscr.converters.vts.QuantizedValueToSpikeTrainConverter;

import java.util.Collections;
import java.util.List;
import java.util.function.BiFunction;
import java.util.function.Function;

public class QuantizedNumericLearningMSNWithConverters implements PrototypedFunctionBuilder<List<Double>, TimedRealFunction> {

  private final double innerLayerRatio;
  private final int nOfInnerLayers;
  private final BiFunction<Integer, Integer, QuantizedSpikingFunction> neuronBuilder;
  private final QuantizedValueToSpikeTrainConverter valueToSpikeTrainConverter;
  private final QuantizedSpikeTrainToValueConverter spikeTrainToValueConverter;

  public QuantizedNumericLearningMSNWithConverters(double innerLayerRatio, int nOfInnerLayers, BiFunction<Integer, Integer, QuantizedSpikingFunction> neuronBuilder, QuantizedValueToSpikeTrainConverter valueToSpikeTrainConverter, QuantizedSpikeTrainToValueConverter spikeTrainToValueConverter) {
    this.innerLayerRatio = innerLayerRatio;
    this.nOfInnerLayers = nOfInnerLayers;
    this.neuronBuilder = neuronBuilder;
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
      if (7 * nOfWeights != values.size()) {
        throw new IllegalArgumentException(String.format(
            "Wrong number of values for learning rules: %d expected, %d found",
            7 * nOfWeights,
            values.size()
        ));
      }
      double[] weights = values.subList(0, values.size() / 7).stream().mapToDouble(d -> d).toArray();
      List<Double> flatRulesGenerator = values.subList(values.size() / 7, values.size());
      double[][] rulesGenerator = new double[flatRulesGenerator.size() / 6][6];
      int j = 0;
      for (int i = 0; i < flatRulesGenerator.size(); i++) {
        int pos = i % 6;
        rulesGenerator[j][pos] = flatRulesGenerator.get(i);
        if (pos == 5) {
          j++;
        }
      }
      STDPLearningRule[] flatLearningRules = STDPLearningRule.createLearningRules(rulesGenerator);
      QuantizedLearningMultilayerSpikingNetwork quantizedLearningMultilayerSpikingNetwork = new QuantizedLearningMultilayerSpikingNetwork(
          nOfInputs, innerNeurons, nOfOutputs,weights, flatLearningRules, neuronBuilder, spikeTrainToValueConverter);
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
        7 * QuantizedMultilayerSpikingNetwork.countWeights(
            MultiLayerPerceptron.countNeurons(
                function.getInputDimension(),
                innerNeurons(function.getInputDimension(), function.getOutputDimension()),
                function.getOutputDimension())
        ),
        0d
    );
  }

}
