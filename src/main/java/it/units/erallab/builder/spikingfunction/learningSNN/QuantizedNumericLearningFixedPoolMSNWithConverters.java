package it.units.erallab.builder.spikingfunction.learningSNN;

import it.units.erallab.builder.PrototypedFunctionBuilder;
import it.units.erallab.hmsrobots.core.controllers.MultiLayerPerceptron;
import it.units.erallab.hmsrobots.core.controllers.TimedRealFunction;
import it.units.erallab.hmsrobots.core.controllers.snn.learning.*;
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
import java.util.stream.IntStream;

public class QuantizedNumericLearningFixedPoolMSNWithConverters implements PrototypedFunctionBuilder<List<Double>, TimedRealFunction> {

  private final double innerLayerRatio;
  private final int nOfInnerLayers;
  private final BiFunction<Integer, Integer, QuantizedSpikingFunction> neuronBuilder;
  private final QuantizedValueToSpikeTrainConverter valueToSpikeTrainConverter;
  private final QuantizedSpikeTrainToValueConverter spikeTrainToValueConverter;
  private final int nOfLearningRules;

  public QuantizedNumericLearningFixedPoolMSNWithConverters(int nRules, double innerLayerRatio, int nOfInnerLayers, BiFunction<Integer, Integer, QuantizedSpikingFunction> neuronBuilder, QuantizedValueToSpikeTrainConverter valueToSpikeTrainConverter, QuantizedSpikeTrainToValueConverter spikeTrainToValueConverter) {
    this.innerLayerRatio = innerLayerRatio;
    this.nOfInnerLayers = nOfInnerLayers;
    this.neuronBuilder = neuronBuilder;
    this.valueToSpikeTrainConverter = valueToSpikeTrainConverter;
    this.spikeTrainToValueConverter = spikeTrainToValueConverter;
    this.nOfLearningRules = 4 * nRules;
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
      if (4 * (nOfWeights + nOfLearningRules) != values.size()) {
        throw new IllegalArgumentException(String.format(
            "Wrong number of values for learning rules: %d expected, %d found",
            7 * nOfWeights,
            values.size()
        ));
      }
      double[] weights = values.subList(0, nOfWeights).stream().mapToDouble(d -> d).toArray();

      double[] learningRulesIndexesGenerator = values.subList(nOfWeights, 4 * nOfWeights).stream().mapToDouble(d -> d).toArray();
      double[] maxes = {IntStream.range(0, nOfWeights).filter(i -> learningRulesIndexesGenerator[3 * i] < 0 && learningRulesIndexesGenerator[3 * i + 1] < 0).mapToDouble(i -> learningRulesIndexesGenerator[3 * i + 2]).max().orElse(0),
          IntStream.range(0, nOfWeights).filter(i -> learningRulesIndexesGenerator[3 * i] < 0 && learningRulesIndexesGenerator[3 * i + 1] > 0).mapToDouble(i -> learningRulesIndexesGenerator[3 * i + 2]).max().orElse(0),
          IntStream.range(0, nOfWeights).filter(i -> learningRulesIndexesGenerator[3 * i] > 0 && learningRulesIndexesGenerator[3 * i + 1] < 0).mapToDouble(i -> learningRulesIndexesGenerator[3 * i + 2]).max().orElse(0),
          IntStream.range(0, nOfWeights).filter(i -> learningRulesIndexesGenerator[3 * i] > 0 && learningRulesIndexesGenerator[3 * i + 1] > 0).mapToDouble(i -> learningRulesIndexesGenerator[3 * i + 2]).max().orElse(0)};
      double[] mins = {IntStream.range(0, nOfWeights).filter(i -> learningRulesIndexesGenerator[3 * i] < 0 && learningRulesIndexesGenerator[3 * i + 1] < 0).mapToDouble(i -> learningRulesIndexesGenerator[3 * i + 2]).min().orElse(0),
          IntStream.range(0, nOfWeights).filter(i -> learningRulesIndexesGenerator[3 * i] < 0 && learningRulesIndexesGenerator[3 * i + 1] > 0).mapToDouble(i -> learningRulesIndexesGenerator[3 * i + 2]).min().orElse(0),
          IntStream.range(0, nOfWeights).filter(i -> learningRulesIndexesGenerator[3 * i] > 0 && learningRulesIndexesGenerator[3 * i + 1] < 0).mapToDouble(i -> learningRulesIndexesGenerator[3 * i + 2]).min().orElse(0),
          IntStream.range(0, nOfWeights).filter(i -> learningRulesIndexesGenerator[3 * i] > 0 && learningRulesIndexesGenerator[3 * i + 1] > 0).mapToDouble(i -> learningRulesIndexesGenerator[3 * i + 2]).min().orElse(0)};
      int[] learningRulesIndexes = new int[nOfWeights];
      for (int i = 0; i < nOfWeights; i++) {
        int ind = 0;
        if (learningRulesIndexesGenerator[3 * i] > 0) {
          learningRulesIndexes[i] += nOfLearningRules / 2;
          ind += 2;
        }
        if (learningRulesIndexesGenerator[3 * i + 1] > 0) {
          learningRulesIndexes[i] += nOfLearningRules / 4;
          ind += 1;
        }
        double step = 4d * (maxes[ind] - mins[ind]) / nOfLearningRules;
        int c = 0;
        for (double s = mins[ind] + step; s <= maxes[ind]; s += step) {
          if (learningRulesIndexesGenerator[3 * i + 2] < s) {
            learningRulesIndexes[i] += c;
            break;
          }
          c++;
        }
      }

      double[] learningRulesParameters = values.subList(4 * nOfWeights, values.size()).stream().mapToDouble(d -> d).toArray();
      STDPLearningRule[] fixedLearningRules = new STDPLearningRule[nOfLearningRules];
      IntStream.range(0, fixedLearningRules.length).forEach(i -> {
        double[] params = {learningRulesParameters[4 * i], learningRulesParameters[4 * i + 1], learningRulesParameters[4 * i + 2], learningRulesParameters[4 * i + 3]};
        if (i < fixedLearningRules.length / 2) {
          SymmetricSTPDLearningRule.scaleParameters(params);
          if (i < fixedLearningRules.length / 4) {
            fixedLearningRules[i] = new SymmetricHebbianLearningRule();
          } else {
            fixedLearningRules[i] = new SymmetricAntiHebbianLearningRule();
          }
        } else {
          AsymmetricSTDPLearningRule.scaleParameters(params);
          if (i < 3 * fixedLearningRules.length / 4) {
            fixedLearningRules[i] = new AsymmetricHebbianLearningRule();
          } else {
            fixedLearningRules[i] = new AsymmetricAntiHebbianLearningRule();
          }
        }
        fixedLearningRules[i].setParams(params);
      });

      STDPLearningRule[] learningRules = Arrays.stream(learningRulesIndexes).mapToObj(i -> fixedLearningRules[i]).toArray(STDPLearningRule[]::new);

      QuantizedLearningMultilayerSpikingNetwork quantizedFixedPoolLearningMultilayerSpikingNetwork =
          new QuantizedLearningMultilayerSpikingNetwork(nOfInputs, innerNeurons, nOfOutputs, weights, learningRules, neuronBuilder, spikeTrainToValueConverter);
      return new QuantizedMultilayerSpikingNetworkWithConverters<>(
          quantizedFixedPoolLearningMultilayerSpikingNetwork,
          valueToSpikeTrainConverter,
          spikeTrainToValueConverter
      );
    };
  }

  @Override
  public List<Double> exampleFor(TimedRealFunction function) {
    return Collections.nCopies(
        4 * (QuantizedMultilayerSpikingNetwork.countWeights(
            MultiLayerPerceptron.countNeurons(
                function.getInputDimension(),
                innerNeurons(function.getInputDimension(), function.getOutputDimension()),
                function.getOutputDimension())
        ) + nOfLearningRules),
        0d
    );
  }

}
