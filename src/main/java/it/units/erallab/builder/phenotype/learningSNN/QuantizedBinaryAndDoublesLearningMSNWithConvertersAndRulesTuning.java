package it.units.erallab.builder.phenotype.learningSNN;

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
import it.units.malelab.jgea.core.util.Pair;
import it.units.malelab.jgea.representation.sequence.bit.BitString;

import java.util.Collections;
import java.util.List;
import java.util.function.BiFunction;
import java.util.function.Function;
import java.util.stream.IntStream;

public class QuantizedBinaryAndDoublesLearningMSNWithConvertersAndRulesTuning implements PrototypedFunctionBuilder<Pair<BitString, List<Double>>, TimedRealFunction> {

  private final double innerLayerRatio;
  private final int nOfInnerLayers;
  private final BiFunction<Integer, Integer, QuantizedSpikingFunction> neuronBuilder;
  private final QuantizedValueToSpikeTrainConverter valueToSpikeTrainConverter;
  private final QuantizedSpikeTrainToValueConverter spikeTrainToValueConverter;

  public QuantizedBinaryAndDoublesLearningMSNWithConvertersAndRulesTuning(double innerLayerRatio, int nOfInnerLayers, BiFunction<Integer, Integer, QuantizedSpikingFunction> neuronBuilder, QuantizedValueToSpikeTrainConverter valueToSpikeTrainConverter, QuantizedSpikeTrainToValueConverter spikeTrainToValueConverter) {
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
  public Function<Pair<BitString, List<Double>>, TimedRealFunction> buildFor(TimedRealFunction function) {
    return values -> {
      int nOfInputs = function.getInputDimension();
      int nOfOutputs = function.getOutputDimension();
      int[] innerNeurons = innerNeurons(nOfInputs, nOfOutputs);
      int nOfWeights = QuantizedMultilayerSpikingNetwork.countWeights(nOfInputs, innerNeurons, nOfOutputs);
      BitString rulesParameters = values.first();
      List<Double> numericParameters = values.second();
      if (3 * nOfWeights != rulesParameters.size()) {
        throw new IllegalArgumentException(String.format(
            "Wrong number of rules bit parameters: %d expected, %d found",
            3 * nOfWeights,
            rulesParameters.size()
        ));
      }
      if (5 * nOfWeights != numericParameters.size()) {
        throw new IllegalArgumentException(String.format(
            "Wrong number of rules and weights numeric parameters: %d expected, %d found",
            nOfWeights,
            numericParameters.size()
        ));
      }
      BitString activityParameters = rulesParameters.slice(0, nOfWeights);
      BitString symmetryParameters = rulesParameters.slice(nOfWeights, 2 * nOfWeights);
      BitString hebbianParameters = rulesParameters.slice(2 * nOfWeights, rulesParameters.size());

      double[] weights = new double[nOfWeights];
      double[][] rulesNumericParameters = new double[nOfWeights][4];
      IntStream.range(0, nOfWeights).forEach(i -> {
        weights[i] = numericParameters.get(5 * i);
        for (int j = 0; j < 4; j++) {
          rulesNumericParameters[i][j] = numericParameters.get(5 * i + 1 + j);
        }
      });
      STDPLearningRule[] learningRules = IntStream.range(0, nOfWeights).mapToObj(i -> createLearningRule(
          activityParameters.get(i),
          symmetryParameters.get(i),
          hebbianParameters.get(i),
          rulesNumericParameters[i]
      )).toArray(STDPLearningRule[]::new);
      QuantizedLearningMultilayerSpikingNetwork quantizedLearningMultilayerSpikingNetwork = new QuantizedLearningMultilayerSpikingNetwork(
          nOfInputs, innerNeurons, nOfOutputs, weights, learningRules, neuronBuilder, spikeTrainToValueConverter);
      return new QuantizedMultilayerSpikingNetworkWithConverters<>(
          quantizedLearningMultilayerSpikingNetwork,
          valueToSpikeTrainConverter,
          spikeTrainToValueConverter
      );
    };
  }

  @Override
  public Pair<BitString, List<Double>> exampleFor(TimedRealFunction function) {
    int nOfWeights = QuantizedMultilayerSpikingNetwork.countWeights(
        MultiLayerPerceptron.countNeurons(
            function.getInputDimension(),
            innerNeurons(function.getInputDimension(), function.getOutputDimension()),
            function.getOutputDimension()));
    return Pair.of(
        new BitString(3 * nOfWeights),
        Collections.nCopies(5 * nOfWeights, 0d)
    );
  }

  private STDPLearningRule createLearningRule(boolean active, boolean symmetric, boolean hebbian, double[] parameters) {
    if (!active) {
      return new DegenerateLearningRule();
    }
    STDPLearningRule learningRule;
    if (symmetric) {
      learningRule = hebbian ? new SymmetricHebbianLearningRule() : new SymmetricAntiHebbianLearningRule();
      learningRule.setParams(SymmetricSTPDLearningRule.scaleParameters(parameters));
    } else {
      learningRule = hebbian ? new AsymmetricHebbianLearningRule() : new AsymmetricAntiHebbianLearningRule();
      learningRule.setParams(AsymmetricSTDPLearningRule.scaleParameters(parameters));
    }
    return learningRule;
  }

}
