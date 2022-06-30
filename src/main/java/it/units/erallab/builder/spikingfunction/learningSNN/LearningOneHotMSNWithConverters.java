package it.units.erallab.builder.spikingfunction.learningSNN;

import it.units.erallab.builder.PrototypedFunctionBuilder;
import it.units.erallab.hmsrobots.core.controllers.MultiLayerPerceptron;
import it.units.erallab.hmsrobots.core.controllers.TimedRealFunction;
import it.units.erallab.hmsrobots.core.controllers.snn.learning.*;
import it.units.erallab.hmsrobots.core.controllers.snndiscr.QuantizedLearningMultilayerSpikingNetwork;
import it.units.erallab.hmsrobots.core.controllers.snndiscr.QuantizedMultilayerSpikingNetwork;
import it.units.erallab.hmsrobots.core.controllers.snndiscr.QuantizedSpikingFunction;
import it.units.erallab.hmsrobots.core.controllers.snndiscr.converters.stv.QuantizedMovingAverageSpikeTrainToValueConverter;
import it.units.erallab.hmsrobots.core.controllers.snndiscr.onehot.OneHotMultilayerSpikingNetworkWithConverters;
import it.units.malelab.jgea.core.util.Pair;
import it.units.malelab.jgea.representation.sequence.bit.BitString;

import java.util.Collections;
import java.util.List;
import java.util.function.BiFunction;
import java.util.function.Function;
import java.util.stream.IntStream;

/**
 * @author eric
 */
public class LearningOneHotMSNWithConverters implements PrototypedFunctionBuilder<Pair<BitString, List<Double>>, TimedRealFunction> {

  private final double innerLayerRatio;
  private final int nOfInnerLayers;
  private final BiFunction<Integer, Integer, QuantizedSpikingFunction> neuronBuilder;
  private final int inputConverterBins;
  private final int outputConverterBins;
  private final double[] symmetricParams;
  private final double[] asymmetricParams;

  public LearningOneHotMSNWithConverters(double innerLayerRatio, int nOfInnerLayers,
                                         BiFunction<Integer, Integer, QuantizedSpikingFunction> neuronBuilder,
                                         int inputConverterBins, int outputConverterBins,
                                         double[] symmetricParams, double[] asymmetricParams) {
    this.innerLayerRatio = innerLayerRatio;
    this.nOfInnerLayers = nOfInnerLayers;
    this.neuronBuilder = neuronBuilder;
    this.inputConverterBins = inputConverterBins;
    this.outputConverterBins = outputConverterBins;
    this.symmetricParams = symmetricParams;
    this.asymmetricParams = asymmetricParams;
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
      int nOfInputs = function.getInputDimension() * inputConverterBins;
      int nOfOutputs = function.getOutputDimension() * outputConverterBins;
      int[] innerNeurons = innerNeurons(nOfInputs, nOfOutputs);
      int nOfWeights = QuantizedMultilayerSpikingNetwork.countWeights(nOfInputs, innerNeurons, nOfOutputs);
      BitString rulesParameters = values.first();
      List<Double> weightsParameters = values.second();
      if (3 * nOfWeights != rulesParameters.size()) {
        throw new IllegalArgumentException(String.format(
            "Wrong number of rules parameters: %d expected, %d found",
            3 * nOfWeights,
            rulesParameters.size()
        ));
      }
      if (nOfWeights != weightsParameters.size()) {
        throw new IllegalArgumentException(String.format(
            "Wrong number of weights parameters: %d expected, %d found",
            nOfWeights,
            weightsParameters.size()
        ));
      }
      double[] weights = weightsParameters.stream().mapToDouble(d -> d).toArray();
      BitString activityParameters = rulesParameters.slice(0, nOfWeights);
      BitString symmetryParameters = rulesParameters.slice(nOfWeights, 2 * nOfWeights);
      BitString hebbianParameters = rulesParameters.slice(2 * nOfWeights, rulesParameters.size());
      STDPLearningRule[] learningRules = IntStream.range(0, nOfWeights).mapToObj(i -> createLearningRule(
          activityParameters.get(i),
          symmetryParameters.get(i),
          hebbianParameters.get(i)
      )).toArray(STDPLearningRule[]::new);
      QuantizedLearningMultilayerSpikingNetwork quantizedLearningMultilayerSpikingNetwork = new QuantizedLearningMultilayerSpikingNetwork(
          nOfInputs,
          innerNeurons,
          nOfOutputs,
          weights,
          learningRules,
          neuronBuilder,
          new QuantizedMovingAverageSpikeTrainToValueConverter(50, 5)
      );
      return new OneHotMultilayerSpikingNetworkWithConverters<>(
          quantizedLearningMultilayerSpikingNetwork,
          inputConverterBins,
          outputConverterBins
      );
    };
  }

  @Override
  public Pair<BitString, List<Double>> exampleFor(TimedRealFunction function) {
    int nOfInputs = function.getInputDimension() * inputConverterBins;
    int nOfOutputs = function.getOutputDimension() * outputConverterBins;
    int nOfWeights = QuantizedMultilayerSpikingNetwork.countWeights(
        MultiLayerPerceptron.countNeurons(
            nOfInputs,
            innerNeurons(nOfInputs, nOfOutputs),
            nOfOutputs));
    return Pair.of(
        new BitString(3 * nOfWeights),
        Collections.nCopies(nOfWeights, 0d)
    );
  }

  private STDPLearningRule createLearningRule(boolean active, boolean symmetric, boolean hebbian) {
    if (!active) {
      return new DegenerateLearningRule();
    }
    STDPLearningRule learningRule;
    if (symmetric) {
      learningRule = hebbian ? new SymmetricHebbianLearningRule() : new SymmetricAntiHebbianLearningRule();
      learningRule.setParams(symmetricParams);
    } else {
      learningRule = hebbian ? new AsymmetricHebbianLearningRule() : new AsymmetricAntiHebbianLearningRule();
      learningRule.setParams(asymmetricParams);
    }
    return learningRule;
  }

}
