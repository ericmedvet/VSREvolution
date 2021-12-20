package it.units.erallab.builder.phenotype;

import it.units.erallab.builder.PrototypedFunctionBuilder;
import it.units.erallab.hmsrobots.core.controllers.MultiLayerPerceptron;
import it.units.erallab.hmsrobots.core.controllers.TimedRealFunction;
import it.units.erallab.hmsrobots.core.controllers.snndiscr.QuantizedMultilayerSpikingNetwork;
import it.units.erallab.hmsrobots.core.controllers.snndiscr.QuantizedSpikingFunction;
import it.units.erallab.hmsrobots.core.controllers.snndiscr.converters.stv.QuantizedMovingAverageSpikeTrainToValueConverter;
import it.units.erallab.hmsrobots.core.controllers.snndiscr.onehot.OneHotMultilayerSpikingNetworkWithConverters;

import java.util.Collections;
import java.util.List;
import java.util.function.BiFunction;
import java.util.function.Function;

/**
 * @author eric
 */
public class OneHotMSNWithConverters implements PrototypedFunctionBuilder<List<Double>, TimedRealFunction> {

  private final double innerLayerRatio;
  private final int nOfInnerLayers;
  private final BiFunction<Integer, Integer, QuantizedSpikingFunction> neuronBuilder;
  private final int inputConverterBins;
  private final int outputConverterBins;

  public OneHotMSNWithConverters(double innerLayerRatio, int nOfInnerLayers,
                                 BiFunction<Integer, Integer, QuantizedSpikingFunction> neuronBuilder,
                                 int inputConverterBins, int outputConverterBins) {
    this.innerLayerRatio = innerLayerRatio;
    this.nOfInnerLayers = nOfInnerLayers;
    this.neuronBuilder = neuronBuilder;
    this.inputConverterBins = inputConverterBins;
    this.outputConverterBins = outputConverterBins;
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
      int nOfInputs = function.getInputDimension() * inputConverterBins;
      int nOfOutputs = function.getOutputDimension() * outputConverterBins;
      int[] innerNeurons = innerNeurons(nOfInputs, nOfOutputs);
      int nOfWeights = QuantizedMultilayerSpikingNetwork.countWeights(nOfInputs, innerNeurons, nOfOutputs);
      if (nOfWeights != values.size()) {
        throw new IllegalArgumentException(String.format(
            "Wrong number of values for weights: %d expected, %d found",
            nOfWeights,
            values.size()
        ));
      }
      QuantizedMultilayerSpikingNetwork quantizedMultilayerSpikingNetwork = new QuantizedMultilayerSpikingNetwork(nOfInputs,
          innerNeurons,
          nOfOutputs,
          values.stream().mapToDouble(d -> d).toArray(),
          neuronBuilder,
          new QuantizedMovingAverageSpikeTrainToValueConverter(50, 5)
      );
      return new OneHotMultilayerSpikingNetworkWithConverters<>(
          quantizedMultilayerSpikingNetwork,
          inputConverterBins,
          outputConverterBins
      );
    };
  }

  @Override
  public List<Double> exampleFor(TimedRealFunction function) {
    int nOfInputs = function.getInputDimension() * inputConverterBins;
    int nOfOutputs = function.getOutputDimension() * outputConverterBins;
    return Collections.nCopies(
        QuantizedMultilayerSpikingNetwork.countWeights(
            MultiLayerPerceptron.countNeurons(
                nOfInputs,
                innerNeurons(nOfInputs, nOfOutputs),
                nOfOutputs)
        ),
        0d
    );
  }

}
