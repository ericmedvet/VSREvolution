package it.units.erallab.builder.spikingfunction;

import it.units.erallab.builder.NamedProvider;
import it.units.erallab.builder.PrototypedFunctionBuilder;
import it.units.erallab.builder.function.MLP;
import it.units.erallab.hmsrobots.core.controllers.MultiLayerPerceptron;
import it.units.erallab.hmsrobots.core.controllers.TimedRealFunction;
import it.units.erallab.hmsrobots.core.controllers.snndiscr.QuantizedMultilayerSpikingNetwork;
import it.units.erallab.hmsrobots.core.controllers.snndiscr.QuantizedSpikingFunction;
import it.units.erallab.hmsrobots.core.controllers.snndiscr.converters.stv.QuantizedMovingAverageSpikeTrainToValueConverter;
import it.units.erallab.hmsrobots.core.controllers.snndiscr.onehot.OneHotMultilayerSpikingNetworkWithConverters;
import it.units.erallab.hmsrobots.util.SerializationUtils;
import it.units.erallab.utils.SnnUtils;

import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.function.Function;

/**
 * @author eric
 */
public class OneHotMSNWithConverters implements NamedProvider<PrototypedFunctionBuilder<List<Double>, TimedRealFunction>> {

  @Override
  public PrototypedFunctionBuilder<List<Double>, TimedRealFunction> build(Map<String, String> params) {
    double innerLayerRatio = Double.parseDouble(params.getOrDefault("r", "0.65"));
    int nOfInnerLayers = Integer.parseInt(params.getOrDefault("nIL", "1"));
    QuantizedSpikingFunction spikingFunction = SnnUtils.buildQuantizedSpikingFunction(params.getOrDefault("m", ""));
    int inputConverterBins = Integer.parseInt(params.getOrDefault("nIB", "5"));
    int outputConverterBins = Integer.parseInt(params.getOrDefault("nOB", "5"));
    return new PrototypedFunctionBuilder<>() {
      @Override
      public Function<List<Double>, TimedRealFunction> buildFor(TimedRealFunction function) {
        return values -> {
          int nOfInputs = function.getInputDimension() * inputConverterBins;
          int nOfOutputs = function.getOutputDimension() * outputConverterBins;
          int[] innerNeurons = MLP.innerNeurons(nOfInputs, nOfOutputs, innerLayerRatio, nOfInnerLayers);
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
              (x, y) -> SerializationUtils.clone(spikingFunction),
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
                    MLP.innerNeurons(function.getInputDimension(), function.getOutputDimension(), innerLayerRatio, nOfInnerLayers),
                    nOfOutputs)
            ),
            0d
        );
      }

    };
  }


}
