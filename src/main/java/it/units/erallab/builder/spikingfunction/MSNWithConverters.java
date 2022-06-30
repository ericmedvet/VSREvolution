package it.units.erallab.builder.spikingfunction;

import it.units.erallab.builder.NamedProvider;
import it.units.erallab.builder.PrototypedFunctionBuilder;
import it.units.erallab.builder.function.MLP;
import it.units.erallab.hmsrobots.core.controllers.MultiLayerPerceptron;
import it.units.erallab.hmsrobots.core.controllers.TimedRealFunction;
import it.units.erallab.hmsrobots.core.controllers.snn.MultilayerSpikingNetwork;
import it.units.erallab.hmsrobots.core.controllers.snn.MultilayerSpikingNetworkWithConverters;
import it.units.erallab.hmsrobots.core.controllers.snn.SpikingFunction;
import it.units.erallab.hmsrobots.core.controllers.snn.converters.stv.SpikeTrainToValueConverter;
import it.units.erallab.hmsrobots.core.controllers.snn.converters.vts.ValueToSpikeTrainConverter;
import it.units.erallab.hmsrobots.util.SerializationUtils;
import it.units.erallab.utils.SnnUtils;

import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.function.Function;

/**
 * @author eric
 */
public class MSNWithConverters implements NamedProvider<PrototypedFunctionBuilder<List<Double>, TimedRealFunction>> {

  @Override
  public PrototypedFunctionBuilder<List<Double>, TimedRealFunction> build(Map<String, String> params) {
    double innerLayerRatio = Double.parseDouble(params.getOrDefault("r", "0.65"));
    int nOfInnerLayers = Integer.parseInt(params.getOrDefault("nIL", "1"));
    SpikingFunction spikingFunction = SnnUtils.buildSpikingFunction(params.getOrDefault("m", ""));
    ValueToSpikeTrainConverter valueToSpikeTrainConverter = SnnUtils.buildValueToSpikeTrainConverter(params.getOrDefault("vts", ""));
    SpikeTrainToValueConverter spikeTrainToValueConverter = SnnUtils.buildSpikeTrainToValueConverter(params.getOrDefault("stv", ""));
    return new PrototypedFunctionBuilder<>() {
      @Override
      public Function<List<Double>, TimedRealFunction> buildFor(TimedRealFunction function) {
        return values -> {
          int nOfInputs = function.getInputDimension();
          int nOfOutputs = function.getOutputDimension();
          int[] innerNeurons = MLP.innerNeurons(nOfInputs, nOfOutputs, innerLayerRatio, nOfInnerLayers);
          int nOfWeights = MultilayerSpikingNetwork.countWeights(nOfInputs, innerNeurons, nOfOutputs);
          if (nOfWeights != values.size()) {
            throw new IllegalArgumentException(String.format(
                "Wrong number of values for weights: %d expected, %d found",
                nOfWeights,
                values.size()
            ));
          }
          MultilayerSpikingNetwork multilayerSpikingNetwork = new MultilayerSpikingNetwork(
              nOfInputs,
              innerNeurons,
              nOfOutputs,
              values.stream().mapToDouble(d -> d).toArray(),
              (x, y) -> SerializationUtils.clone(spikingFunction)
          );
          return new MultilayerSpikingNetworkWithConverters<>(
              multilayerSpikingNetwork,
              valueToSpikeTrainConverter,
              spikeTrainToValueConverter
          );
        };
      }

      @Override
      public List<Double> exampleFor(TimedRealFunction function) {
        return Collections.nCopies(
            MultilayerSpikingNetwork.countWeights(
                MultiLayerPerceptron.countNeurons(
                    function.getInputDimension(),
                    MLP.innerNeurons(function.getInputDimension(), function.getOutputDimension(), innerLayerRatio, nOfInnerLayers),
                    function.getOutputDimension())
            ),
            0d
        );
      }
    };
  }


}
