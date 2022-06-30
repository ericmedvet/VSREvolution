package it.units.erallab.builder.spikingfunction;

import it.units.erallab.builder.NamedProvider;
import it.units.erallab.builder.PrototypedFunctionBuilder;
import it.units.erallab.builder.function.MLP;
import it.units.erallab.hmsrobots.core.controllers.MultiLayerPerceptron;
import it.units.erallab.hmsrobots.core.controllers.snndiscr.QuantizedMultilayerSpikingNetwork;
import it.units.erallab.hmsrobots.core.controllers.snndiscr.QuantizedMultivariateSpikingFunction;
import it.units.erallab.hmsrobots.core.controllers.snndiscr.QuantizedSpikingFunction;
import it.units.erallab.hmsrobots.util.SerializationUtils;
import it.units.erallab.utils.SnnUtils;

import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.function.Function;

/**
 * @author eric
 */
public class QuantizedMSN implements NamedProvider<PrototypedFunctionBuilder<List<Double>, QuantizedMultivariateSpikingFunction>> {

  @Override
  public PrototypedFunctionBuilder<List<Double>, QuantizedMultivariateSpikingFunction> build(Map<String, String> params) {
    double innerLayerRatio = Double.parseDouble(params.getOrDefault("r", "0.65"));
    int nOfInnerLayers = Integer.parseInt(params.getOrDefault("nIL", "1"));
    QuantizedSpikingFunction spikingFunction = SnnUtils.buildQuantizedSpikingFunction(params.getOrDefault("m", ""));
    return new PrototypedFunctionBuilder<>() {
      @Override
      public Function<List<Double>, QuantizedMultivariateSpikingFunction> buildFor(QuantizedMultivariateSpikingFunction function) {
        return values -> {
          int nOfInputs = function.getInputDimension();
          int nOfOutputs = function.getOutputDimension();
          int[] innerNeurons = MLP.innerNeurons(nOfInputs, nOfOutputs, innerLayerRatio, nOfInnerLayers);
          int nOfWeights = QuantizedMultilayerSpikingNetwork.countWeights(nOfInputs, innerNeurons, nOfOutputs);
          if (nOfWeights != values.size()) {
            throw new IllegalArgumentException(String.format(
                "Wrong number of values for weights: %d expected, %d found",
                nOfWeights,
                values.size()
            ));
          }
          return new QuantizedMultilayerSpikingNetwork(
              nOfInputs,
              innerNeurons,
              nOfOutputs,
              values.stream().mapToDouble(d -> d).toArray(),
              (x, y) -> SerializationUtils.clone(spikingFunction)
          );
        };
      }

      @Override
      public List<Double> exampleFor(QuantizedMultivariateSpikingFunction function) {
        return Collections.nCopies(
            QuantizedMultilayerSpikingNetwork.countWeights(
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
