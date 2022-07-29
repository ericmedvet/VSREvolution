package it.units.erallab.builder.robot;

import it.units.erallab.builder.NamedProvider;
import it.units.erallab.builder.PrototypedFunctionBuilder;
import it.units.erallab.hmsrobots.core.controllers.snndiscr.QuantizedDistributedSpikingSensing;
import it.units.erallab.hmsrobots.core.controllers.snndiscr.QuantizedDistributedSpikingSensingNonDirectional;
import it.units.erallab.hmsrobots.core.controllers.snndiscr.QuantizedLIFNeuron;
import it.units.erallab.hmsrobots.core.controllers.snndiscr.QuantizedMultivariateSpikingFunction;
import it.units.erallab.hmsrobots.core.controllers.snndiscr.converters.stv.QuantizedSpikeTrainToValueConverter;
import it.units.erallab.hmsrobots.core.controllers.snndiscr.converters.vts.QuantizedValueToSpikeTrainConverter;
import it.units.erallab.hmsrobots.core.objects.Robot;
import it.units.erallab.hmsrobots.core.objects.Voxel;
import it.units.erallab.hmsrobots.util.Grid;
import it.units.erallab.hmsrobots.util.SerializationUtils;
import it.units.erallab.utils.SnnUtils;

import java.util.Map;
import java.util.function.Function;

import static it.units.erallab.builder.robot.BrainHomoDistributed.getIODim;

/**
 * @author giorgia
 */
public class BrainHomoQuantizedSpikingDistributed implements NamedProvider<PrototypedFunctionBuilder<QuantizedMultivariateSpikingFunction, Robot>> {

  @Override
  public PrototypedFunctionBuilder<QuantizedMultivariateSpikingFunction, Robot> build(Map<String, String> params) {
    int signals = Integer.parseInt(params.getOrDefault("s", "1"));
    boolean directional = params.getOrDefault("d", "t").startsWith("t");
    QuantizedValueToSpikeTrainConverter valueToSpikeTrainConverter = SnnUtils.buildQuantizedValueToSpikeTrainConverter(params.getOrDefault("vts", ""));
    QuantizedSpikeTrainToValueConverter spikeTrainToValueConverter = SnnUtils.buildQuantizedSpikeTrainToValueConverter(params.getOrDefault("stv", ""));
    return new PrototypedFunctionBuilder<>() {
      @Override
      public Function<QuantizedMultivariateSpikingFunction, Robot> buildFor(Robot robot) {
        int[] dim = getIODim(robot, signals, directional);
        Grid<Voxel> body = robot.getVoxels();
        int nOfInputs = dim[0];
        int nOfOutputs = dim[1];
        return function -> {
          if (function.getInputDimension() != nOfInputs) {
            throw new IllegalArgumentException(String.format(
                "Wrong number of function input args: %d expected, %d found",
                nOfInputs,
                function.getInputDimension()
            ));
          }
          if (function.getOutputDimension() != nOfOutputs) {
            throw new IllegalArgumentException(String.format(
                "Wrong number of function output args: %d expected, %d found",
                nOfOutputs,
                function.getOutputDimension()
            ));
          }
          QuantizedDistributedSpikingSensing controller = directional ?
              new QuantizedDistributedSpikingSensing(body, signals, new QuantizedLIFNeuron(), valueToSpikeTrainConverter, spikeTrainToValueConverter) :
              new QuantizedDistributedSpikingSensingNonDirectional(body, signals, new QuantizedLIFNeuron(), valueToSpikeTrainConverter, spikeTrainToValueConverter);
          for (Grid.Entry<Voxel> entry : body) {
            if (entry.value() != null) {
              controller.getFunctions().set(entry.key().x(), entry.key().y(), SerializationUtils.clone(function));
            }
          }
          return new Robot(
              controller,
              SerializationUtils.clone(body)
          );
        };
      }

      @Override
      public QuantizedMultivariateSpikingFunction exampleFor(Robot robot) {
        int[] dim = getIODim(robot, signals, directional);
        return QuantizedMultivariateSpikingFunction.build(d -> d, dim[0], dim[1]);
      }
    };
  }

}
