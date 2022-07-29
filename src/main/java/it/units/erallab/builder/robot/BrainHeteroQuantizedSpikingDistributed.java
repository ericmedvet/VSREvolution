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

import static it.units.erallab.builder.robot.BrainHeteroDistributed.getIODims;

/**
 * @author giorgia
 */
public class BrainHeteroQuantizedSpikingDistributed implements NamedProvider<PrototypedFunctionBuilder<Grid<QuantizedMultivariateSpikingFunction>, Robot>> {

  @Override
  public PrototypedFunctionBuilder<Grid<QuantizedMultivariateSpikingFunction>, Robot> build(Map<String, String> params) {
    int signals = Integer.parseInt(params.getOrDefault("s", "1"));
    boolean directional = params.getOrDefault("d", "t").startsWith("t");
    QuantizedValueToSpikeTrainConverter valueToSpikeTrainConverter = SnnUtils.buildQuantizedValueToSpikeTrainConverter(params.getOrDefault("vts", ""));
    QuantizedSpikeTrainToValueConverter spikeTrainToValueConverter = SnnUtils.buildQuantizedSpikeTrainToValueConverter(params.getOrDefault("stv", ""));
    return new PrototypedFunctionBuilder<>() {
      @Override
      public Function<Grid<QuantizedMultivariateSpikingFunction>, Robot> buildFor(Robot robot) {
        Grid<BrainHeteroDistributed.IODimensions> dims = getIODims(robot, signals, directional);
        Grid<Voxel> body = robot.getVoxels();
        return functions -> {
          //check
          if (dims.getW() != functions.getW() || dims.getH() != functions.getH()) {
            throw new IllegalArgumentException(String.format(
                "Wrong size of functions grid: %dx%d expected, %dx%d found",
                dims.getW(), dims.getH(),
                functions.getW(), functions.getH()
            ));
          }
          for (Grid.Entry<BrainHeteroDistributed.IODimensions> entry : dims) {
            if (entry.value() == null) {
              continue;
            }
            if (functions.get(entry.key().x(), entry.key().y()).getInputDimension() != entry.value().input()) {
              throw new IllegalArgumentException(String.format(
                  "Wrong number of function input args at (%d,%d): %d expected, %d found",
                  entry.key().x(), entry.key().y(),
                  entry.value().input(),
                  functions.get(entry.key().x(), entry.key().y()).getInputDimension()
              ));
            }
            if (functions.get(entry.key().x(), entry.key().y()).getOutputDimension() != entry.value().output()) {
              throw new IllegalArgumentException(String.format(
                  "Wrong number of function output args at (%d,%d): %d expected, %d found",
                  entry.key().x(), entry.key().y(),
                  entry.value().output(),
                  functions.get(entry.key().x(), entry.key().y()).getOutputDimension()
              ));
            }
          }
          //return
          QuantizedDistributedSpikingSensing controller = directional ?
              new QuantizedDistributedSpikingSensing(body, signals, new QuantizedLIFNeuron(), valueToSpikeTrainConverter, spikeTrainToValueConverter) :
              new QuantizedDistributedSpikingSensingNonDirectional(body, signals, new QuantizedLIFNeuron(), valueToSpikeTrainConverter, spikeTrainToValueConverter);
          for (Grid.Entry<Voxel> entry : body) {
            if (entry.value() != null) {
              controller.getFunctions().set(entry.key().x(), entry.key().y(), functions.get(entry.key().x(), entry.key().y()));
            }
          }
          return new Robot(
              controller,
              SerializationUtils.clone(body)
          );
        };
      }

      @Override
      public Grid<QuantizedMultivariateSpikingFunction> exampleFor(Robot robot) {
        return Grid.create(
            getIODims(robot, signals, directional),
            dim -> dim == null ? null : QuantizedMultivariateSpikingFunction.build(d -> d, dim.input(), dim.output())
        );
      }
    };
  }


}
