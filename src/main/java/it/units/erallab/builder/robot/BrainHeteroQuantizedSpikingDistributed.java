package it.units.erallab.builder.robot;

import it.units.erallab.builder.NamedProvider;
import it.units.erallab.builder.PrototypedFunctionBuilder;
import it.units.erallab.hmsrobots.core.controllers.DistributedSensing;
import it.units.erallab.hmsrobots.core.controllers.snndiscr.QuantizedDistributedSpikingSensing;
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

/**
 * @author giorgia
 */
public class BrainHeteroQuantizedSpikingDistributed implements NamedProvider<PrototypedFunctionBuilder<Grid<QuantizedMultivariateSpikingFunction>, Robot>> {

  @Override
  public PrototypedFunctionBuilder<Grid<QuantizedMultivariateSpikingFunction>, Robot> build(Map<String, String> params) {
    int signals = Integer.parseInt(params.getOrDefault("s", "1"));
    QuantizedValueToSpikeTrainConverter valueToSpikeTrainConverter = SnnUtils.buildQuantizedValueToSpikeTrainConverter(params.getOrDefault("vts", ""));
    QuantizedSpikeTrainToValueConverter spikeTrainToValueConverter = SnnUtils.buildQuantizedSpikeTrainToValueConverter(params.getOrDefault("stv", ""));
    return new PrototypedFunctionBuilder<>() {
      @Override
      public Function<Grid<QuantizedMultivariateSpikingFunction>, Robot> buildFor(Robot robot) {
        Grid<int[]> dims = getIODims(robot, signals);
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
          for (Grid.Entry<int[]> entry : dims) {
            if (entry.value() == null) {
              continue;
            }
            if (functions.get(entry.key().x(), entry.key().y()).getInputDimension() != entry.value()[0]) {
              throw new IllegalArgumentException(String.format(
                  "Wrong number of function input args at (%d,%d): %d expected, %d found",
                  entry.key().x(), entry.key().y(),
                  entry.value()[0],
                  functions.get(entry.key().x(), entry.key().y()).getInputDimension()
              ));
            }
            if (functions.get(entry.key().x(), entry.key().y()).getOutputDimension() != entry.value()[1]) {
              throw new IllegalArgumentException(String.format(
                  "Wrong number of function output args at (%d,%d): %d expected, %d found",
                  entry.key().x(), entry.key().y(),
                  entry.value()[1],
                  functions.get(entry.key().x(), entry.key().y()).getOutputDimension()
              ));
            }
          }
          //return
          QuantizedDistributedSpikingSensing controller = new QuantizedDistributedSpikingSensing(body, signals, new QuantizedLIFNeuron(), valueToSpikeTrainConverter, spikeTrainToValueConverter);
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
            getIODims(robot, signals),
            dim -> dim == null ? null : QuantizedMultivariateSpikingFunction.build(d -> d, dim[0], dim[1])
        );
      }
    };
  }

  private Grid<int[]> getIODims(Robot robot, int signals) {
    Grid<Voxel> body = robot.getVoxels();
    return Grid.create(
        body,
        v -> v == null ? null : new int[]{
            DistributedSensing.nOfInputs(v, signals),
            DistributedSensing.nOfOutputs(v, signals)
        }
    );
  }

}
