package it.units.erallab.builder.robot;

import it.units.erallab.builder.NamedProvider;
import it.units.erallab.builder.PrototypedFunctionBuilder;
import it.units.erallab.hmsrobots.core.controllers.DistributedSensing;
import it.units.erallab.hmsrobots.core.controllers.snn.DistributedSpikingSensing;
import it.units.erallab.hmsrobots.core.controllers.snn.LIFNeuron;
import it.units.erallab.hmsrobots.core.controllers.snn.MultivariateSpikingFunction;
import it.units.erallab.hmsrobots.core.controllers.snn.converters.stv.SpikeTrainToValueConverter;
import it.units.erallab.hmsrobots.core.controllers.snn.converters.vts.ValueToSpikeTrainConverter;
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
public class BrainHeteroSpikingDistributed implements NamedProvider<PrototypedFunctionBuilder<Grid<MultivariateSpikingFunction>, Robot>> {

  @Override
  public PrototypedFunctionBuilder<Grid<MultivariateSpikingFunction>, Robot> build(Map<String, String> params) {
    int signals = Integer.parseInt(params.getOrDefault("s", "1"));
    ValueToSpikeTrainConverter valueToSpikeTrainConverter = SnnUtils.buildValueToSpikeTrainConverter(params.getOrDefault("vts", ""));
    SpikeTrainToValueConverter spikeTrainToValueConverter = SnnUtils.buildSpikeTrainToValueConverter(params.getOrDefault("stv", ""));
    return new PrototypedFunctionBuilder<>() {
      @Override
      public Function<Grid<MultivariateSpikingFunction>, Robot> buildFor(Robot robot) {
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
          DistributedSpikingSensing controller = new DistributedSpikingSensing(body, signals, new LIFNeuron(), valueToSpikeTrainConverter, spikeTrainToValueConverter);
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
      public Grid<MultivariateSpikingFunction> exampleFor(Robot robot) {
        return Grid.create(
            getIODims(robot, signals),
            dim -> dim == null ? null : MultivariateSpikingFunction.build(d -> d, dim[0], dim[1])
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
