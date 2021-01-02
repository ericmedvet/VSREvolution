package it.units.erallab.builder.robot;

import it.units.erallab.RealFunction;
import it.units.erallab.builder.PrototypedFunctionBuilder;
import it.units.erallab.hmsrobots.core.controllers.CentralizedSensing;
import it.units.erallab.hmsrobots.core.objects.Robot;
import it.units.erallab.hmsrobots.core.objects.SensingVoxel;
import it.units.erallab.hmsrobots.core.sensors.Sensor;
import it.units.erallab.hmsrobots.util.Grid;
import it.units.erallab.hmsrobots.util.SerializationUtils;

import java.util.List;
import java.util.Objects;
import java.util.function.Function;

/**
 * @author eric on 2021/01/02 for VSREvolution
 */
public class SensorCentralized implements PrototypedFunctionBuilder<List<RealFunction>, Robot<? extends SensingVoxel>> {

  @Override
  public Function<List<RealFunction>, Robot<? extends SensingVoxel>> buildFor(Robot<? extends SensingVoxel> robot) {
    List<Sensor> prototypeSensors = SensorAndBodyAndHomoDistributed.getPrototypeSensors(robot);
    int nOfVoxels = (int) robot.getVoxels().values().stream().filter(Objects::nonNull).count();
    int sensorDim = prototypeSensors.get(0).domains().length;
    return pair -> {
      if (pair.size() != 2) {
        throw new IllegalArgumentException(String.format(
            "Wrong number of functions: 2 expected, %d found",
            pair.size()
        ));
      }
      RealFunction sensorizingFunction = pair.get(0);
      RealFunction brainFunction = pair.get(1);
      //check function sizes
      if (sensorizingFunction.getNOfInputs() != 2 || sensorizingFunction.getNOfOutputs() != prototypeSensors.size()) {
        throw new IllegalArgumentException(String.format(
            "Wrong number of sensorizing function args: 2->%d expected, %d->%d found",
            prototypeSensors.size(),
            sensorizingFunction.getNOfInputs(),
            sensorizingFunction.getNOfOutputs()
        ));
      }
      if (brainFunction.getNOfInputs() != nOfVoxels * sensorDim) {
        throw new IllegalArgumentException(String.format(
            "Wrong number of brain function input args: %d expected, %d found",
            nOfVoxels * sensorDim,
            brainFunction.getNOfInputs()
        ));
      }
      if (brainFunction.getNOfOutputs() != nOfVoxels) {
        throw new IllegalArgumentException(String.format(
            "Wrong number of brain function output args: %d expected, %d found",
            nOfVoxels,
            brainFunction.getNOfOutputs()
        ));
      }
      //sensorize body
      int w = robot.getVoxels().getW();
      int h = robot.getVoxels().getH();
      Grid<double[]> values = Grid.create(
          w, h,
          (x, y) -> sensorizingFunction.apply(new double[]{(double) x / ((double) w - 1d), (double) y / ((double) h - 1d)})
      );
      Grid<? extends SensingVoxel> body = Grid.create(
          w, h,
          (x, y) -> {
            if (robot.getVoxels().get(x, y) == null) {
              return null;
            }
            List<Sensor> availableSensors = robot.getVoxels().get(x, y).getSensors();
            return new SensingVoxel(List.of(
                SerializationUtils.clone(availableSensors.get(SensorAndBodyAndHomoDistributed.indexOfMax(values.get(x, y))))
            ));
          }
      );
      return new Robot<>(new CentralizedSensing(body, brainFunction), body);
    };
  }

  @Override
  public List<RealFunction> exampleFor(Robot<? extends SensingVoxel> robot) {
    List<Sensor> prototypeSensors = SensorAndBodyAndHomoDistributed.getPrototypeSensors(robot);
    int nOfVoxels = (int) robot.getVoxels().values().stream().filter(Objects::nonNull).count();
    int sensorDim = prototypeSensors.get(0).domains().length;
    return List.of(
        RealFunction.from(2, prototypeSensors.size(), d -> d),
        RealFunction.from(
            nOfVoxels * sensorDim,
            nOfVoxels,
            v -> v
        )
    );
  }

}
