package it.units.erallab.builder.robot;

import it.units.erallab.builder.NamedProvider;
import it.units.erallab.builder.PrototypedFunctionBuilder;
import it.units.erallab.hmsrobots.core.controllers.CentralizedSensing;
import it.units.erallab.hmsrobots.core.controllers.RealFunction;
import it.units.erallab.hmsrobots.core.controllers.TimedRealFunction;
import it.units.erallab.hmsrobots.core.objects.Robot;
import it.units.erallab.hmsrobots.core.objects.Voxel;
import it.units.erallab.hmsrobots.core.sensors.Sensor;
import it.units.erallab.hmsrobots.util.Grid;
import it.units.erallab.hmsrobots.util.SerializationUtils;

import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.function.Function;

/**
 * @author eric on 2021/01/02 for VSREvolution
 */
public class SensorBrainCentralized implements NamedProvider<PrototypedFunctionBuilder<List<TimedRealFunction>,
    Robot>> {

  @Override
  public PrototypedFunctionBuilder<List<TimedRealFunction>, Robot> build(Map<String, String> params) {
    return new PrototypedFunctionBuilder<>() {
      @Override
      public Function<List<TimedRealFunction>, Robot> buildFor(Robot robot) {
        List<Sensor> prototypeSensors = BodySensorBrainHomoDistributed.getPrototypeSensors(robot);
        int nOfVoxels = (int) robot.getVoxels().values().stream().filter(Objects::nonNull).count();
        int sensorDim = prototypeSensors.get(0).getDomains().length;
        return pair -> {
          if (pair.size() != 2) {
            throw new IllegalArgumentException(String.format(
                "Wrong number of functions: 2 expected, %d found",
                pair.size()
            ));
          }
          TimedRealFunction sensorizingFunction = pair.get(0);
          TimedRealFunction brainFunction = pair.get(1);
          //check function sizes
          if (sensorizingFunction.getInputDimension() != 2 || sensorizingFunction.getOutputDimension() != prototypeSensors.size()) {
            throw new IllegalArgumentException(String.format(
                "Wrong number of sensorizing function args: 2->%d expected, %d->%d found",
                prototypeSensors.size(),
                sensorizingFunction.getInputDimension(),
                sensorizingFunction.getOutputDimension()
            ));
          }
          if (brainFunction.getInputDimension() != nOfVoxels * sensorDim) {
            throw new IllegalArgumentException(String.format(
                "Wrong number of brain function input args: %d expected, %d found",
                nOfVoxels * sensorDim,
                brainFunction.getInputDimension()
            ));
          }
          if (brainFunction.getOutputDimension() != nOfVoxels) {
            throw new IllegalArgumentException(String.format(
                "Wrong number of brain function output args: %d expected, %d found",
                nOfVoxels,
                brainFunction.getOutputDimension()
            ));
          }
          //sensorize body
          int w = robot.getVoxels().getW();
          int h = robot.getVoxels().getH();
          Grid<double[]> values = Grid.create(
              w, h,
              (x, y) -> sensorizingFunction.apply(
                  0,
                  new double[]{(double) x / ((double) w - 1d), (double) y / ((double) h - 1d)}
              )
          );
          Grid<Voxel> body = Grid.create(
              w, h,
              (x, y) -> {
                if (robot.getVoxels().get(x, y) == null) {
                  return null;
                }
                List<Sensor> availableSensors = robot.getVoxels().get(x, y).getSensors();
                return new Voxel(List.of(
                    SerializationUtils.clone(availableSensors.get(BodySensorBrainHomoDistributed.indexOfMax(values.get(
                        x,
                        y
                    ))))
                ));
              }
          );
          return new Robot(new CentralizedSensing(body, brainFunction), body);
        };
      }

      @Override
      public List<TimedRealFunction> exampleFor(Robot robot) {
        List<Sensor> prototypeSensors = BodySensorBrainHomoDistributed.getPrototypeSensors(robot);
        int nOfVoxels = (int) robot.getVoxels().values().stream().filter(Objects::nonNull).count();
        int sensorDim = prototypeSensors.get(0).getDomains().length;
        return List.of(
            RealFunction.build(d -> d, 2, prototypeSensors.size()),
            RealFunction.build(d -> d, nOfVoxels * sensorDim, nOfVoxels)
        );
      }
    };
  }

}
