package it.units.erallab.builder.robot;

import it.units.erallab.RealFunction;
import it.units.erallab.builder.PrototypedFunctionBuilder;
import it.units.erallab.builder.phenotype.MLP;
import it.units.erallab.hmsrobots.core.controllers.CentralizedSensing;
import it.units.erallab.hmsrobots.core.controllers.MultiLayerPerceptron;
import it.units.erallab.hmsrobots.core.objects.Robot;
import it.units.erallab.hmsrobots.core.objects.SensingVoxel;
import it.units.erallab.hmsrobots.core.sensors.Sensor;
import it.units.erallab.hmsrobots.tasks.locomotion.Locomotion;
import it.units.erallab.hmsrobots.util.Grid;
import it.units.erallab.hmsrobots.util.RobotUtils;
import it.units.erallab.hmsrobots.util.SerializationUtils;
import it.units.erallab.hmsrobots.viewers.GridOnlineViewer;
import it.units.malelab.jgea.core.Factory;
import it.units.malelab.jgea.representation.sequence.FixedLengthListFactory;
import it.units.malelab.jgea.representation.sequence.numeric.UniformDoubleFactory;
import org.dyn4j.dynamics.Settings;

import java.util.List;
import java.util.Objects;
import java.util.Random;
import java.util.function.Function;
import java.util.stream.Collectors;

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

  public static void main(String[] args) {
    Robot<? extends SensingVoxel> target = new Robot<>(
        null,
        RobotUtils.buildSensorizingFunction("uniform-t+ax+ay+l1-0").apply(RobotUtils.buildShape("biped-8x5"))
    );
    PrototypedFunctionBuilder<List<Double>, Robot<? extends SensingVoxel>> builder = new SensorCentralized()
        .compose(PrototypedFunctionBuilder.of(List.of(
            new MLP(4d, 5, MultiLayerPerceptron.ActivationFunction.SIN),
            new MLP(2d, 1)
        )))
        .compose(PrototypedFunctionBuilder.merger());
    System.out.printf("Size: %d%n", builder.exampleFor(target).size());
    Factory<List<Double>> factory = new FixedLengthListFactory<>(builder.exampleFor(target).size(), new UniformDoubleFactory(-1d, 1d));
    Random random = new Random();
    Locomotion locomotion = new Locomotion(30, Locomotion.createTerrain("flat"), new Settings());
    GridOnlineViewer.run(
        locomotion,
        factory.build(9, random).stream().map(vs -> builder.buildFor(target).apply(vs)).collect(Collectors.toList())
    );
  }

}
