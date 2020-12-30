package it.units.erallab.builder.robot;

import it.units.erallab.RealFunction;
import it.units.erallab.builder.PrototypedFunctionBuilder;
import it.units.erallab.builder.phenotype.MLP;
import it.units.erallab.hmsrobots.core.controllers.DistributedSensing;
import it.units.erallab.hmsrobots.core.controllers.MultiLayerPerceptron;
import it.units.erallab.hmsrobots.core.objects.Robot;
import it.units.erallab.hmsrobots.core.objects.SensingVoxel;
import it.units.erallab.hmsrobots.core.sensors.Constant;
import it.units.erallab.hmsrobots.core.sensors.Sensor;
import it.units.erallab.hmsrobots.tasks.locomotion.Locomotion;
import it.units.erallab.hmsrobots.util.Grid;
import it.units.erallab.hmsrobots.util.RobotUtils;
import it.units.erallab.hmsrobots.util.SerializationUtils;
import it.units.erallab.hmsrobots.util.Utils;
import it.units.erallab.hmsrobots.viewers.GridOnlineViewer;
import it.units.malelab.jgea.core.Factory;
import it.units.malelab.jgea.representation.sequence.FixedLengthListFactory;
import it.units.malelab.jgea.representation.sequence.numeric.UniformDoubleFactory;
import org.apache.commons.math3.stat.descriptive.rank.Percentile;
import org.dyn4j.dynamics.Settings;

import java.util.List;
import java.util.Objects;
import java.util.Random;
import java.util.function.Function;
import java.util.stream.Collectors;

/**
 * @author eric
 */
public class SensorAndBodyAndHomoDistributed implements PrototypedFunctionBuilder<List<RealFunction>, Robot<? extends SensingVoxel>> {
  private final int signals;
  private final double percentile;
  private final boolean withPositionSensors;

  public SensorAndBodyAndHomoDistributed(int signals, double percentile, boolean withPositionSensors) {
    this.signals = signals;
    this.percentile = percentile;
    this.withPositionSensors = withPositionSensors;
  }

  @Override
  public Function<List<RealFunction>, Robot<? extends SensingVoxel>> buildFor(Robot<? extends SensingVoxel> robot) {
    int w = robot.getVoxels().getW();
    int h = robot.getVoxels().getH();
    List<Sensor> prototypeSensors = getPrototypeSensors(robot);
    int nOfInputs = DistributedSensing.nOfInputs(new SensingVoxel(prototypeSensors.subList(0, 1)), signals) + (withPositionSensors ? 2 : 0);
    int nOfOutputs = DistributedSensing.nOfOutputs(new SensingVoxel(prototypeSensors.subList(0, 1)), signals);
    //build body
    return pair -> {
      if (pair.size() != 2) {
        throw new IllegalArgumentException(String.format(
            "Wrong number of functions: 2 expected, %d found",
            pair.size()
        ));
      }
      RealFunction bodyFunction = pair.get(0);
      RealFunction brainFunction = pair.get(1);
      //check function sizes
      if (bodyFunction.getNOfInputs() != 2 || bodyFunction.getNOfOutputs() != prototypeSensors.size()) {
        throw new IllegalArgumentException(String.format(
            "Wrong number of body function args: 2->%d expected, %d->%d found",
            prototypeSensors.size(),
            bodyFunction.getNOfInputs(),
            bodyFunction.getNOfOutputs()
        ));
      }
      if (brainFunction.getNOfInputs() != nOfInputs) {
        throw new IllegalArgumentException(String.format(
            "Wrong number of brain function input args: %d expected, %d found",
            nOfInputs,
            brainFunction.getNOfInputs()
        ));
      }
      if (brainFunction.getNOfOutputs() != nOfOutputs) {
        throw new IllegalArgumentException(String.format(
            "Wrong number of brain function output args: %d expected, %d found",
            nOfOutputs,
            brainFunction.getNOfOutputs()
        ));
      }
      //build body
      Grid<double[]> values = Grid.create(w, h, (x, y) -> bodyFunction.apply(new double[]{(double) x / (double) w, (double) y / (double) h}));
      double threshold = new Percentile().evaluate(
          values.values().stream().mapToDouble(this::max).toArray(),
          percentile
      );
      values = Grid.create(values, vs -> max(vs) >= threshold ? vs : null);
      values = Utils.gridLargestConnected(values, Objects::nonNull);
      values = Utils.cropGrid(values, Objects::nonNull);
      Grid<SensingVoxel> body = new Grid<>(values.getW(), values.getH(), null);
      for (int x = 0; x < body.getW(); x++) {
        for (int y = 0; y < body.getH(); y++) {
          int rx = (int) Math.floor((double) x / (double) body.getW() * (double) w);
          int ry = (int) Math.floor((double) y / (double) body.getH() * (double) h);
          if (values.get(x, y) != null) {
            List<Sensor> availableSensors = robot.getVoxels().get(rx, ry) != null ? robot.getVoxels().get(rx, ry).getSensors() : prototypeSensors;
            body.set(x, y, new SensingVoxel(withPositionSensors ?
                List.of(
                    SerializationUtils.clone(availableSensors.get(indexOfMax(values.get(x, y)))),
                    new Constant((double) x / ((double) body.getW() - 1d), (double) y / ((double) body.getH() - 1d))
                ) :
                List.of(
                    SerializationUtils.clone(availableSensors.get(indexOfMax(values.get(x, y))))
                )
            ));
          }
        }
      }
      if (body.values().stream().noneMatch(Objects::nonNull)) {
        body = Grid.create(1, 1, new SensingVoxel(List.of(SerializationUtils.clone(prototypeSensors.get(indexOfMax(values.get(0, 0)))))));
      }
      //build brain
      DistributedSensing controller = new DistributedSensing(body, signals);
      for (Grid.Entry<? extends SensingVoxel> entry : body) {
        if (entry.getValue() != null) {
          controller.getFunctions().set(entry.getX(), entry.getY(), SerializationUtils.clone(brainFunction));
        }
      }
      return new Robot<>(controller, body);
    };
  }

  private double max(double[] vs) {
    double max = vs[0];
    for (int i = 1; i < vs.length; i++) {
      max = Math.max(max, vs[i]);
    }
    return max;
  }

  private int indexOfMax(double[] vs) {
    int indexOfMax = 0;
    for (int i = 1; i < vs.length; i++) {
      if (vs[i] > vs[indexOfMax]) {
        indexOfMax = i;
      }
    }
    return indexOfMax;
  }

  @Override
  public List<RealFunction> exampleFor(Robot<? extends SensingVoxel> robot) {
    List<Sensor> sensors = getPrototypeSensors(robot);
    return List.of(
        RealFunction.from(2, sensors.size(), d -> d),
        RealFunction.from(
            DistributedSensing.nOfInputs(new SensingVoxel(sensors.subList(0, 1)), signals) + (withPositionSensors ? 2 : 0),
            DistributedSensing.nOfOutputs(new SensingVoxel(sensors.subList(0, 1)), signals),
            d -> d
        )
    );
  }

  private List<Sensor> getPrototypeSensors(Robot<? extends SensingVoxel> robot) {
    SensingVoxel voxelPrototype = robot.getVoxels().values().stream().filter(Objects::nonNull).findFirst().orElse(null);
    if (voxelPrototype == null) {
      throw new IllegalArgumentException("Target robot has no voxels");
    }
    if (voxelPrototype.getSensors().isEmpty()) {
      throw new IllegalArgumentException("Target robot has no sensors");
    }
    if (voxelPrototype.getSensors().stream().mapToInt(s -> s.domains().length).distinct().count() != 1) {
      throw new IllegalArgumentException(String.format(
          "Target robot has sensors with different number of outputs: %s",
          voxelPrototype.getSensors().stream().mapToInt(s -> s.domains().length).distinct()
      ));
    }
    return voxelPrototype.getSensors();
  }

  public static void main(String[] args) {
    Robot<? extends SensingVoxel> target = new Robot<>(
        null,
        RobotUtils.buildSensorizingFunction("uniform-t+l1-0").apply(RobotUtils.buildShape("box-5x5"))
    );
    PrototypedFunctionBuilder<List<Double>, Robot<? extends SensingVoxel>> builder = new SensorAndBodyAndHomoDistributed(2, 50, true)
        .compose(PrototypedFunctionBuilder.of(List.of(
            new MLP(2d, 3, MultiLayerPerceptron.ActivationFunction.SIN),
            new MLP(1.5d, 2)
        )))
        .compose(PrototypedFunctionBuilder.merger());
    Factory<List<Double>> factory = new FixedLengthListFactory<>(builder.exampleFor(target).size(), new UniformDoubleFactory(-1d, 1d));
    Random random = new Random();
    Robot<? extends SensingVoxel> robot = builder.buildFor(target).apply(factory.independent().build(random));
    Locomotion locomotion = new Locomotion(30, Locomotion.createTerrain("flat"), new Settings());
    GridOnlineViewer.run(
        locomotion,
        factory.build(16, random).stream().map(vs -> builder.buildFor(target).apply(vs)).collect(Collectors.toList())
    );
  }
}
