package it.units.erallab.builder.robot;

import it.units.erallab.RealFunction;
import it.units.erallab.builder.PrototypedFunctionBuilder;
import it.units.erallab.builder.phenotype.MLP;
import it.units.erallab.hmsrobots.core.controllers.DistributedSensing;
import it.units.erallab.hmsrobots.core.controllers.MultiLayerPerceptron;
import it.units.erallab.hmsrobots.core.objects.Robot;
import it.units.erallab.hmsrobots.core.objects.SensingVoxel;
import it.units.erallab.hmsrobots.core.sensors.Sensor;
import it.units.erallab.hmsrobots.util.Grid;
import it.units.erallab.hmsrobots.util.SerializationUtils;
import it.units.erallab.hmsrobots.util.Utils;
import org.apache.commons.math3.stat.descriptive.rank.Percentile;

import java.util.List;
import java.util.Objects;
import java.util.Random;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * @author eric
 */
public class SensorAndBodyAndHomoDistributed implements PrototypedFunctionBuilder<List<RealFunction>, Robot<? extends SensingVoxel>> {
  private final int signals;
  private final double percentile;

  public SensorAndBodyAndHomoDistributed(int signals, double percentile) {
    this.signals = signals;
    this.percentile = percentile;
  }

  @Override
  public Function<List<RealFunction>, Robot<? extends SensingVoxel>> buildFor(Robot<? extends SensingVoxel> robot) {
    int w = robot.getVoxels().getW();
    int h = robot.getVoxels().getH();
    List<Sensor> sensors = getPrototypeSensors(robot);
    int nOfInputs = DistributedSensing.nOfInputs(new SensingVoxel(sensors.subList(0, 1)), signals);
    int nOfOutputs = DistributedSensing.nOfOutputs(new SensingVoxel(sensors.subList(0, 1)), signals);
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
      if (bodyFunction.getNOfInputs() != 2 || bodyFunction.getNOfOutputs() != sensors.size()) {
        throw new IllegalArgumentException(String.format(
            "Wrong number of body function args: 2->%d expected, %d->%d found",
            sensors.size(),
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
      Grid<SensingVoxel> body = Grid.create(
          values,
          vs -> (max(vs) >= threshold) ? new SensingVoxel(List.of(SerializationUtils.clone(sensors.get(indexOfmax(vs))))) : null
      );
      if (body.values().stream().noneMatch(Objects::nonNull)) {
        body = Grid.create(1, 1, new SensingVoxel(List.of(SerializationUtils.clone(sensors.get(indexOfmax(values.get(0, 0)))))));
      }
      body = Utils.gridLargestConnected(body, Objects::nonNull);
      body = Utils.cropGrid(body, Objects::nonNull);
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

  private int indexOfmax(double[] vs) {
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
            DistributedSensing.nOfInputs(new SensingVoxel(sensors.subList(0, 1)), signals),
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
    List<Sensor> sensors = voxelPrototype.getSensors();
    if (sensors.isEmpty()) {
      throw new IllegalArgumentException("Target robot has no sensors");
    }
    if (sensors.stream().mapToInt(s -> s.domains().length).distinct().count() != 1) {
      throw new IllegalArgumentException(String.format(
          "Target robot has sensors with different number of outputs: %s",
          sensors.stream().mapToInt(s -> s.domains().length).distinct()
      ));
    }
    return sensors;
  }

}
