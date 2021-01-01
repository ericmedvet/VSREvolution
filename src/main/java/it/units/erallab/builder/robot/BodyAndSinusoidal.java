package it.units.erallab.builder.robot;

import it.units.erallab.RealFunction;
import it.units.erallab.builder.PrototypedFunctionBuilder;
import it.units.erallab.builder.phenotype.MLP;
import it.units.erallab.hmsrobots.core.controllers.MultiLayerPerceptron;
import it.units.erallab.hmsrobots.core.controllers.TimeFunctions;
import it.units.erallab.hmsrobots.core.objects.ControllableVoxel;
import it.units.erallab.hmsrobots.core.objects.Robot;
import it.units.erallab.hmsrobots.core.objects.SensingVoxel;
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
public class BodyAndSinusoidal implements PrototypedFunctionBuilder<RealFunction, Robot<?>> {

  private final double minF;
  private final double maxF;
  private final double percentile;

  public BodyAndSinusoidal(double minF, double maxF, double percentile) {
    this.minF = minF;
    this.maxF = maxF;
    this.percentile = percentile;
  }

  @Override
  public Function<RealFunction, Robot<?>> buildFor(Robot<?> robot) {
    int w = robot.getVoxels().getW();
    int h = robot.getVoxels().getH();
    ControllableVoxel voxelPrototype = robot.getVoxels().values().stream().filter(Objects::nonNull).findFirst().orElse(null);
    if (voxelPrototype == null) {
      throw new IllegalArgumentException("Target robot has no valid voxels");
    }
    //build body
    return function -> {
      //check function sizes
      if (function.getNOfInputs() != 2 || function.getNOfOutputs() != 4) {
        throw new IllegalArgumentException(String.format(
            "Wrong number of body function args: 2->4 expected, %d->%d found",
            function.getNOfInputs(),
            function.getNOfOutputs()
        ));
      }
      //build body
      Grid<Double> values = Grid.create(w, h, (x, y) -> function.apply(new double[]{(double) x / (double) w, (double) y / (double) h})[0]);
      double threshold = new Percentile().evaluate(values.values().stream().mapToDouble(v -> v).toArray(), percentile);
      values = Grid.create(values, v -> v >= threshold ? v : null);
      values = Utils.gridLargestConnected(values, Objects::nonNull);
      values = Utils.cropGrid(values, Objects::nonNull);
      Grid<ControllableVoxel> body = Grid.create(values, v -> (v != null) ? SerializationUtils.clone(voxelPrototype) : null);
      if (body.values().stream().noneMatch(Objects::nonNull)) {
        body = Grid.create(1, 1, SerializationUtils.clone(voxelPrototype));
      }
      //build controller
      int bodyW = body.getW();
      int bodyH = body.getH();
      TimeFunctions controller = new TimeFunctions(Grid.create(
          body.getW(),
          body.getH(),
          (x, y) -> {
            double[] vs = function.apply(new double[]{(double) x / (double) bodyW, (double) y / (double) bodyH});
            double freq = minF + (maxF - minF) * (clip(vs[1]) + 1d) / 2d;
            double phase = Math.PI * (clip(vs[2]) + 1d) / 2d;
            double amplitude = (clip(vs[2]) + 1d) / 2d;
            System.out.printf("%d,%d -> f=%.3f p=%.3f a=%.3f%n", x, y, freq, phase, amplitude);
            return t -> amplitude * Math.sin(2 * Math.PI * freq * t + phase);
          }
      ));
      return new Robot<>(controller, body);
    };
  }

  private static double clip(double value) {
    return Math.min(Math.max(value, -1), 1);
  }

  @Override
  public RealFunction exampleFor(Robot<?> robot) {
    ControllableVoxel prototypeVoxel = robot.getVoxels().values().stream().filter(Objects::nonNull).findFirst().orElse(null);
    if (prototypeVoxel == null) {
      throw new IllegalArgumentException("Target robot has no valid voxels");
    }
    return RealFunction.from(2, 4, d -> d);
  }

  public static void main(String[] args) {
    Robot<? extends SensingVoxel> target = new Robot<>(
        null,
        RobotUtils.buildSensorizingFunction("uniform-t-0").apply(RobotUtils.buildShape("box-5x5"))
    );
    PrototypedFunctionBuilder<List<Double>, Robot<?>> builder = new BodyAndSinusoidal(0.1, 1, 50)
        .compose(new MLP(2d, 3, MultiLayerPerceptron.ActivationFunction.SIN));
    Factory<List<Double>> factory = new FixedLengthListFactory<>(builder.exampleFor(target).size(), new UniformDoubleFactory(-1d, 1d));
    Random random = new Random();
    Locomotion locomotion = new Locomotion(30, Locomotion.createTerrain("flat"), new Settings());
    GridOnlineViewer.run(
        locomotion,
        factory.build(9, random).stream().map(vs -> builder.buildFor(target).apply(vs)).collect(Collectors.toList())
    );
  }
}
