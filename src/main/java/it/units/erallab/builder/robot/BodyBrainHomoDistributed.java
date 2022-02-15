package it.units.erallab.builder.robot;

import it.units.erallab.builder.NamedProvider;
import it.units.erallab.builder.PrototypedFunctionBuilder;
import it.units.erallab.hmsrobots.core.controllers.DistributedSensing;
import it.units.erallab.hmsrobots.core.controllers.RealFunction;
import it.units.erallab.hmsrobots.core.controllers.TimedRealFunction;
import it.units.erallab.hmsrobots.core.objects.Robot;
import it.units.erallab.hmsrobots.core.objects.Voxel;
import it.units.erallab.hmsrobots.util.Grid;
import it.units.erallab.hmsrobots.util.SerializationUtils;
import it.units.erallab.hmsrobots.util.Utils;
import org.apache.commons.math3.stat.descriptive.rank.Percentile;

import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.function.Function;

/**
 * @author eric
 */
public class BodyBrainHomoDistributed implements NamedProvider<PrototypedFunctionBuilder<List<TimedRealFunction>,
    Robot>> {
  @Override
  public PrototypedFunctionBuilder<List<TimedRealFunction>, Robot> build(Map<String, String> params) {
    int signals = Integer.parseInt(params.getOrDefault("s", "1"));
    double percentile = Double.parseDouble(params.getOrDefault("p", "0.5"));
    return new PrototypedFunctionBuilder<>() {
      @Override
      public Function<List<TimedRealFunction>, Robot> buildFor(Robot robot) {
        int w = robot.getVoxels().getW();
        int h = robot.getVoxels().getH();
        Voxel voxelPrototype = robot.getVoxels().values().stream().filter(Objects::nonNull).findFirst().orElse(null);
        if (voxelPrototype == null) {
          throw new IllegalArgumentException("Target robot has no voxels");
        }
        int nOfInputs = DistributedSensing.nOfInputs(voxelPrototype, signals);
        int nOfOutputs = DistributedSensing.nOfOutputs(voxelPrototype, signals);
        //build body
        return pair -> {
          if (pair.size() != 2) {
            throw new IllegalArgumentException(String.format(
                "Wrong number of functions: 2 expected, %d found",
                pair.size()
            ));
          }
          TimedRealFunction bodyFunction = pair.get(0);
          TimedRealFunction brainFunction = pair.get(1);
          //check function sizes
          if (bodyFunction.getInputDimension() != 2 || bodyFunction.getOutputDimension() != 1) {
            throw new IllegalArgumentException(String.format(
                "Wrong number of body function args: 2->1 expected, %d->%d found",
                bodyFunction.getInputDimension(),
                bodyFunction.getOutputDimension()
            ));
          }
          if (brainFunction.getInputDimension() != nOfInputs) {
            throw new IllegalArgumentException(String.format(
                "Wrong number of brain function input args: %d expected, %d found",
                nOfInputs,
                brainFunction.getInputDimension()
            ));
          }
          if (brainFunction.getOutputDimension() != nOfOutputs) {
            throw new IllegalArgumentException(String.format(
                "Wrong number of brain function output args: %d expected, %d found",
                nOfOutputs,
                brainFunction.getOutputDimension()
            ));
          }
          //build body
          Grid<Double> values = Grid.create(
              w,
              h,
              (x, y) -> bodyFunction.apply(
                  0d,
                  new double[]{(double) x / ((double) w - 1d), (double) y / ((double) h - 1d)}
              )[0]
          );
          double threshold = new Percentile().evaluate(
              values.values().stream().mapToDouble(v -> v).toArray(),
              percentile
          );
          values = Grid.create(values, v -> v >= threshold ? v : null);
          values = Utils.gridLargestConnected(values, Objects::nonNull);
          values = Utils.cropGrid(values, Objects::nonNull);
          Grid<Voxel> body = Grid.create(values, v -> (v != null) ? SerializationUtils.clone(voxelPrototype) : null);
          if (body.values().stream().noneMatch(Objects::nonNull)) {
            body = Grid.create(1, 1, SerializationUtils.clone(voxelPrototype));
          }
          //build brain
          DistributedSensing controller = new DistributedSensing(body, signals);
          for (Grid.Entry<Voxel> entry : body) {
            if (entry.value() != null) {
              controller.getFunctions().set(entry.key().x(), entry.key().y(), SerializationUtils.clone(brainFunction));
            }
          }
          return new Robot(controller, body);
        };
      }

      @Override
      public List<TimedRealFunction> exampleFor(Robot robot) {
        Voxel voxelPrototype = robot.getVoxels().values().stream().filter(Objects::nonNull).findFirst().orElse(null);
        if (voxelPrototype == null) {
          throw new IllegalArgumentException("Target robot has no voxels");
        }
        return List.of(RealFunction.build(d -> d, 2, 1), RealFunction.build(
            d -> d,
            DistributedSensing.nOfInputs(voxelPrototype, signals),
            DistributedSensing.nOfOutputs(voxelPrototype, signals)
        ));
      }
    };
  }

}
