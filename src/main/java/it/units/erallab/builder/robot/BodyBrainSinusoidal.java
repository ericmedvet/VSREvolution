package it.units.erallab.builder.robot;

import it.units.erallab.builder.NamedProvider;
import it.units.erallab.builder.PrototypedFunctionBuilder;
import it.units.erallab.hmsrobots.core.controllers.TimeFunctions;
import it.units.erallab.hmsrobots.core.objects.Robot;
import it.units.erallab.hmsrobots.core.objects.Voxel;
import it.units.erallab.hmsrobots.util.Grid;
import it.units.erallab.hmsrobots.util.SerializationUtils;
import it.units.erallab.hmsrobots.util.Utils;
import org.apache.commons.math3.stat.descriptive.rank.Percentile;

import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.function.Function;

/**
 * @author eric
 */
public class BodyBrainSinusoidal implements NamedProvider<PrototypedFunctionBuilder<Grid<double[]>, Robot>> {

  private final Set<Component> components;

  public BodyBrainSinusoidal(Set<Component> components) {
    this.components = components;
  }

  public enum Component {FREQUENCY, AMPLITUDE, PHASE}

  private static double clip(double value) {
    return Math.min(Math.max(value, -1), 1);
  }

  @Override
  public PrototypedFunctionBuilder<Grid<double[]>, Robot> build(Map<String, String> params) {
    double minF = Double.parseDouble(params.getOrDefault("minF", "0.5"));
    double maxF = Double.parseDouble(params.getOrDefault("maxF", "2"));
    double percentile = Double.parseDouble(params.getOrDefault("p", "0.5"));
    return new PrototypedFunctionBuilder<>() {
      @Override
      public Function<Grid<double[]>, Robot> buildFor(Robot robot) {
        Voxel voxelPrototype = robot.getVoxels().values().stream().filter(Objects::nonNull).findFirst().orElse(null);
        if (voxelPrototype == null) {
          throw new IllegalArgumentException("Target robot has no valid voxels");
        }
        //build body
        return grid -> {
          //check grid sizes
          if (grid.getW() != robot.getVoxels().getW() || grid.getH() != robot.getVoxels().getH()) {
            throw new IllegalArgumentException(String.format(
                "Wrong grid size: %dx%d expected, %dx%d found",
                robot.getVoxels().getW(), robot.getVoxels().getH(),
                grid.getW(), grid.getH()
            ));
          }
          //check grid element size
          if (grid.values().stream().anyMatch(v -> v.length != 1 + components.size())) {
            Grid.Entry<double[]> firstWrong = grid.stream()
                .filter(e -> e.value().length != 1 + components.size())
                .findFirst()
                .orElse(null);
            if (firstWrong == null) {
              throw new IllegalArgumentException("Unexpected empty wrong grid item");
            }
            throw new IllegalArgumentException(String.format(
                "Wrong number of values in grid at %d,%d: %d expected, %d found",
                firstWrong.key().x(), firstWrong.key().y(),
                1 + components.size(),
                firstWrong.value().length
            ));
          }
          //build body
          double threshold = percentile > 0 ? new Percentile().evaluate(grid.values()
              .stream()
              .mapToDouble(v -> v[0])
              .toArray(), percentile) : 0d;
          Grid<double[]> cropped = Utils.cropGrid(Utils.gridLargestConnected(
              Grid.create(grid, v -> v[0] >= threshold ? v : null),
              Objects::nonNull
          ), Objects::nonNull);
          Grid<Voxel> body = Grid.create(cropped, v -> (v != null) ? SerializationUtils.clone(voxelPrototype) : null);
          if (body.values().stream().noneMatch(Objects::nonNull)) {
            body = Grid.create(1, 1, SerializationUtils.clone(voxelPrototype));
          }
          //build controller
          TimeFunctions controller = new TimeFunctions(Grid.create(
              body.getW(),
              body.getH(),
              (x, y) -> {
                int c = 1;
                double freq;
                double phase;
                double amplitude;
                double[] vs = cropped.get(x, y);
                if (vs == null) {
                  vs = new double[1 + Component.values().length];
                }
                if (components.contains(Component.FREQUENCY)) {
                  freq = minF + (maxF - minF) * (clip(vs[c]) + 1d) / 2d;
                  c = c + 1;
                } else {
                  freq = (minF + maxF) / 2d;
                }
                if (components.contains(Component.PHASE)) {
                  phase = Math.PI * (clip(vs[c]) + 1d) / 2d;
                  c = c + 1;
                } else {
                  phase = 0d;
                }
                if (components.contains(Component.AMPLITUDE)) {
                  amplitude = (clip(vs[c]) + 1d) / 2d;
                  c = c + 1;
                } else {
                  amplitude = 1d;
                }
                return t -> amplitude * Math.sin(2 * Math.PI * freq * t + phase);
              }
          ));
          return new Robot(controller, body);
        };
      }

      @Override
      public Grid<double[]> exampleFor(Robot robot) {
        Voxel prototypeVoxel = robot.getVoxels().values().stream().filter(Objects::nonNull).findFirst().orElse(null);
        if (prototypeVoxel == null) {
          throw new IllegalArgumentException("Target robot has no valid voxels");
        }
        return Grid.create(robot.getVoxels().getW(), robot.getVoxels().getH(), new double[1 + components.size()]);
      }
    };
  }

}
