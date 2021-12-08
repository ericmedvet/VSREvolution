package it.units.erallab.builder.devofunction;

import it.units.erallab.builder.PrototypedFunctionBuilder;
import it.units.erallab.hmsrobots.core.controllers.AbstractController;
import it.units.erallab.hmsrobots.core.controllers.Controller;
import it.units.erallab.hmsrobots.core.controllers.TimeFunctions;
import it.units.erallab.hmsrobots.core.objects.ControllableVoxel;
import it.units.erallab.hmsrobots.core.objects.Robot;
import it.units.erallab.hmsrobots.tasks.devolocomotion.DistanceBasedDevoLocomotion;
import it.units.erallab.hmsrobots.tasks.locomotion.Locomotion;
import it.units.erallab.hmsrobots.util.Grid;
import it.units.erallab.hmsrobots.util.RobotUtils;
import it.units.erallab.hmsrobots.util.SerializationUtils;
import it.units.erallab.hmsrobots.util.Utils;
import org.dyn4j.dynamics.Settings;

import java.io.IOException;
import java.util.Objects;
import java.util.function.Function;
import java.util.function.UnaryOperator;

/**
 * @author "Eric Medvet" on 2021/09/29 for VSREvolution
 */
public class DevoPhasesValues implements PrototypedFunctionBuilder<Grid<double[]>, UnaryOperator<Robot<?>>> {

  private final double frequency;
  private final double amplitude;
  private final int nInitial;
  private final int nStep;
  private final double controllerStep;

  public DevoPhasesValues(double frequency, double amplitude, int nInitial, int nStep, double controllerStep) {
    this.frequency = frequency;
    this.amplitude = amplitude;
    this.nInitial = nInitial;
    this.nStep = nStep;
    this.controllerStep = controllerStep;
  }

  @Override
  public Function<Grid<double[]>, UnaryOperator<Robot<?>>> buildFor(UnaryOperator<Robot<?>> robotUnaryOperator) {
    Robot<?> target = robotUnaryOperator.apply(null);
    ControllableVoxel voxelPrototype = target.getVoxels().values().stream().filter(Objects::nonNull).findFirst().orElse(null);
    if (voxelPrototype == null) {
      throw new IllegalArgumentException("Target robot has no valid voxels");
    }
    int targetW = target.getVoxels().getW();
    int targetH = target.getVoxels().getH();
    return grid -> {
      //check grid sizes
      if (grid.getW() != targetW || grid.getH() != targetH) {
        throw new IllegalArgumentException(String.format(
            "Wrong grid size: %dx%d expected, %dx%d found",
            targetW, targetH,
            grid.getW(), grid.getH()
        ));
      }
      //check grid element size
      if (grid.values().stream().anyMatch(v -> v.length != 2)) {
        Grid.Entry<double[]> firstWrong = grid.stream().filter(e -> e.getValue().length != 2).findFirst().orElse(null);
        if (firstWrong == null) {
          throw new IllegalArgumentException("Unexpected empty wrong grid item");
        }
        throw new IllegalArgumentException(String.format(
            "Wrong number of values in grid at %d,%d: %d expected, %d found",
            firstWrong.getX(), firstWrong.getY(),
            2,
            firstWrong.getValue().length
        ));
      }
      Grid<Double> strengths = Grid.create(targetW, targetH, (x, y) -> grid.get(x, y)[0]);
      Grid<Double> phases = Grid.create(targetW, targetH, (x, y) -> grid.get(x, y)[1]);
      return previous -> {
        int n;
        if (previous == null) {
          n = nInitial;
        } else {
          n = (int) previous.getVoxels().values().stream().filter(Objects::nonNull).count() + nStep;
        }

        Grid<Double> selected = Utils.gridConnected(strengths, Double::compareTo, n);
        Grid<ControllableVoxel> body = Grid.create(selected, v -> (v != null) ? SerializationUtils.clone(voxelPrototype) : null);
        if (body.values().stream().noneMatch(Objects::nonNull)) {
          body = Grid.create(1, 1, SerializationUtils.clone(voxelPrototype));
        }
        //build controller
        double localAmplitude = amplitude; // copy needed to enable lambdas serialization
        double localFrequency = frequency;
        AbstractController<?> controller = new TimeFunctions(Grid.create(
            body.getW(),
            body.getH(),
            (x, y) -> t -> localAmplitude * Math.sin(2 * Math.PI * localFrequency * t + phases.get(x, y))
        ));
        if (controllerStep > 0) {
          controller =controller.step(controllerStep);
        }
        return new Robot<>((Controller<ControllableVoxel>) controller, body);
      };
    };
  }

  @Override
  public Grid<double[]> exampleFor(UnaryOperator<Robot<?>> robotUnaryOperator) {
    Robot<?> target = robotUnaryOperator.apply(null);
    int targetW = target.getVoxels().getW();
    int targetH = target.getVoxels().getH();
    return Grid.create(targetW, targetH, new double[]{0d, 0d});
  }

  public static void main(String[] args) throws IOException {
    UnaryOperator<Robot<?>> devoComb = youngerRobot -> {
      int w = youngerRobot == null ? 2 : youngerRobot.getVoxels().getW() + 1;
      double f = -0.75d;
      Grid<Boolean> body = Grid.create(w, 2, (x, y) -> y == 1 || (x % 2 == 0));
      return new Robot<>(
          new TimeFunctions(Grid.create(
              body.getW(),
              body.getH(),
              (final Integer x, final Integer y) -> (Double t) -> Math.sin(
                  -2 * Math.PI * f * t + 2 * Math.PI * ((double) x / (double) body.getW()) + Math.PI * ((double) y / (double) body.getH())
              )
          )),
          RobotUtils.buildSensorizingFunction("uniform-a-0.01").apply(body)
      );
    };
    UnaryOperator<Robot<?>> target = r -> new Robot<>(
        Controller.empty(),
        RobotUtils.buildSensorizingFunction("uniform-a-0.01").apply(RobotUtils.buildShape("box-10x10"))
    );
    UnaryOperator<Robot<?>> devoPhases = new DevoPhasesValues(1, 1, 5, 1, 0d)
        .buildFor(target)
        .apply(Grid.create(10, 10, (x, y) -> new double[]{x + y, x * y / 10d}));
    DistanceBasedDevoLocomotion devoLocomotion = new DistanceBasedDevoLocomotion(
        10, 20, 60,
        Locomotion.createTerrain("downhill-30"),
        new Settings()
    );
    System.out.println(devoLocomotion.apply(devoPhases));
    //GridOnlineViewer.run(devoLocomotion, devoPhases);
    /*System.out.println(devoLocomotion.apply(devoFunction));
    GridFileWriter.save(
        devoLocomotion,
        Grid.create(1, 1, Pair.of("devo",devoFunction)),
        800, 400, 0, 24,
        VideoUtils.EncoderFacility.FFMPEG_SMALL,
        new File("/home/eric/devo-comb.mp4"),
        Drawers::basicWithMiniWorldAndFootprintsAndPosture
    );*/
  }

}
