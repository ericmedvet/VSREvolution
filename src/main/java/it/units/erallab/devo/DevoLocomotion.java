package it.units.erallab.devo;

import it.units.erallab.hmsrobots.core.controllers.TimeFunctions;
import it.units.erallab.hmsrobots.core.geometry.BoundingBox;
import it.units.erallab.hmsrobots.core.objects.Ground;
import it.units.erallab.hmsrobots.core.objects.Robot;
import it.units.erallab.hmsrobots.core.objects.WorldObject;
import it.units.erallab.hmsrobots.core.snapshots.SnapshotListener;
import it.units.erallab.hmsrobots.tasks.AbstractTask;
import it.units.erallab.hmsrobots.tasks.locomotion.Locomotion;
import it.units.erallab.hmsrobots.tasks.locomotion.Outcome;
import it.units.erallab.hmsrobots.util.Grid;
import it.units.erallab.hmsrobots.util.RobotUtils;
import it.units.erallab.hmsrobots.viewers.GridFileWriter;
import it.units.erallab.hmsrobots.viewers.GridOnlineViewer;
import it.units.erallab.hmsrobots.viewers.VideoUtils;
import it.units.erallab.hmsrobots.viewers.drawers.Drawers;
import org.apache.commons.lang3.time.StopWatch;
import org.apache.commons.lang3.tuple.Pair;
import org.dyn4j.dynamics.Settings;
import org.dyn4j.dynamics.World;
import org.dyn4j.geometry.Vector2;

import java.io.File;
import java.io.IOException;
import java.util.*;
import java.util.concurrent.TimeUnit;
import java.util.function.UnaryOperator;

/**
 * @author "Eric Medvet" on 2021/09/27 for VSREvolution
 */
public class DevoLocomotion extends AbstractTask<UnaryOperator<Robot<?>>, List<Outcome>> {

  private final double stageMinDistance;
  private final double stageMaxT;
  private final double maxT;
  private final double[][] groundProfile;
  private final double initialPlacement;

  public DevoLocomotion(double stageMinDistance, double stageMaxT, double maxT, double[][] groundProfile, double initialPlacement, Settings settings) {
    super(settings);
    this.stageMinDistance = stageMinDistance;
    this.stageMaxT = stageMaxT;
    this.maxT = maxT;
    this.groundProfile = groundProfile;
    this.initialPlacement = initialPlacement;
  }

  public DevoLocomotion(double stageMinDistance, double stageMaxT, double maxT, double[][] groundProfile, Settings settings) {
    this(stageMinDistance, stageMaxT, maxT, groundProfile, groundProfile[0][1] + Locomotion.INITIAL_PLACEMENT_X_GAP, settings);
  }

  @Override
  public List<Outcome> apply(UnaryOperator<Robot<?>> solution, SnapshotListener listener) {
    StopWatch stopWatch = StopWatch.createStarted();
    //init world
    World world = new World();
    world.setSettings(settings);
    Ground ground = new Ground(groundProfile[0], groundProfile[1]);
    Robot<?> robot = solution.apply(null);
    rebuildWorld(ground, robot, world, initialPlacement - robot.boundingBox().min.x);
    List<WorldObject> worldObjects = List.of(ground, robot);
    //run
    List<Outcome> outcomes = new ArrayList<>();
    Map<Double, Outcome.Observation> observations = new HashMap<>();
    double t = 0d;
    double stageT = t;
    double stageMinX = robot.boundingBox().min.x;
    while (t < maxT) {
      t = AbstractTask.updateWorld(t, settings.getStepFrequency(), world, worldObjects, listener);
      observations.put(t, new Outcome.Observation(
          Grid.create(robot.getVoxels(), v -> v == null ? null : v.getVoxelPoly()),
          ground.yAt(robot.getCenter().x),
          (double) stopWatch.getTime(TimeUnit.MILLISECONDS) / 1000d
      ));
      //check if stage ended
      if (t - stageT > stageMaxT) {
        break;
      }
      //check if develop
      if (robot.boundingBox().min.x - stageMinX > stageMinDistance) {
        stageT = t;
        //develop
        double minX = robot.boundingBox().min.x;
        robot = solution.apply(robot);
        //place
        world.removeAllBodies();
        rebuildWorld(ground, robot, world, minX);
        worldObjects = List.of(ground, robot);
        stageMinX = robot.getCenter().x;
        //save outcome
        outcomes.add(new Outcome(observations));
        observations = new HashMap<>();
      }
    }
    outcomes.add(new Outcome(observations));
    stopWatch.stop();
    //prepare outcome
    return outcomes;
  }

  private void rebuildWorld(Ground ground, Robot<?> robot, World world, double newMinX) {
    ground.addTo(world);
    robot.addTo(world);
    //position robot: translate on x
    robot.translate(new Vector2(newMinX - robot.boundingBox().min.x, 0));
    //translate on y
    double minYGap = robot.getVoxels().values().stream()
        .filter(Objects::nonNull)
        .mapToDouble(v -> v.boundingBox().min.y - ground.yAt(v.getCenter().x))
        .min().orElse(0d);
    robot.translate(new Vector2(0, Locomotion.INITIAL_PLACEMENT_Y_GAP - minYGap));
  }

  public static void main(String[] args) throws IOException {
    UnaryOperator<Robot<?>> devoFunction = youngerRobot -> {
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
    DevoLocomotion devoLocomotion = new DevoLocomotion(
        10, 10, 60,
        Locomotion.createTerrain("downhill-30"),
        new Settings()
    );
    //GridOnlineViewer.run(devoLocomotion, devoFunction);
    System.out.println(devoLocomotion.apply(devoFunction));
    GridFileWriter.save(
        devoLocomotion,
        Grid.create(1, 1, Pair.of("devo",devoFunction)),
        800, 400, 0, 24,
        VideoUtils.EncoderFacility.FFMPEG_SMALL,
        new File("/home/eric/devo-comb.mp4"),
        Drawers::basicWithMiniWorldAndFootprintsAndPosture
    );
  }
}
