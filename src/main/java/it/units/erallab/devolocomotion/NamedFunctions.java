package it.units.erallab.devolocomotion;

import it.units.erallab.devolocomotion.Starter.ValidationOutcome;
import it.units.erallab.hmsrobots.core.objects.Robot;
import it.units.erallab.hmsrobots.tasks.Task;
import it.units.erallab.hmsrobots.tasks.devolocomotion.DevoOutcome;
import it.units.erallab.hmsrobots.tasks.devolocomotion.DistanceBasedDevoLocomotion;
import it.units.erallab.hmsrobots.tasks.devolocomotion.TimeBasedDevoLocomotion;
import it.units.erallab.hmsrobots.tasks.locomotion.Locomotion;
import it.units.erallab.hmsrobots.util.Grid;
import it.units.erallab.hmsrobots.util.SerializationUtils;
import it.units.erallab.hmsrobots.viewers.GridFileWriter;
import it.units.erallab.hmsrobots.viewers.VideoUtils;
import it.units.malelab.jgea.core.listener.AccumulatorFactory;
import it.units.malelab.jgea.core.listener.NamedFunction;
import it.units.malelab.jgea.core.listener.TableBuilder;
import it.units.malelab.jgea.core.solver.Individual;
import it.units.malelab.jgea.core.solver.state.POSetPopulationState;
import it.units.malelab.jgea.core.util.ImagePlotters;
import it.units.malelab.jgea.core.util.Misc;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.*;
import java.util.function.Function;
import java.util.function.UnaryOperator;
import java.util.logging.Logger;
import java.util.stream.Collectors;

import static it.units.malelab.jgea.core.listener.NamedFunctions.*;

/**
 * @author "Eric Medvet" on 2021/10/03 for VSREvolution
 */
public class NamedFunctions {

  private static final Logger L = Logger.getLogger(NamedFunctions.class.getName());

  private NamedFunctions() {
  }

  public static List<NamedFunction<? super POSetPopulationState<?, UnaryOperator<Robot>, DevoOutcome>, ?>> basicFunctions() {
    return List.of(iterations(), births(), fitnessEvaluations(), elapsedSeconds());
  }

  public static List<NamedFunction<? super Individual<?, UnaryOperator<Robot>, DevoOutcome>, ?>> basicIndividualFunctions(
      Function<DevoOutcome, Double> fitnessFunction
  ) {
    NamedFunction<Individual<?, UnaryOperator<Robot>, DevoOutcome>, ?> size = size().of(genotype());
    NamedFunction<Individual<?, UnaryOperator<Robot>, DevoOutcome>, ? extends Grid<?>> firstShape = f("shape",
        (Function<Robot, Grid<?>>) Robot::getVoxels
    ).of(f(
        "first",
        (Function<DevoOutcome, Robot>) l -> l.getRobots().get(0)
    )).of(fitness());
    NamedFunction<Individual<?, UnaryOperator<Robot>, DevoOutcome>, ? extends Grid<?>> lastShape = f("shape",
        (Function<Robot, Grid<?>>) Robot::getVoxels
    ).of(f(
        "last",
        (Function<DevoOutcome, Robot>) l -> l.getRobots().get(l.getRobots().size() - 1)
    )).of(fitness());
    return List.of(f("num.voxel", "%2d", (Function<Grid<?>, Number>) g -> g.count(Objects::nonNull)).of(firstShape),
        f("num.voxel", "%2d", (Function<Grid<?>, Number>) g -> g.count(Objects::nonNull)).of(lastShape),
        f("num.stages", "%2d", i -> i.fitness().getRobots().size()),
        size.reformat("%5d"),
        genotypeBirthIteration(),
        f("fitness", "%5.1f", fitnessFunction).of(fitness())
    );
  }

  public static NamedFunction<POSetPopulationState<?, UnaryOperator<Robot>, DevoOutcome>, Individual<?,
      UnaryOperator<Robot>, DevoOutcome>> best() {
    return ((NamedFunction<POSetPopulationState<?, UnaryOperator<Robot>, DevoOutcome>, Individual<?,
        UnaryOperator<Robot>, DevoOutcome>>) state -> Misc.first(
        state.getPopulation().firsts())).rename("best");
  }

  public static AccumulatorFactory<POSetPopulationState<?, UnaryOperator<Robot>, DevoOutcome>, File, Map<String,
      Object>> bestVideo(
      double stageMinDistance,
      double stageMaxT,
      List<Double> developmentSchedule,
      double maxT,
      boolean distanceBasedDevelopment
  ) {
    return AccumulatorFactory.last((state, keys) -> {
      Random random = new Random(0);
      String terrainName = keys.get("terrain").toString();
      UnaryOperator<Robot> solution = Misc.first(state.getPopulation().firsts()).solution();
      Task<UnaryOperator<Robot>, DevoOutcome> devoLocomotion;
      if (distanceBasedDevelopment) {
        devoLocomotion = new DistanceBasedDevoLocomotion(stageMinDistance,
            stageMaxT,
            maxT,
            Locomotion.createTerrain(terrainName.replace("-rnd", "-" + random.nextInt(10000))),
            Starter.PHYSICS_SETTINGS
        );
      } else {
        devoLocomotion = new TimeBasedDevoLocomotion(developmentSchedule,
            maxT,
            Locomotion.createTerrain(terrainName.replace("-rnd", "-" + random.nextInt(10000))),
            Starter.PHYSICS_SETTINGS
        );
      }
      File file;
      try {
        file = File.createTempFile("robot-video", ".mp4");
        GridFileWriter.save(devoLocomotion, solution, 300, 200, 0, 25, VideoUtils.EncoderFacility.JCODEC, file);
        file.deleteOnExit();
      } catch (IOException ioException) {
        L.warning(String.format("Cannot save video of best: %s", ioException));
        return null;
      }
      return file;
    });
  }

  public static AccumulatorFactory<POSetPopulationState<?, UnaryOperator<Robot>, DevoOutcome>, BufferedImage,
      Map<String, Object>> fitnessPlot(
      Function<DevoOutcome, Double> fitnessFunction
  ) {
    return new TableBuilder<POSetPopulationState<?, UnaryOperator<Robot>, DevoOutcome>, Number, Map<String, Object>>(List.of(
        iterations(),
        f("fitness", fitnessFunction).of(fitness()).of(best()),
        min(Double::compare).of(each(f("fitness", fitnessFunction).of(fitness()))).of(all()),
        median(Double::compare).of(each(f("fitness", fitnessFunction).of(fitness()))).of(all())
        ),
        List.of()
    ).then(t -> ImagePlotters.xyLines(600, 400).apply(t));
  }

  public static List<NamedFunction<? super Map<String, Object>, ?>> keysFunctions() {
    return List.of(attribute("experiment.name"),
        attribute("fitness"),
        attribute("seed").reformat("%2d"),
        attribute("terrain"),
        attribute("devo.function"),
        attribute("solver"),
        attribute("episode.time"),
        attribute("stage.max.time"),
        attribute("stage.min.dist"),
        attribute("development.schedule"),
        attribute("development.criterion")
    );
  }

  public static AccumulatorFactory<POSetPopulationState<?, UnaryOperator<Robot>, DevoOutcome>, String, Map<String,
      Object>> lastEventToString(
      Function<DevoOutcome, Double> fitnessFunction
  ) {
    final List<NamedFunction<? super POSetPopulationState<?, UnaryOperator<Robot>, DevoOutcome>, ?>> functions =
        Misc.concat(
        List.of(basicFunctions(),
            populationFunctions(fitnessFunction),
            best().then(basicIndividualFunctions(fitnessFunction)),
            outcomesFunctions(false).stream().map(f -> f.of(fitness()).of(best())).toList()
        ));
    List<NamedFunction<? super Map<String, Object>, ?>> keysFunctions = keysFunctions();
    return AccumulatorFactory.last((state, keys) -> {
      String s = keysFunctions.stream()
          .map(f -> String.format(f.getName() + ": " + f.getFormat(), f.apply(keys)))
          .collect(Collectors.joining("\n"));
      s = s + functions.stream().map(f -> String.format(f.getName() + ": " + f.getFormat(), f.apply(state))).collect(
          Collectors.joining("\n"));
      return s;
    });
  }

  public static List<NamedFunction<? super DevoOutcome, ?>> outcomesFunctions(boolean serialize) {
    List<NamedFunction<? super DevoOutcome, ?>> functions = new ArrayList<>();
    functions.add(f("speed.average",
        "%4.1f",
        o -> o.getVelocities().stream().filter(v -> !v.isNaN()).mapToDouble(d -> d).average().orElse(Double.NaN)
    ));
    functions.add(f("speed.min",
        "%4.1f",
        o -> o.getVelocities().stream().filter(v -> !v.isNaN()).mapToDouble(d -> d).min().orElse(Double.NaN)
    ));
    functions.add(f("speed.max",
        "%4.1f",
        o -> o.getVelocities().stream().filter(v -> !v.isNaN()).mapToDouble(d -> d).max().orElse(Double.NaN)
    ));
    functions.add(f("time", "%2.1f", o -> o.getTimes().stream().mapToDouble(d -> d).sum()));
    if (serialize) {
      functions.addAll(serializedOutcomesInformation());
    }
    return functions;
  }

  public static List<NamedFunction<? super POSetPopulationState<?, UnaryOperator<Robot>, DevoOutcome>, ?>> populationFunctions(
      Function<DevoOutcome, Double> fitnessFunction
  ) {
    NamedFunction<? super POSetPopulationState<?, UnaryOperator<Robot>, DevoOutcome>, ?> min = min(Double::compare).of(
        each(f("fitness", fitnessFunction).of(fitness()))).of(all());
    NamedFunction<? super POSetPopulationState<?, UnaryOperator<Robot>, DevoOutcome>, ?> median =
        median(Double::compare).of(
        each(f("fitness", fitnessFunction).of(fitness()))).of(all());
    return List.of(size().of(all()),
        size().of(firsts()),
        size().of(lasts()),
        uniqueness().of(each(genotype())).of(all()),
        uniqueness().of(each(solution())).of(all()),
        uniqueness().of(each(fitness())).of(all()),
        min.reformat("%+4.1f"),
        median.reformat("%5.1f")
    );
  }

  public static List<NamedFunction<? super DevoOutcome, ?>> serializedOutcomesInformation() {
    return List.of(f("outcomes.speeds",
        o -> o.getVelocities().stream().map(v -> Double.toString(v)).collect(Collectors.joining(","))
    ), f("devo.robots",
        o -> o.getRobots()
            .stream()
            .map(r -> SerializationUtils.serialize(r, SerializationUtils.Mode.GZIPPED_JSON))
            .collect(Collectors.joining(","))
    ));
  }

  public static Function<? super Individual<?, UnaryOperator<Robot>, DevoOutcome>, Collection<ValidationOutcome>> validation(
      List<String> terrainNames,
      List<Integer> seeds,
      double stageMinDistance,
      double stageMaxT,
      List<Double> developmentSchedule,
      double maxT,
      boolean distanceBasedDevelopment
  ) {
    return i -> {
      List<ValidationOutcome> outcomes = new ArrayList<>();
      for (String terrainName : terrainNames) {
        for (int seed : seeds) {
          outcomes.add(Starter.validate(i.solution(),
              terrainName,
              seed,
              stageMinDistance,
              stageMaxT,
              developmentSchedule,
              maxT,
              distanceBasedDevelopment
          ));
        }
      }
      return outcomes;
    };
  }

  public static List<NamedFunction<? super Individual<?, UnaryOperator<Robot>, DevoOutcome>, ?>> visualIndividualFunctions() {
    return List.of();
  }

}
