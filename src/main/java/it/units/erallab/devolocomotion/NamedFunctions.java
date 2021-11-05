package it.units.erallab.devolocomotion;

import it.units.erallab.devolocomotion.Starter.DevoValidationOutcome;
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
import it.units.erallab.locomotion.Starter;
import it.units.malelab.jgea.core.Individual;
import it.units.malelab.jgea.core.evolver.Event;
import it.units.malelab.jgea.core.listener.Accumulator;
import it.units.malelab.jgea.core.listener.NamedFunction;
import it.units.malelab.jgea.core.listener.TableBuilder;
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

  public static List<NamedFunction<Individual<?, ? extends UnaryOperator<Robot<?>>, ? extends DevoOutcome>, ?>> basicIndividualFunctions(Function<DevoOutcome, Double> fitnessFunction) {
    NamedFunction<Individual<?, ? extends UnaryOperator<Robot<?>>, ? extends DevoOutcome>, ?> size = size().of(genotype());
    NamedFunction<Individual<?, ? extends UnaryOperator<Robot<?>>, ? extends DevoOutcome>, ? extends Grid<?>> firstShape =
        f("shape", (Function<Robot<?>, Grid<?>>) Robot::getVoxels)
            .of(f("first", (Function<DevoOutcome, Robot<?>>) l -> l.getRobots().get(0)))
            .of(fitness());
    NamedFunction<Individual<?, ? extends UnaryOperator<Robot<?>>, ? extends DevoOutcome>, ? extends Grid<?>> lastShape =
        f("shape", (Function<Robot<?>, Grid<?>>) Robot::getVoxels)
            .of(f("last", (Function<DevoOutcome, Robot<?>>) l -> l.getRobots().get(l.getRobots().size() - 1)))
            .of(fitness());
    return List.of(
        f("num.voxel", "%2d", (Function<Grid<?>, Number>) g -> g.count(Objects::nonNull)).of(firstShape),
        f("num.voxel", "%2d", (Function<Grid<?>, Number>) g -> g.count(Objects::nonNull)).of(lastShape),
        f("num.stages", "%2d", i -> i.getFitness().getRobots().size()),
        size.reformat("%5d"),
        genotypeBirthIteration(),
        f("fitness", "%5.1f", fitnessFunction).of(fitness())
    );
  }

  public static List<NamedFunction<Event<?, ? extends UnaryOperator<Robot<?>>, ? extends DevoOutcome>, ?>> keysFunctions() {
    return List.of(
        eventAttribute("experiment.name"),
        eventAttribute("fitness"),
        eventAttribute("seed", "%2d"),
        eventAttribute("terrain"),
        eventAttribute("devo.function"),
        eventAttribute("evolver"),
        eventAttribute("episode.time"),
        eventAttribute("stage.max.time"),
        eventAttribute("stage.min.dist"),
        eventAttribute("development.schedule"),
        eventAttribute("development.criterion")
    );
  }

  public static List<NamedFunction<Event<?, ? extends UnaryOperator<Robot<?>>, ? extends DevoOutcome>, ?>> basicFunctions() {
    return List.of(
        iterations(),
        births(),
        fitnessEvaluations(),
        elapsedSeconds()
    );
  }

  public static List<NamedFunction<Event<?, ? extends UnaryOperator<Robot<?>>, ? extends DevoOutcome>, ?>> populationFunctions(Function<DevoOutcome, Double> fitnessFunction) {
    NamedFunction<Event<?, ? extends UnaryOperator<Robot<?>>, ? extends DevoOutcome>, ?> min = min(Double::compare).of(each(f("fitness", fitnessFunction).of(fitness()))).of(all());
    NamedFunction<Event<?, ? extends UnaryOperator<Robot<?>>, ? extends DevoOutcome>, ?> median = median(Double::compare).of(each(f("fitness", fitnessFunction).of(fitness()))).of(all());
    return List.of(
        size().of(all()),
        size().of(firsts()),
        size().of(lasts()),
        uniqueness().of(each(genotype())).of(all()),
        uniqueness().of(each(solution())).of(all()),
        uniqueness().of(each(fitness())).of(all()),
        min.reformat("%+4.1f"),
        median.reformat("%5.1f")
    );
  }

  public static List<NamedFunction<DevoOutcome, ?>> outcomesFunctions() {
    return List.of(
        f("speed.average", "%4.1f", o -> o.getVelocities().stream()
            .filter(v -> !v.isNaN())
            .mapToDouble(d -> d)
            .average().orElse(Double.NaN)),
        f("speed.min", "%4.1f", o -> o.getVelocities().stream()
            .filter(v -> !v.isNaN())
            .mapToDouble(d -> d)
            .min().orElse(Double.NaN)),
        f("speed.max", "%4.1f", o -> o.getVelocities().stream()
            .filter(v -> !v.isNaN())
            .mapToDouble(d -> d)
            .max().orElse(Double.NaN)),
        f("time", "%2.1f", o -> o.getTimes().stream().mapToDouble(d -> d).sum())
    );
  }

  public static List<NamedFunction<DevoOutcome, ?>> serializedOutcomesInformation() {
    return List.of(
        f("outcomes.speeds",
            o -> o.getVelocities().stream().map(v -> Double.toString(v))
                .collect(Collectors.joining(","))),
        f("devo.robots",
            o -> o.getRobots().stream().map(r -> SerializationUtils.serialize(r, SerializationUtils.Mode.GZIPPED_JSON))
                .collect(Collectors.joining(",")))
    );
  }

  public static Accumulator.Factory<Event<?, ? extends UnaryOperator<Robot<?>>, ? extends DevoOutcome>, String> lastEventToString(Function<DevoOutcome, Double> fitnessFunction) {
    NamedFunction<Event<?, ? extends UnaryOperator<Robot<?>>, ? extends DevoOutcome>, DevoOutcome> bestFitness = f("best.fitness", event -> Misc.first(event.getOrderedPopulation().firsts()).getFitness());
    final List<NamedFunction<Event<?, ? extends UnaryOperator<Robot<?>>, ? extends DevoOutcome>, ?>> functions = Misc.concat(List.of(
        keysFunctions(),
        basicFunctions(),
        populationFunctions(fitnessFunction),
        NamedFunction.then(best(), basicIndividualFunctions(fitnessFunction)),
        NamedFunction.then(bestFitness, outcomesFunctions())
    ));
    return Accumulator.Factory.<Event<?, ? extends UnaryOperator<Robot<?>>, ? extends DevoOutcome>>last().then(
        e -> functions.stream()
            .map(f -> f.getName() + ": " + f.applyAndFormat(e))
            .collect(Collectors.joining("\n"))
    );
  }

  public static Accumulator.Factory<Event<?, ? extends UnaryOperator<Robot<?>>, ? extends DevoOutcome>, BufferedImage> fitnessPlot(Function<DevoOutcome, Double> fitnessFunction) {
    return new TableBuilder<Event<?, ? extends UnaryOperator<Robot<?>>, ? extends DevoOutcome>, Number>(List.of(
        iterations(),
        f("fitness", fitnessFunction).of(fitness()).of(best()),
        min(Double::compare).of(each(f("fitness", fitnessFunction).of(fitness()))).of(all()),
        median(Double::compare).of(each(f("fitness", fitnessFunction).of(fitness()))).of(all())
    )).then(ImagePlotters.xyLines(600, 400));
  }

  public static Accumulator.Factory<Event<?, ? extends UnaryOperator<Robot<?>>, ? extends DevoOutcome>, File> bestVideo(
      double stageMinDistance, double stageMaxT, List<Double> developmentSchedule, double maxT, boolean distanceBasedDevelopment) {
    return Accumulator.Factory.<Event<?, ? extends UnaryOperator<Robot<?>>, ? extends DevoOutcome>>last().then(
        event -> {
          Random random = new Random(0);
          SortedMap<Long, String> terrainSequence = Starter.getSequence((String) event.getAttributes().get("terrain"));
          String terrainName = terrainSequence.get(terrainSequence.lastKey());
          UnaryOperator<Robot<?>> solution = Misc.first(event.getOrderedPopulation().firsts()).getSolution();
          Task<UnaryOperator<Robot<?>>, ? extends DevoOutcome> devoLocomotion;
          if (distanceBasedDevelopment) {
            devoLocomotion = new DistanceBasedDevoLocomotion(
                stageMinDistance, stageMaxT, maxT,
                Locomotion.createTerrain(terrainName.replace("-rnd", "-" + random.nextInt(10000))),
                Starter.PHYSICS_SETTINGS
            );
          } else {
            devoLocomotion = new TimeBasedDevoLocomotion(
                developmentSchedule, maxT,
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
        }
    );
  }

  public static List<NamedFunction<Individual<?, ? extends UnaryOperator<Robot<?>>, ? extends DevoOutcome>, ?>> visualIndividualFunctions() {
    return List.of();
  }

  public static Function<Event<?, ? extends UnaryOperator<Robot<?>>, ? extends DevoOutcome>, Collection<DevoValidationOutcome>> validation(
      List<String> validationTerrainNames, List<Integer> seeds,
      double stageMinDistance, double stageMaxT, List<Double> developmentSchedule, double maxT, boolean distanceBasedDevelopment) {
    return event -> {
      UnaryOperator<Robot<?>> solution = Misc.first(event.getOrderedPopulation().firsts()).getSolution();
      List<DevoValidationOutcome> devoValidationOutcomes = new ArrayList<>();
      for (String validationTerrainName : validationTerrainNames) {
        for (int seed : seeds) {
          Random random = new Random(seed);
          Task<UnaryOperator<Robot<?>>, ? extends DevoOutcome> devoLocomotion;
          if (distanceBasedDevelopment) {
            devoLocomotion = new DistanceBasedDevoLocomotion(
                stageMinDistance, stageMaxT, maxT,
                Locomotion.createTerrain(validationTerrainName.replace("-rnd", "-" + random.nextInt(10000))),
                Starter.PHYSICS_SETTINGS
            );
          } else {
            devoLocomotion = new TimeBasedDevoLocomotion(
                developmentSchedule, maxT,
                Locomotion.createTerrain(validationTerrainName.replace("-rnd", "-" + random.nextInt(10000))),
                Starter.PHYSICS_SETTINGS
            );
          }

          DevoOutcome outcomes = devoLocomotion.apply(solution);
          devoValidationOutcomes.add(new DevoValidationOutcome(
              event,
              Map.ofEntries(
                  Map.entry("validation.terrain", validationTerrainName),
                  Map.entry("validation.seed", seed)
              ),
              outcomes
          ));
        }
      }
      return devoValidationOutcomes;
    };
  }

}
