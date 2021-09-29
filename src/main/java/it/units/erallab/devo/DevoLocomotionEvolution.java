package it.units.erallab.devo;

import com.google.common.base.Stopwatch;
import it.units.erallab.LocomotionEvolution;
import it.units.erallab.builder.DirectNumbersGrid;
import it.units.erallab.builder.PrototypedFunctionBuilder;
import it.units.erallab.hmsrobots.core.controllers.Controller;
import it.units.erallab.hmsrobots.core.objects.Robot;
import it.units.erallab.hmsrobots.tasks.locomotion.Locomotion;
import it.units.erallab.hmsrobots.tasks.locomotion.Outcome;
import it.units.erallab.hmsrobots.util.Grid;
import it.units.erallab.hmsrobots.util.RobotUtils;
import it.units.erallab.hmsrobots.util.SerializationUtils;
import it.units.erallab.hmsrobots.viewers.GridFileWriter;
import it.units.erallab.hmsrobots.viewers.VideoUtils;
import it.units.malelab.jgea.Worker;
import it.units.malelab.jgea.core.Individual;
import it.units.malelab.jgea.core.evolver.Event;
import it.units.malelab.jgea.core.evolver.Evolver;
import it.units.malelab.jgea.core.evolver.stopcondition.FitnessEvaluations;
import it.units.malelab.jgea.core.listener.*;
import it.units.malelab.jgea.core.listener.telegram.TelegramUpdater;
import it.units.malelab.jgea.core.order.PartialComparator;
import it.units.malelab.jgea.core.util.ImagePlotters;
import it.units.malelab.jgea.core.util.Misc;
import it.units.malelab.jgea.core.util.TextPlotter;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.*;
import java.util.concurrent.TimeUnit;
import java.util.function.Function;
import java.util.function.UnaryOperator;

import static it.units.erallab.hmsrobots.util.Utils.params;
import static it.units.malelab.jgea.core.listener.NamedFunctions.*;
import static it.units.malelab.jgea.core.util.Args.*;

/**
 * @author "Eric Medvet" on 2021/09/27 for VSREvolution
 */
public class DevoLocomotionEvolution extends Worker {

  public DevoLocomotionEvolution(String[] args) {
    super(args);
  }

  public static void main(String[] args) {
    new DevoLocomotionEvolution(args);
  }

  @Override
  public void run() {
    //main params
    double episodeTime = d(a("episodeTime", "60"));
    double stageMaxTime = d(a("stageMaxTime", "10"));
    double stageMinDistance = d(a("stageMinDistance", "10"));
    double episodeTransientTime = d(a("episodeTransientTime", "1"));
    int nEvals = i(a("nEvals", "1000"));
    int[] seeds = ri(a("seed", "0:1"));
    String experimentName = a("expName", "short");
    int gridW = i(a("gridW", "10"));
    int gridH = i(a("gridH", "10"));
    List<String> terrainNames = l(a("terrain", "flat"));
    List<String> devoFunctionNames = l(a("devoFunction", "devoFixedPhases-1.0-3<directNumGrid"));
    List<String> targetSensorConfigNames = l(a("sensorConfig", "uniform-t+a-0.01"));
    List<String> evolverNames = l(a("evolver", "ES-16-0.35"));
    String lastFileName = a("lastFile", null);
    String bestFileName = a("bestFile", null);
    String allFileName = a("allFile", null);
    boolean deferred = a("deferred", "true").startsWith("t");
    String telegramBotId = a("telegramBotId", null);
    long telegramChatId = Long.parseLong(a("telegramChatId", "0"));
    //fitness function
    Function<List<Outcome>, Double> fitnessFunction = outcomes -> outcomes.stream().mapToDouble(Outcome::getDistance).sum();
    //consumers
    List<NamedFunction<Event<?, ? extends UnaryOperator<Robot<?>>, ? extends List<Outcome>>, ?>> keysFunctions = keyFunctions();
    List<NamedFunction<Event<?, ? extends UnaryOperator<Robot<?>>, ? extends List<Outcome>>, ?>> basicFunctions = basicFunctions();
    List<NamedFunction<Event<?, ? extends UnaryOperator<Robot<?>>, ? extends List<Outcome>>, ?>> populationFunctions = List.of();
    List<NamedFunction<Individual<?, ? extends UnaryOperator<Robot<?>>, ? extends List<Outcome>>, ?>> basicIndividualFunctions = basicIndividualFunctions(fitnessFunction);
    List<NamedFunction<List<Outcome>, ?>> outcomesFunctions = List.of();
    List<NamedFunction<Individual<?, ? extends UnaryOperator<Robot<?>>, ? extends List<Outcome>>, ?>> visualIndividualFunctions = List.of();
    Listener.Factory<Event<?, ? extends UnaryOperator<Robot<?>>, ? extends List<Outcome>>> factory = Listener.Factory.deaf();
    NamedFunction<Event<?, ? extends UnaryOperator<Robot<?>>, ? extends List<Outcome>>, List<Outcome>> bestFitness = f("best.fitness", event -> Misc.first(event.getOrderedPopulation().firsts()).getFitness());
    //screen listener
    if (bestFileName == null) {
      factory = factory.and(new TabularPrinter<>(Misc.concat(List.of(
          basicFunctions,
          populationFunctions,
          NamedFunction.then(best(), basicIndividualFunctions),
          NamedFunction.then(best(), visualIndividualFunctions),
          NamedFunction.then(bestFitness, outcomesFunctions)
      ))));
    }
    if (telegramBotId != null && telegramChatId != 0) {
      factory = factory.and(new TelegramUpdater<>(List.of(
          fitnessPlot(fitnessFunction),
          bestVideo(stageMinDistance, stageMaxTime, episodeTime)
      ), telegramBotId, telegramChatId));
    }
    //summarize params
    L.info("Experiment name: " + experimentName);
    L.info("Evolvers: " + evolverNames);
    L.info("Devo functions: " + devoFunctionNames);
    L.info("Sensor configs: " + targetSensorConfigNames);
    L.info("Terrains: " + terrainNames);
    //start iterations
    int nOfRuns = seeds.length * terrainNames.size() * devoFunctionNames.size() * evolverNames.size();
    int counter = 0;
    for (int seed : seeds) {
      for (String terrainName : terrainNames) {
        for (String devoFunctionName : devoFunctionNames) {
          for (String targetSensorConfigName : targetSensorConfigNames) {
            for (String evolverName : evolverNames) {
              counter = counter + 1;
              final Random random = new Random(seed);
              //prepare keys
              Map<String, Object> keys = Map.ofEntries(
                  Map.entry("experiment.name", experimentName),
                  Map.entry("seed", seed),
                  Map.entry("terrain", terrainName),
                  Map.entry("devo.function", devoFunctionName),
                  Map.entry("sensor.config", targetSensorConfigName),
                  Map.entry("evolver", evolverName)
              );
              //prepare target
              UnaryOperator<Robot<?>> target = r -> new Robot<>(
                  Controller.empty(),
                  RobotUtils.buildSensorizingFunction(targetSensorConfigName).apply(RobotUtils.buildShape("box-" + gridW + "x" + gridH))
              );
              //build evolver
              Evolver<?, UnaryOperator<Robot<?>>, List<Outcome>> evolver;
              try {
                evolver = buildEvolver(evolverName, devoFunctionName, target, fitnessFunction);
              } catch (ClassCastException | IllegalArgumentException e) {
                L.warning(String.format(
                    "Cannot instantiate %s for %s: %s",
                    evolverName,
                    devoFunctionName,
                    e
                ));
                continue;
              }
              Listener<Event<?, ? extends UnaryOperator<Robot<?>>, ? extends List<Outcome>>> listener = Listener.all(List.of(
                  new EventAugmenter(keys),
                  factory.build()
              ));
              if (deferred) {
                listener = listener.deferred(executorService);
              }
              //optimize
              Stopwatch stopwatch = Stopwatch.createStarted();
              L.info(String.format("Progress %s (%d/%d); Starting %s",
                  TextPlotter.horizontalBar(counter - 1, 0, nOfRuns, 8),
                  counter, nOfRuns,
                  keys
              ));
              //build task
              try {
                Collection<UnaryOperator<Robot<?>>> solutions = evolver.solve(
                    buildLocomotionTask(terrainName,stageMinDistance,stageMaxTime,episodeTime,random),
                    new FitnessEvaluations(nEvals),
                    random,
                    executorService,
                    listener
                );
                L.info(String.format("Progress %s (%d/%d); Done: %d solutions in %4ds",
                    TextPlotter.horizontalBar(counter, 0, nOfRuns, 8),
                    counter, nOfRuns,
                    solutions.size(),
                    stopwatch.elapsed(TimeUnit.SECONDS)
                ));
              } catch (Exception e) {
                L.severe(String.format("Cannot complete %s due to %s",
                    keys,
                    e
                ));
                e.printStackTrace(); // TODO possibly to be removed
              }
            }
          }
        }
      }
    }
    factory.shutdown();
  }

  private static List<NamedFunction<Event<?, ? extends UnaryOperator<Robot<?>>, ? extends List<Outcome>>, ?>> keyFunctions() {
    return List.of(
        eventAttribute("experiment.name"),
        eventAttribute("seed", "%2d"),
        eventAttribute("terrain"),
        eventAttribute("devo.function"),
        eventAttribute("evolver")
    );
  }

  private static List<NamedFunction<Event<?, ? extends UnaryOperator<Robot<?>>, ? extends List<Outcome>>, ?>> basicFunctions() {
    return List.of(
        iterations(),
        births(),
        fitnessEvaluations(),
        elapsedSeconds()
    );
  }

  public static Accumulator.Factory<Event<?, ? extends UnaryOperator<Robot<?>>, ? extends List<Outcome>>, BufferedImage> fitnessPlot(Function<List<Outcome>, Double> fitnessFunction) {
    return new TableBuilder<Event<?, ? extends UnaryOperator<Robot<?>>, ? extends List<Outcome>>, Number>(List.of(
        iterations(),
        f("fitness", fitnessFunction).of(fitness()).of(best()),
        min(Double::compare).of(each(f("fitness", fitnessFunction).of(fitness()))).of(all()),
        median(Double::compare).of(each(f("fitness", fitnessFunction).of(fitness()))).of(all())
    )).then(ImagePlotters.xyLines(600, 400));
  }

  public static Accumulator.Factory<Event<?, ? extends UnaryOperator<Robot<?>>, ? extends List<Outcome>>, File> bestVideo(double stageMinDistance, double stageMaxT, double maxT) {
    return Accumulator.Factory.<Event<?, ? extends UnaryOperator<Robot<?>>, ? extends List<Outcome>>>last().then(
        event -> {
          Random random = new Random(0);
          SortedMap<Long, String> terrainSequence = LocomotionEvolution.getSequence((String) event.getAttributes().get("terrain"));
          String terrainName = terrainSequence.get(terrainSequence.lastKey());
          UnaryOperator<Robot<?>> solution = Misc.first(event.getOrderedPopulation().firsts()).getSolution();
          DevoLocomotion devoLocomotion = new DevoLocomotion(
              stageMinDistance, stageMaxT, maxT,
              Locomotion.createTerrain(terrainName.replace("-rnd", "-" + random.nextInt(10000))),
              LocomotionEvolution.PHYSICS_SETTINGS
          );
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

  public static Function<UnaryOperator<Robot<?>>, List<Outcome>> buildLocomotionTask(String terrainName, double stageMinDistance, double stageMaxT, double maxT, Random random) {
    if (!terrainName.contains("-rnd")) {
      return Misc.cached(new DevoLocomotion(
          stageMinDistance, stageMaxT, maxT,
          Locomotion.createTerrain(terrainName),
          LocomotionEvolution.PHYSICS_SETTINGS
      ), LocomotionEvolution.CACHE_SIZE);
    }
    return r -> new DevoLocomotion(
        stageMinDistance, stageMaxT, maxT,
        Locomotion.createTerrain(terrainName.replace("-rnd", "-" + random.nextInt(10000))),
        LocomotionEvolution.PHYSICS_SETTINGS
    ).apply(r);
  }

  @SuppressWarnings({"unchecked", "rawtypes"})
  private static Evolver<?, UnaryOperator<Robot<?>>, List<Outcome>> buildEvolver(String evolverName, String devoFunctionName, UnaryOperator<Robot<?>> target, Function<List<Outcome>, Double> outcomeMeasure) {
    PrototypedFunctionBuilder<?, ?> devoFunctionBuilder = null;
    for (String piece : devoFunctionName.split(LocomotionEvolution.MAPPER_PIPE_CHAR)) {
      if (devoFunctionBuilder == null) {
        devoFunctionBuilder = getDevoFunctionByName(piece);
      } else {
        devoFunctionBuilder = devoFunctionBuilder.compose((PrototypedFunctionBuilder) getDevoFunctionByName(piece));
      }
    }
    return LocomotionEvolution.getEvolverBuilderFromName(evolverName).build(
        (PrototypedFunctionBuilder) devoFunctionBuilder,
        target,
        PartialComparator.from(Double.class).comparing(outcomeMeasure).reversed()
    );
  }

  private static List<NamedFunction<Individual<?, ? extends UnaryOperator<Robot<?>>, ? extends List<Outcome>>, ?>> basicIndividualFunctions(Function<List<Outcome>, Double> fitnessFunction) {
    NamedFunction<Individual<?, ? extends UnaryOperator<Robot<?>>, ? extends List<Outcome>>, ?> size = size().of(genotype());
    NamedFunction<Individual<?, ? extends UnaryOperator<Robot<?>>, ? extends List<Outcome>>, ? extends Grid<?>> firstShape =
        f("shape", (Function<Outcome, Grid<?>>) o -> o.getObservations().get(o.getObservations().firstKey()).getVoxelPolies())
            .of(f("first", (Function<List<Outcome>, Outcome>) l -> l.get(0)))
            .of(fitness());
    NamedFunction<Individual<?, ? extends UnaryOperator<Robot<?>>, ? extends List<Outcome>>, ? extends Grid<?>> lastShape =
        f("shape", (Function<Outcome, Grid<?>>) o -> o.getObservations().get(o.getObservations().firstKey()).getVoxelPolies())
            .of(f("last", (Function<List<Outcome>, Outcome>) l -> l.get(l.size() - 1)))
            .of(fitness());
    return List.of(
        f("w", "%2d", (Function<Grid<?>, Number>) Grid::getW).of(firstShape),
        f("h", "%2d", (Function<Grid<?>, Number>) Grid::getW).of(firstShape),
        f("num.voxel", "%2d", (Function<Grid<?>, Number>) g -> g.count(Objects::nonNull)).of(firstShape),
        f("w", "%2d", (Function<Grid<?>, Number>) Grid::getW).of(lastShape),
        f("h", "%2d", (Function<Grid<?>, Number>) Grid::getW).of(lastShape),
        f("num.voxel", "%2d", (Function<Grid<?>, Number>) g -> g.count(Objects::nonNull)).of(lastShape),
        f("num.stages", "%2d", i -> i.getFitness().size()),
        size.reformat("%5d"),
        genotypeBirthIteration(),
        f("fitness", "%5.1f", fitnessFunction).of(fitness())
    );
  }

  private static Evolver<?, List<Robot<?>>, List<Outcome>> buildEvolver(String evolverName, String devoFunctionName, Robot<?>
      target, Function<Outcome, Double> outcomeMeasure) {
    return null; // TODO fix
  }

  private static PrototypedFunctionBuilder<?, ?> getDevoFunctionByName(String name) {
    String fixedPhases = "devoFixedPhases-(?<f>\\d+(\\.\\d+)?)-(?<n>\\d+)";
    String directNumGrid = "directNumGrid";
    Map<String, String> params;
    //devo functions
    if ((params = params(fixedPhases, name)) != null) {
      return new DevoFixedPhasesValues(
          Double.parseDouble(params.get("f")),
          1d,
          Integer.parseInt(params.get("n"))
      );
    }
    //misc
    if ((params = params(directNumGrid, name)) != null) {
      return new DirectNumbersGrid();
    }
    throw new IllegalArgumentException(String.format("Unknown devo function name: %s", name));
  }

}
