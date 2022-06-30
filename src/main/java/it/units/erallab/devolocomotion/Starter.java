package it.units.erallab.devolocomotion;

import com.google.common.base.Stopwatch;
import it.units.erallab.builder.NamedProvider;
import it.units.erallab.builder.PrototypedFunctionBuilder;
import it.units.erallab.builder.devofunction.*;
import it.units.erallab.builder.misc.DirectNumbersGrid;
import it.units.erallab.builder.robot.BrainPhaseValues;
import it.units.erallab.builder.solver.*;
import it.units.erallab.hmsrobots.core.controllers.Controller;
import it.units.erallab.hmsrobots.core.objects.Robot;
import it.units.erallab.hmsrobots.core.objects.Voxel;
import it.units.erallab.hmsrobots.tasks.Task;
import it.units.erallab.hmsrobots.tasks.devolocomotion.DevoOutcome;
import it.units.erallab.hmsrobots.tasks.devolocomotion.DistanceBasedDevoLocomotion;
import it.units.erallab.hmsrobots.tasks.devolocomotion.TimeBasedDevoLocomotion;
import it.units.erallab.hmsrobots.tasks.locomotion.Locomotion;
import it.units.erallab.hmsrobots.util.RobotUtils;
import it.units.malelab.jgea.Worker;
import it.units.malelab.jgea.core.TotalOrderQualityBasedProblem;
import it.units.malelab.jgea.core.listener.*;
import it.units.malelab.jgea.core.listener.telegram.TelegramProgressMonitor;
import it.units.malelab.jgea.core.listener.telegram.TelegramUpdater;
import it.units.malelab.jgea.core.solver.Individual;
import it.units.malelab.jgea.core.solver.IterativeSolver;
import it.units.malelab.jgea.core.solver.state.POSetPopulationState;
import it.units.malelab.jgea.core.util.Misc;
import org.dyn4j.dynamics.Settings;

import java.io.File;
import java.util.*;
import java.util.concurrent.TimeUnit;
import java.util.function.Function;
import java.util.function.UnaryOperator;
import java.util.random.RandomGenerator;
import java.util.stream.Collectors;

import static it.units.erallab.devolocomotion.NamedFunctions.*;
import static it.units.erallab.hmsrobots.util.Utils.params;
import static it.units.malelab.jgea.core.listener.NamedFunctions.f;
import static it.units.malelab.jgea.core.listener.NamedFunctions.fitness;
import static it.units.malelab.jgea.core.util.Args.*;

/**
 * @author "Eric Medvet" on 2021/09/27 for VSREvolution
 */
public class Starter extends Worker {

  public final static Settings PHYSICS_SETTINGS = new Settings();
  public static final int CACHE_SIZE = 1000;
  public static final String MAPPER_PIPE_CHAR = "<";

  public Starter(String[] args) {
    super(args);
  }

  public record Problem(
      Function<UnaryOperator<Robot>, DevoOutcome> qualityFunction, Comparator<DevoOutcome> totalOrderComparator
  ) implements TotalOrderQualityBasedProblem<UnaryOperator<Robot>, DevoOutcome> {
  }

  public record ValidationOutcome(String terrainName, int seed, DevoOutcome outcome) {
  }

  public static Function<UnaryOperator<Robot>, DevoOutcome> buildDevoLocomotionTask(
      String terrainName,
      double stageMinDistance,
      double stageMaxT,
      List<Double> developmentSchedule,
      double maxT,
      boolean distanceBasedDevelopment,
      RandomGenerator random
  ) {
    if (!terrainName.contains("-rnd")) {
      Task<UnaryOperator<Robot>, DevoOutcome> devoLocomotion;
      if (distanceBasedDevelopment) {
        devoLocomotion = new DistanceBasedDevoLocomotion(
            stageMinDistance,
            stageMaxT,
            maxT,
            Locomotion.createTerrain(terrainName),
            PHYSICS_SETTINGS
        );
      } else {
        devoLocomotion = new TimeBasedDevoLocomotion(
            developmentSchedule,
            maxT,
            Locomotion.createTerrain(terrainName),
            PHYSICS_SETTINGS
        );
      }
      return Misc.cached(devoLocomotion, CACHE_SIZE);
    }
    return r -> {
      Task<UnaryOperator<Robot>, DevoOutcome> devoLocomotion;
      if (distanceBasedDevelopment) {
        devoLocomotion = new DistanceBasedDevoLocomotion(
            stageMinDistance,
            stageMaxT,
            maxT,
            Locomotion.createTerrain(terrainName.replace("-rnd", "-" + random.nextInt(10000))),
            it.units.erallab.locomotion.Starter.PHYSICS_SETTINGS
        );
      } else {
        devoLocomotion = new TimeBasedDevoLocomotion(
            developmentSchedule,
            maxT,
            Locomotion.createTerrain(terrainName.replace("-rnd", "-" + random.nextInt(10000))),
            it.units.erallab.locomotion.Starter.PHYSICS_SETTINGS
        );
      }
      return devoLocomotion.apply(r);
    };
  }

  @SuppressWarnings({"unchecked", "rawtypes"})
  private static IterativeSolver<? extends POSetPopulationState<?, UnaryOperator<Robot>, DevoOutcome>,
      TotalOrderQualityBasedProblem<UnaryOperator<Robot>, DevoOutcome>, UnaryOperator<Robot>> buildSolver(
      String solverName,
      String devoFunctionMapperName,
      UnaryOperator<Robot> target,
      NamedProvider<SolverBuilder<?>> solverBuilderProvider,
      NamedProvider<PrototypedFunctionBuilder<?, ?>> mapperBuilderProvider
  ) {
    PrototypedFunctionBuilder<?, ?> mapperBuilder = null;
    for (String piece : devoFunctionMapperName.split(MAPPER_PIPE_CHAR)) {
      if (mapperBuilder == null) {
        mapperBuilder = mapperBuilderProvider.build(piece).orElseThrow();
      } else {
        mapperBuilder = mapperBuilder.compose((PrototypedFunctionBuilder) mapperBuilderProvider.build(piece)
            .orElseThrow());
      }
    }
    SolverBuilder<Object> solverBuilder = (SolverBuilder<Object>) solverBuilderProvider.build(solverName).orElseThrow();
    return solverBuilder.build((PrototypedFunctionBuilder<Object, UnaryOperator<Robot>>) mapperBuilder, target);
  }

  private static Function<DevoOutcome, Double> getFitnessFunctionFromName(String name) {
    String distance = "distance";
    String maxSpeed = "maxSpeed";
    if (params(distance, name) != null) {
      return devoOutcome -> devoOutcome.getDistances().stream().mapToDouble(d -> d).sum();
    }
    if (params(maxSpeed, name) != null) {
      return devoOutcome -> devoOutcome.getVelocities()
          .stream()
          .filter(v -> !v.isNaN())
          .max(Double::compare)
          .orElse(0d);
    }
    throw new IllegalArgumentException(String.format("Unknown fitness function name: %s", name));
  }

  public static void main(String[] args) {
    new Starter(args);
  }

  public static ValidationOutcome validate(
      UnaryOperator<Robot> devoFunction,
      String terrainName,
      int seed,
      double stageMinDistance,
      double stageMaxT,
      List<Double> developmentSchedule,
      double maxT,
      boolean distanceBasedDevelopment
  ) {
    RandomGenerator random = new Random(seed);
    return new ValidationOutcome(
        terrainName,
        seed,
        buildDevoLocomotionTask(
            terrainName,
            stageMinDistance,
            stageMaxT,
            developmentSchedule,
            maxT,
            distanceBasedDevelopment,
            random
        ).apply(devoFunction)
    );
  }

  @Override
  public void run() {
    //main params
    String developmentCriterion = a("devoCriterion", "distance");
    boolean distanceBasedDevelopment = developmentCriterion.startsWith("d");
    double episodeTime = d(a("episodeTime", "60"));
    double stageMaxTime = d(a("stageMaxTime", "20"));
    double stageMinDistance = d(a("stageMinDistance", "20"));
    String stringDevelopmentSchedule = a("devoSchedule", "10,20,30,45");
    List<Double> developmentSchedule = l(stringDevelopmentSchedule).stream()
        .map(Double::parseDouble)
        .collect(Collectors.toList());
    double validationEpisodeTime = d(a("validationEpisodeTime", Double.toString(episodeTime)));
    double validationStageMaxTime = d(a("validationStageMaxTime", Double.toString(stageMaxTime)));
    double validationStageMinDistance = d(a("validationStageMinDistance", Double.toString(stageMinDistance)));
    List<Double> validationDevelopmentSchedule = l(a("validationDevoSchedule", stringDevelopmentSchedule)).stream().map(
        Double::parseDouble).toList();
    int nEvals = i(a("nEvals", "800"));
    int[] seeds = ri(a("seed", "0:1"));
    String experimentName = a("expName", "short");
    int gridW = i(a("gridW", "5"));
    int gridH = i(a("gridH", "5"));
    List<String> terrainNames = l(a("terrain", "flat"));
    List<String> devoFunctionMapperNames = l(a("devoFunction", "devoPhases-1.0-5-1<directNumGrid"));
    List<String> targetSensorConfigNames = l(a("sensorConfig", "uniform-t+a-0.01"));
    List<String> solverNames = l(a("solver", "ES-16-0.35"));
    String lastFileName = a("lastFile", null);
    String bestFileName = a("bestFile", null);
    String validationFileName = a("validationFile", null);
    boolean deferred = a("deferred", "true").startsWith("t");
    List<String> serializationFlags = l(a("serialization", "")); //last,best,validation
    boolean output = a("output", "false").startsWith("t");
    String telegramBotId = a("telegramBotId", null);
    long telegramChatId = Long.parseLong(a("telegramChatId", "0"));
    List<String> validationTerrainNames = l(a("validationTerrain", "flat,downhill-30")).stream()
        .filter(s -> !s.isEmpty())
        .collect(Collectors.toList());
    //fitness function
    String fitnessFunctionName = a("fitness", "distance");
    Function<DevoOutcome, Double> fitnessFunction = getFitnessFunctionFromName(fitnessFunctionName);
    //providers
    NamedProvider<SolverBuilder<?>> solverBuilderProvider = NamedProvider.of(Map.ofEntries(
        Map.entry("binaryGA", new BitsStandard(0.75, 0.05, 3, 0.01)),
        Map.entry("numGA", new DoublesStandard(0.75, 0.05, 3, 0.35)),
        Map.entry("ES", new SimpleES(0.35, 0.4)),
        Map.entry("treeNumGA", new TreeAndDoubles(0.75, 0.05, 3, 0.01, 3, 6)),
        Map.entry("treePairGA", new TreeAndDoubles(0.75, 0.05, 3, 0.01, 3, 6))
    ));
    NamedProvider<PrototypedFunctionBuilder<?, ?>> mapperBuilderProvider = NamedProvider.of(Map.ofEntries(
        Map.entry("devoHomoMLP", new DevoHomoMLP()),
        Map.entry("devoRndHomoMLP", new DevoRandomHomoMLP()),
        Map.entry("devoRndAddHomoMLP", new DevoRandomAdditionHomoMLP()),
        Map.entry("devoTreeHomoMLP", new DevoTreeHomoMLP()),
        Map.entry("devoTreePhases", new DevoTreePhases()),
        Map.entry("devoCAHomoMLP", new DevoCaMLP()),
        Map.entry("devoCAPhases", new DevoCaPhases())
    ));
    mapperBuilderProvider = mapperBuilderProvider.and(NamedProvider.of(Map.ofEntries(
        Map.entry("devoCondHomoMLP", new DevoConditionedHomoMLP(
            Voxel::getAreaRatioEnergy, true
        )),
        Map.entry("devoCondTreeHomoMLP", new DevoConditionedTreeHomoMLP(
            Voxel::getAreaRatioEnergy, true
        ))
    )));
    mapperBuilderProvider = mapperBuilderProvider.and(NamedProvider.of(Map.ofEntries(
        Map.entry("dirNumGrid", new DirectNumbersGrid()),
        Map.entry("brainPhaseVals", new BrainPhaseValues())
    )));

    //consumers
    List<NamedFunction<? super POSetPopulationState<?, UnaryOperator<Robot>, DevoOutcome>, ?>> basicFunctions =
        basicFunctions();
    List<NamedFunction<? super POSetPopulationState<?, UnaryOperator<Robot>, DevoOutcome>, ?>> populationFunctions =
        populationFunctions(
            fitnessFunction);
    List<NamedFunction<? super Individual<?, UnaryOperator<Robot>, DevoOutcome>, ?>> basicIndividualFunctions =
        basicIndividualFunctions(
            fitnessFunction);
    List<NamedFunction<? super Individual<?, UnaryOperator<Robot>, DevoOutcome>, ?>> visualIndividualFunctions =
        visualIndividualFunctions();
    List<ListenerFactory<? super POSetPopulationState<?, UnaryOperator<Robot>, DevoOutcome>, Map<String, Object>>> factories = new ArrayList<>();
    ProgressMonitor progressMonitor = new ScreenProgressMonitor(System.out);
    //screen listener
    if ((bestFileName == null) || output) {
      factories.add(new TabularPrinter<>(Misc.concat(List.of(
          basicFunctions,
          populationFunctions,
          best().then(basicIndividualFunctions),
          best().then(visualIndividualFunctions),
          outcomesFunctions(false).stream().map(f -> f.of(fitness()).of(best())).toList()
      )), List.of()));
    }
    //file listeners
    if (lastFileName != null) {
      factories.add(new CSVPrinter<>(Misc.concat(List.of(
          basicFunctions,
          populationFunctions,
          best().then(basicIndividualFunctions),
          outcomesFunctions(serializationFlags.contains("last")).stream()
              .map(f -> f.of(fitness()).of(best()))
              .toList()
      )), keysFunctions(), new File(lastFileName)).onLast());
    }
    if (bestFileName != null) {
      factories.add(new CSVPrinter<>(Misc.concat(List.of(
          basicFunctions,
          populationFunctions,
          best().then(basicIndividualFunctions),
          outcomesFunctions(serializationFlags.contains("best")).stream()
              .map(f -> f.of(fitness()).of(best()))
              .toList()
      )), keysFunctions(), new File(bestFileName)));
    }
    //validation listener
    if (validationFileName != null) {
      if (validationTerrainNames.isEmpty()) {
        validationTerrainNames.add(terrainNames.get(0));
      }
      List<NamedFunction<? super ValidationOutcome, ?>> functions = new ArrayList<>();
      functions.add(f("validation.terrain", ValidationOutcome::terrainName));
      functions.add(f("validation.seed", ValidationOutcome::seed));
      functions.addAll(f(
          "validation.outcome",
          ValidationOutcome::outcome
      ).then(outcomesFunctions(serializationFlags.contains("validation"))));
      factories.add(new CSVPrinter<>(functions, keysFunctions(), new File(validationFileName)).forEach(
          best().andThen(validation(
              validationTerrainNames,
              List.of(0),
              validationStageMinDistance,
              validationStageMaxTime,
              validationDevelopmentSchedule,
              validationEpisodeTime,
              distanceBasedDevelopment
          ))).onLast());
    }
    //telegram listener
    if (telegramBotId != null && telegramChatId != 0) {
      factories.add(new TelegramUpdater<>(List.of(
          lastEventToString(fitnessFunction),
          fitnessPlot(fitnessFunction),
          bestVideo(stageMinDistance, stageMaxTime, developmentSchedule, episodeTime, distanceBasedDevelopment)
      ), telegramBotId, telegramChatId));
      progressMonitor = progressMonitor.and(new TelegramProgressMonitor(telegramBotId, telegramChatId));
    }
    ListenerFactory<? super POSetPopulationState<?, UnaryOperator<Robot>, DevoOutcome>, Map<String, Object>> factory
        = ListenerFactory.all(
        factories);
    //summarize params
    L.info("Experiment name: " + experimentName);
    L.info("Solvers: " + solverNames);
    L.info("Devo functions: " + devoFunctionMapperNames);
    L.info("Sensor configs: " + targetSensorConfigNames);
    L.info("Terrains: " + terrainNames);
    L.info("Fitness: " + fitnessFunctionName);
    //start iterations
    int nOfRuns = seeds.length * terrainNames.size() * devoFunctionMapperNames.size() * solverNames.size();
    int counter = 0;
    for (int seed : seeds) {
      for (String terrainName : terrainNames) {
        for (String devoFunctionMapperName : devoFunctionMapperNames) {
          for (String targetSensorConfigName : targetSensorConfigNames) {
            for (String solverName : solverNames) {
              counter = counter + 1;
              final Random random = new Random(seed);
              //prepare keys
              Map<String, Object> keys = Map.ofEntries(
                  Map.entry("experiment.name", experimentName),
                  Map.entry("fitness", fitnessFunctionName),
                  Map.entry("seed", seed),
                  Map.entry("terrain", terrainName),
                  Map.entry("devo.function", devoFunctionMapperName),
                  Map.entry("sensor.config", targetSensorConfigName),
                  Map.entry("solver", solverName),
                  Map.entry("episode.time", episodeTime),
                  Map.entry("stage.max.time", stageMaxTime),
                  Map.entry("stage.min.dist", stageMinDistance),
                  Map.entry("development.schedule", stringDevelopmentSchedule),
                  Map.entry("development.criterion", developmentCriterion)
              );
              //prepare target
              UnaryOperator<Robot> target = r -> new Robot(
                  Controller.empty(),
                  RobotUtils.buildSensorizingFunction(targetSensorConfigName)
                      .apply(RobotUtils.buildShape("box-" + gridW + "x" + gridH))
              );
              //build solver
              IterativeSolver<? extends POSetPopulationState<?, UnaryOperator<Robot>, DevoOutcome>,
                  TotalOrderQualityBasedProblem<UnaryOperator<Robot>, DevoOutcome>, UnaryOperator<Robot>> solver;
              try {
                solver = buildSolver(solverName, devoFunctionMapperName, target, solverBuilderProvider, mapperBuilderProvider);
              } catch (ClassCastException | IllegalArgumentException e) {
                L.warning(String.format("Cannot instantiate %s for %s: %s", solverName, devoFunctionMapperName, e));
                continue;
              }
              //optimize
              Stopwatch stopwatch = Stopwatch.createStarted();
              progressMonitor.notify(
                  ((float) counter - 1) / nOfRuns,
                  String.format("(%d/%d); Starting %s", counter, nOfRuns, keys)
              );
              try {
                Listener<? super POSetPopulationState<?, UnaryOperator<Robot>, DevoOutcome>> listener = factory.build(
                    keys);
                if (deferred) {
                  listener = listener.deferred(executorService);
                }
                Problem problem = new Problem(buildDevoLocomotionTask(terrainName,
                    stageMinDistance,
                    stageMaxTime,
                    developmentSchedule,
                    episodeTime,
                    distanceBasedDevelopment, random
                ), Comparator.comparing(fitnessFunction).reversed());
                Collection<UnaryOperator<Robot>> solutions = solver.solve(problem, random, executorService, listener);
                progressMonitor.notify((float) counter / nOfRuns, String.format(
                    "(%d/%d); Done: %d solutions in %4ds",
                    counter,
                    nOfRuns,
                    solutions.size(),
                    stopwatch.elapsed(TimeUnit.SECONDS)
                ));
              } catch (Exception e) {
                L.severe(String.format("Cannot complete %s due to %s", keys, e));
              }
            }
          }
        }
      }
    }
    factory.shutdown();
  }
}
