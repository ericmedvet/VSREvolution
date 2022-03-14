/*
 * Copyright 2020 Eric Medvet <eric.medvet@gmail.com> (as eric)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package it.units.erallab.locomotion;

import com.google.common.base.Stopwatch;
import com.google.common.collect.Lists;
import it.units.erallab.builder.NamedProvider;
import it.units.erallab.builder.PrototypedFunctionBuilder;
import it.units.erallab.builder.function.FGraph;
import it.units.erallab.builder.function.MLP;
import it.units.erallab.builder.function.PruningMLP;
import it.units.erallab.builder.misc.DirectNumbersGrid;
import it.units.erallab.builder.misc.FunctionNumbersGrid;
import it.units.erallab.builder.misc.FunctionsGrid;
import it.units.erallab.builder.robot.*;
import it.units.erallab.builder.solver.*;
import it.units.erallab.hmsrobots.core.controllers.Controller;
import it.units.erallab.hmsrobots.core.controllers.MultiLayerPerceptron;
import it.units.erallab.hmsrobots.core.controllers.PruningMultiLayerPerceptron;
import it.units.erallab.hmsrobots.core.objects.Robot;
import it.units.erallab.hmsrobots.tasks.locomotion.Locomotion;
import it.units.erallab.hmsrobots.tasks.locomotion.Outcome;
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
import it.units.malelab.jgea.core.util.Pair;
import org.dyn4j.dynamics.Settings;

import java.io.File;
import java.util.*;
import java.util.concurrent.TimeUnit;
import java.util.function.Function;
import java.util.random.RandomGenerator;
import java.util.stream.Collectors;

import static it.units.erallab.locomotion.NamedFunctions.*;
import static it.units.malelab.jgea.core.listener.NamedFunctions.f;
import static it.units.malelab.jgea.core.listener.NamedFunctions.fitness;
import static it.units.malelab.jgea.core.util.Args.*;

/**
 * @author eric
 */
public class Starter extends Worker {

  public final static Settings PHYSICS_SETTINGS = new Settings();
  public static final int CACHE_SIZE = 1000;
  public static final String MAPPER_PIPE_CHAR = "<";

  public Starter(String[] args) {
    super(args);
  }

  public record Problem(
      Function<Robot, Outcome> qualityFunction, Comparator<Outcome> totalOrderComparator
  ) implements TotalOrderQualityBasedProblem<Robot, Outcome> {}

  public record ValidationOutcome(
      String terrainName, String transformationName, int seed, Outcome outcome
  ) {}

  public static Function<Robot, Outcome> buildLocomotionTask(
      String terrainName, double episodeT, RandomGenerator random, boolean cacheOutcome
  ) {
    if (!terrainName.contains("-rnd") && cacheOutcome) {
      return Misc.cached(new Locomotion(episodeT, Locomotion.createTerrain(terrainName), PHYSICS_SETTINGS), CACHE_SIZE);
    }
    return r -> new Locomotion(
        episodeT,
        Locomotion.createTerrain(terrainName.replace("-rnd", "-" + random.nextInt(10000))),
        PHYSICS_SETTINGS
    ).apply(r);
  }

  @SuppressWarnings({"unchecked", "rawtypes"})
  private static IterativeSolver<? extends POSetPopulationState<?, Robot, Outcome>,
      TotalOrderQualityBasedProblem<Robot, Outcome>, Robot> buildSolver(
      String solverName,
      String robotMapperName,
      Robot target,
      NamedProvider<SolverBuilder<?>> solverBuilderProvider,
      NamedProvider<PrototypedFunctionBuilder<?, ?>> mapperBuilderProvider
  ) {
    PrototypedFunctionBuilder<?, ?> mapperBuilder = null;
    for (String piece : robotMapperName.split(MAPPER_PIPE_CHAR)) {
      if (mapperBuilder == null) {
        mapperBuilder = mapperBuilderProvider.build(piece).orElseThrow();
      } else {
        mapperBuilder = mapperBuilder.compose((PrototypedFunctionBuilder) mapperBuilderProvider.build(piece)
            .orElseThrow());
      }
    }
    SolverBuilder<Object> solverBuilder = (SolverBuilder<Object>) solverBuilderProvider.build(solverName).orElseThrow();
    return solverBuilder.build((PrototypedFunctionBuilder<Object, Robot>) mapperBuilder, target);
  }

  public static void main(String[] args) {
    new Starter(args);
  }

  public static ValidationOutcome validate(
      Robot robot, String terrainName, String transformationName, int seed, double episodeTime, double transientTime
  ) {
    RandomGenerator random = new Random(seed);
    robot = RobotUtils.buildRobotTransformation(transformationName, random).apply(robot);
    return new ValidationOutcome(terrainName, transformationName, seed, buildLocomotionTask(
            terrainName,
            episodeTime,
            random,
            false
        )
        .apply(robot)
        .subOutcome(transientTime, episodeTime));
  }

  @Override
  public void run() {
    int spectrumSize = 8;
    double spectrumMinFreq = 0d;
    double spectrumMaxFreq = 4d;
    double episodeTime = d(a("episodeTime", "10"));
    double episodeTransientTime = d(a("episodeTransientTime", "1"));
    double validationEpisodeTime = d(a("validationEpisodeTime", Double.toString(episodeTime)));
    double validationTransientTime = d(a("validationTransientTime", Double.toString(episodeTransientTime)));
    double videoEpisodeTime = d(a("videoEpisodeTime", "10"));
    double videoEpisodeTransientTime = d(a("videoEpisodeTransientTime", "0"));
    int[] seeds = ri(a("seed", "0:1"));
    String experimentName = a("expName", "short");
    List<String> terrainNames = l(a("terrain", "flat"));//"hilly-1-10-rnd"));
    List<String> targetShapeNames = l(a("shape", "biped-4x3"));
    List<String> targetSensorConfigNames = l(a("sensorConfig", "uniform-a-0.01"));
    List<String> transformationNames = l(a("transformation", "identity"));
    List<String> solverNames = l(a("solver", "numGA;nPop=16;nEval=100"));
    List<String> mapperNames = l(a("mapper", "brainPhaseVals;f=0.5"));
    String lastFileName = a("lastFile", null);
    String bestFileName = a("bestFile", null);
    String allFileName = a("allFile", null);
    String finalFileName = a("finalFile", null);
    String validationFileName = a("validationFile", null);
    boolean deferred = a("deferred", "true").startsWith("t");
    String telegramBotId = a("telegramBotId", null);
    long telegramChatId = Long.parseLong(a("telegramChatId", "0"));
    List<String> serializationFlags = l(a("serialization", "")); //last,best,all,final
    boolean output = a("output", "false").startsWith("t");
    boolean detailedOutput = a("detailedOutput", "false").startsWith("t");
    boolean cacheOutcome = a("cache", "false").startsWith("t");
    List<String> validationTransformationNames = l(a("validationTransformation", "")).stream()
        .filter(s -> !s.isEmpty())
        .collect(Collectors.toList());
    List<String> validationTerrainNames = l(a("validationTerrain", "flat,downhill-30")).stream()
        .filter(s -> !s.isEmpty())
        .collect(Collectors.toList());
    Function<Outcome, Double> fitnessFunction = Outcome::getVelocity;
    //providers
    NamedProvider<SolverBuilder<?>> solverBuilderProvider = NamedProvider.of(Map.ofEntries(
        Map.entry("binaryGA", new BitsStandard(0.75, 0.05, 3, 0.01)),
        Map.entry("numGA", new DoublesStandard(0.75, 0.05, 3, 0.35)),
        Map.entry("intGA", new IntegersStandard(0.75, 0.05, 3)),
        Map.entry("ES", new SimpleES(0.35, 0.4)),
        Map.entry("numSpeciated", new DoublesSpeciated(
            0.75,
            0.35,
            0.75,
            (Function<Individual<?, Robot, Outcome>, double[]>) i -> i.fitness()
                .getAveragePosture(8)
                .values()
                .stream()
                .mapToDouble(b -> b ? 1d : 0d)
                .toArray()
        ))
    ));
    NamedProvider<PrototypedFunctionBuilder<?, ?>> mapperBuilderProvider = NamedProvider.of(Map.ofEntries(
        Map.entry("brainCentralized", new BrainCentralized()),
        Map.entry("brainPhaseVals", new BrainPhaseValues()),
        Map.entry("brainPhaseFun", new BrainPhaseFunction()),
        Map.entry("brainHomoDist", new BrainHomoDistributed()),
        Map.entry("brainHeteroDist", new BrainHeteroDistributed()),
        Map.entry("brainAutoPoses", new BrainAutoPoses(16)),
        Map.entry(
            "sensorBrainCentralized",
            new SensorBrainCentralized().then(b -> b.compose(PrototypedFunctionBuilder.of(List.of(
                new MLP().build("r=2;nIL=2").orElseThrow(),
                new MLP().build("r=1.5;nIL=1").orElseThrow()
            )))).then(b -> b.compose(PrototypedFunctionBuilder.merger()))
        ),
        Map.entry("bodyBrainSin", new BodyBrainSinusoidal(EnumSet.of(
            BodyBrainSinusoidal.Component.PHASE,
            BodyBrainSinusoidal.Component.FREQUENCY
        ))),
        Map.entry(
            "bodySensorBrainHomoDist",
            new BodySensorBrainHomoDistributed(false).then(b -> b.compose(PrototypedFunctionBuilder.of(List.of(
                new MLP().build("r=2;nIL=2").orElseThrow(),
                new MLP().build("r=1.5;nIL=1").orElseThrow()
            )))).then(b -> b.compose(PrototypedFunctionBuilder.merger()))
        ),
        Map.entry(
            "bodyBrainHomoDist",
            new BodyBrainHomoDistributed().then(b -> b.compose(PrototypedFunctionBuilder.of(List.of(
                new MLP().build("r=2;nIL=2").orElseThrow(),
                new MLP().build("r=0.65;nIL=1").orElseThrow()
            )))).then(b -> b.compose(PrototypedFunctionBuilder.merger()))
        )
    ));
    mapperBuilderProvider = mapperBuilderProvider.and(NamedProvider.of(Map.ofEntries(
        Map.entry("dirNumGrid", new DirectNumbersGrid()),
        Map.entry("funNumGrid", new FunctionNumbersGrid()),
        Map.entry(
            "funsGrid",
            new FunctionsGrid(new MLP(MultiLayerPerceptron.ActivationFunction.TANH).build(Map.ofEntries(
                Map.entry("r", "0.65"),
                Map.entry("nIL", "1")
            )))
        )
    )));
    mapperBuilderProvider = mapperBuilderProvider.and(NamedProvider.of(Map.ofEntries(
        Map.entry("fGraph", new FGraph()),
        Map.entry("mlp", new MLP(MultiLayerPerceptron.ActivationFunction.TANH)),
        Map.entry(
            "pMlp",
            new PruningMLP(
                MultiLayerPerceptron.ActivationFunction.TANH,
                PruningMultiLayerPerceptron.Context.NETWORK,
                PruningMultiLayerPerceptron.Criterion.ABS_SIGNAL_MEAN
            )
        )
    )));
    //consumers
    List<NamedFunction<? super POSetPopulationState<?, Robot, Outcome>, ?>> basicFunctions = basicFunctions();
    List<NamedFunction<? super Individual<?, Robot, Outcome>, ?>> basicIndividualFunctions =
        individualFunctions(
            fitnessFunction);
    List<NamedFunction<? super POSetPopulationState<?, Robot, Outcome>, ?>> populationFunctions =
        populationFunctions(
            fitnessFunction);
    List<NamedFunction<? super POSetPopulationState<?, Robot, Outcome>, ?>> visualFunctions = Misc.concat(List.of(
        visualPopulationFunctions(fitnessFunction),
        detailedOutput ? best().then(visualIndividualFunctions()) : List.of()
    ));
    List<NamedFunction<? super Outcome, ?>> basicOutcomeFunctions = basicOutcomeFunctions();
    List<NamedFunction<? super Outcome, ?>> detailedOutcomeFunctions = detailedOutcomeFunctions(
        spectrumMinFreq,
        spectrumMaxFreq,
        spectrumSize
    );
    List<NamedFunction<? super Outcome, ?>> visualOutcomeFunctions = detailedOutput ?
        visualOutcomeFunctions(
            spectrumMinFreq,
            spectrumMaxFreq
        ) : List.of();
    List<ListenerFactory<? super POSetPopulationState<?, Robot, Outcome>, Map<String, Object>>> factories =
        new ArrayList<>();
    ProgressMonitor progressMonitor = new ScreenProgressMonitor(System.out);
    //screen listener
    if (bestFileName == null || output) {
      factories.add(new TabularPrinter<>(Misc.concat(List.of(
          basicFunctions,
          populationFunctions,
          visualFunctions,
          best().then(basicIndividualFunctions),
          basicOutcomeFunctions.stream().map(f -> f.of(fitness()).of(best())).toList(),
          visualOutcomeFunctions.stream().map(f -> f.of(fitness()).of(best())).toList()
      )), List.of()));
    }
    //file listeners
    if (lastFileName != null) {
      factories.add(new CSVPrinter<>(Misc.concat(List.of(
          basicFunctions,
          populationFunctions,
          best().then(basicIndividualFunctions),
          basicOutcomeFunctions.stream().map(f -> f.of(fitness()).of(best())).toList(),
          detailedOutcomeFunctions.stream().map(f -> f.of(fitness()).of(best())).toList(),
          best().then(serializationFunction(serializationFlags.contains("last")))
      )), keysFunctions(), new File(lastFileName)).onLast());
    }
    if (bestFileName != null) {
      factories.add(new CSVPrinter<>(Misc.concat(List.of(
          basicFunctions,
          populationFunctions,
          best().then(basicIndividualFunctions),
          basicOutcomeFunctions.stream().map(f -> f.of(fitness()).of(best())).toList(),
          detailedOutcomeFunctions.stream().map(f -> f.of(fitness()).of(best())).toList(),
          best().then(serializationFunction(serializationFlags.contains("last")))
      )), keysFunctions(), new File(bestFileName)));
    }
    if (allFileName != null) {
      List<NamedFunction<? super Pair<POSetPopulationState<?, Robot, Outcome>, Individual<?, Robot, Outcome>>, ?>> functions = new ArrayList<>();
      functions.addAll(stateExtractor().then(basicFunctions));
      functions.addAll(individualExtractor().then(basicIndividualFunctions));
      functions.addAll(individualExtractor()
          .then(serializationFunction(serializationFlags.contains("final"))));
      factories.add(new CSVPrinter<>(
          functions,
          keysFunctions(),
          new File(allFileName)
      ).forEach(populationSplitter()));
    }
    if (finalFileName != null) {
      List<NamedFunction<? super Pair<POSetPopulationState<?, Robot, Outcome>, Individual<?, Robot, Outcome>>, ?>> functions = new ArrayList<>();
      functions.addAll(stateExtractor().then(basicFunctions));
      functions.addAll(individualExtractor().then(basicIndividualFunctions));
      functions.addAll(individualExtractor()
          .then(serializationFunction(serializationFlags.contains("final"))));
      factories.add(new CSVPrinter<>(
          functions,
          keysFunctions(),
          new File(finalFileName)
      ).forEach(populationSplitter()).onLast());
    }
    //validation listener
    if (validationFileName != null) {
      if (!validationTerrainNames.isEmpty() && validationTransformationNames.isEmpty()) {
        validationTransformationNames.add("identity");
      }
      if (validationTerrainNames.isEmpty() && !validationTransformationNames.isEmpty()) {
        validationTerrainNames.add(terrainNames.get(0));
      }
      List<NamedFunction<? super ValidationOutcome, ?>> functions = new ArrayList<>();
      functions.add(f("validation.terrain", ValidationOutcome::terrainName));
      functions.add(f("validation.transformation", ValidationOutcome::transformationName));
      functions.add(f("validation.seed", ValidationOutcome::seed));
      functions.addAll(f("validation.outcome", ValidationOutcome::outcome).then(basicOutcomeFunctions));
      functions.addAll(f("validation.outcome", ValidationOutcome::outcome).then(detailedOutcomeFunctions));
      factories.add(new CSVPrinter<>(functions, keysFunctions(), new File(validationFileName)).forEach(
          best()
              .andThen(validation(
                  validationTerrainNames,
                  validationTransformationNames,
                  List.of(0),
                  validationEpisodeTime,
                  validationTransientTime
              ))).onLast());
    }
    //telegram listener
    if (telegramBotId != null && telegramChatId != 0) {
      factories.add(new TelegramUpdater<>(List.of(
          lastEventToString(fitnessFunction),
          fitnessPlot(fitnessFunction),
          centerPositionPlot(),
          bestVideo(videoEpisodeTransientTime, videoEpisodeTime)
      ), telegramBotId, telegramChatId));
      progressMonitor = progressMonitor.and(new TelegramProgressMonitor(telegramBotId, telegramChatId));
    }
    ListenerFactory<? super POSetPopulationState<?, Robot, Outcome>, Map<String, Object>> factory = ListenerFactory.all(
        factories);
    //summarize params
    L.info("Experiment name: " + experimentName);
    L.info("Solvers: " + solverNames);
    L.info("Mappers: " + mapperNames);
    L.info("Shapes: " + targetShapeNames);
    L.info("Sensor configs: " + targetSensorConfigNames);
    L.info("Terrains: " + terrainNames);
    L.info("Transformations: " + transformationNames);
    L.info("Validations: " + Lists.cartesianProduct(validationTerrainNames, validationTransformationNames));
    //start iterations
    int nOfRuns =
        seeds.length * terrainNames.size() * targetShapeNames.size() * targetSensorConfigNames.size() * mapperNames.size() * transformationNames.size() * solverNames.size();
    int counter = 0;
    for (int seed : seeds) {
      for (String terrainName : terrainNames) {
        for (String targetShapeName : targetShapeNames) {
          for (String targetSensorConfigName : targetSensorConfigNames) {
            for (String mapperName : mapperNames) {
              for (String transformationName : transformationNames) {
                for (String solverName : solverNames) {
                  counter = counter + 1;
                  final RandomGenerator random = new Random(seed);
                  //prepare keys
                  Map<String, Object> keys = Map.ofEntries(
                      Map.entry("experiment.name", experimentName),
                      Map.entry("seed", seed),
                      Map.entry("terrain", terrainName),
                      Map.entry("shape", targetShapeName),
                      Map.entry("sensor.config", targetSensorConfigName),
                      Map.entry("mapper", mapperName),
                      Map.entry("transformation", transformationName),
                      Map.entry("solver", solverName),
                      Map.entry("episode.time", episodeTime),
                      Map.entry("episode.transient.time", episodeTransientTime)
                  );
                  //prepare target
                  Robot target = new Robot(
                      Controller.empty(),
                      RobotUtils.buildSensorizingFunction(targetSensorConfigName)
                          .apply(RobotUtils.buildShape(targetShapeName))
                  );
                  //build evolver
                  IterativeSolver<? extends POSetPopulationState<?, Robot, Outcome>,
                      TotalOrderQualityBasedProblem<Robot, Outcome>, Robot> solver;
                  try {
                    solver = buildSolver(solverName, mapperName, target, solverBuilderProvider, mapperBuilderProvider);
                  } catch (NoSuchElementException e) {
                    L.warning(String.format("Cannot instantiate %s for %s: %s", solverName, mapperName, e));
                    continue;
                  }
                  //optimize
                  Stopwatch stopwatch = Stopwatch.createStarted();
                  progressMonitor.notify(
                      ((float) counter - 1) / nOfRuns,
                      String.format("(%d/%d); Starting %s", counter, nOfRuns, keys)
                  );
                  try {
                    Listener<? super POSetPopulationState<?, Robot, Outcome>> listener = factory.build(keys);
                    if (deferred) {
                      listener = listener.deferred(executorService);
                    }
                    Problem problem = new Problem(
                        RobotUtils.buildRobotTransformation(transformationName, random)
                            .andThen(buildLocomotionTask(terrainName, episodeTime, random, cacheOutcome))
                            .andThen(o -> o.subOutcome(
                                episodeTransientTime,
                                episodeTime
                            )),
                        Comparator.comparing(fitnessFunction).reversed()
                    );
                    Collection<Robot> solutions = solver.solve(problem, random, executorService, listener);
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
      }
    }
    factory.shutdown();
  }

}
