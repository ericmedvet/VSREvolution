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
import it.units.erallab.builder.robot.BrainCentralized;
import it.units.erallab.builder.robot.BrainPhaseFunction;
import it.units.erallab.builder.robot.BrainPhaseValues;
import it.units.erallab.builder.solver.*;
import it.units.erallab.hmsrobots.core.controllers.Controller;
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
import it.units.malelab.jgea.core.util.SequentialFunction;
import org.dyn4j.dynamics.Settings;

import java.io.File;
import java.util.*;
import java.util.concurrent.TimeUnit;
import java.util.function.Function;
import java.util.random.RandomGenerator;
import java.util.stream.Collectors;

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
  public static final String SEQUENCE_SEPARATOR_CHAR = ">";
  public static final String SEQUENCE_ITERATION_CHAR = ":";

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

  private static Function<Robot, Outcome> buildTaskFromName(
      String transformationSequenceName,
      String terrainSequenceName,
      double episodeT,
      RandomGenerator random,
      boolean cacheOutcome
  ) {
    //for sequence, assume format '99:name>99:name'
    //transformations
    Function<Robot, Robot> transformation;
    if (transformationSequenceName.contains(SEQUENCE_SEPARATOR_CHAR)) {
      transformation = new SequentialFunction<>(getSequence(transformationSequenceName).entrySet()
          .stream()
          .collect(Collectors.toMap(
              Map.Entry::getKey,
              e -> RobotUtils.buildRobotTransformation(e.getValue(), random)
          )));
    } else {
      transformation = RobotUtils.buildRobotTransformation(transformationSequenceName, random);
    }
    //terrains
    Function<Robot, Outcome> task;
    if (terrainSequenceName.contains(SEQUENCE_SEPARATOR_CHAR)) {
      task = new SequentialFunction<>(getSequence(terrainSequenceName).entrySet()
          .stream()
          .collect(Collectors.toMap(
              Map.Entry::getKey,
              e -> buildLocomotionTask(e.getValue(), episodeT, random, cacheOutcome)
          )));
    } else {
      task = buildLocomotionTask(terrainSequenceName, episodeT, random, cacheOutcome);
    }
    return task.compose(transformation);
  }

  /*
  public static EvolverBuilder<?> getEvolverBuilderFromName(String name) {
    String numGA = "numGA-(?<nPop>\\d+)-(?<diversity>(t|f))-(?<remap>(t|f))";
    String intGA = "intGA-(?<nPop>\\d+)-(?<diversity>(t|f))-(?<remap>(t|f))";
    String numGASpeciated =
        "numGASpec-(?<nPop>\\d+)-(?<nSpecies>\\d+)-(?<criterion>(" + Arrays.stream(DoublesSpeciated
        .SpeciationCriterion.values())
            .map(c -> c.name().toLowerCase(Locale.ROOT))
            .collect(Collectors.joining("|")) + "))-(?<remap>(t|f))";
    String cmaES = "CMAES";
    String eS = "ES-(?<nPop>\\d+)-(?<sigma>\\d+(\\.\\d+)?)-(?<remap>(t|f))";
    Map<String, String> params;
    if ((params = params(numGA, name)) != null) {
      return new DoublesStandard(
          Integer.parseInt(params.get("nPop")),
          (int) Math.max(Math.round((double) Integer.parseInt(params.get("nPop")) / 10d), 3),
          0.75d,
          params.get("diversity").equals("t"),
          params.get("remap").equals("t")
      );
    }
    if ((params = params(intGA, name)) != null) {
      return new IntegersStandard(
          Integer.parseInt(params.get("nPop")),
          (int) Math.max(Math.round((double) Integer.parseInt(params.get("nPop")) / 10d), 3),
          0.75d,
          params.get("diversity").equals("t"),
          params.get("remap").equals("t")
      );
    }
    if ((params = params(numGASpeciated, name)) != null) {
      return new DoublesSpeciated(
          Integer.parseInt(params.get("nPop")),
          Integer.parseInt(params.get("nSpecies")),
          0.75d,
          DoublesSpeciated.SpeciationCriterion.valueOf(params.get("criterion").toUpperCase()),
          params.get("remap").equals("t")
      );
    }
    if ((params = params(eS, name)) != null) {
      return new ES(
          Double.parseDouble(params.get("sigma")),
          Integer.parseInt(params.get("nPop")),
          params.get("remap").equals("t")
      );
    }
    //noinspection UnusedAssignment
    if ((params = params(cmaES, name)) != null) {
      return new CMAES();
    }
    throw new IllegalArgumentException(String.format("Unknown evolver builder name: %s", name));
  }
  */

  /*
  @SuppressWarnings({"unchecked", "rawtypes"})
  private static PrototypedFunctionBuilder<?, ?> getMapperBuilderFromName(String name) {
    String fixedCentralized = "fixedCentralized";
    String fixedHomoDistributed = "fixedHomoDist-(?<nSignals>\\d+)";
    String fixedHeteroDistributed = "fixedHeteroDist-(?<nSignals>\\d+)";
    String fixedPhasesFunction = "fixedPhasesFunct-(?<f>\\d+)";
    String fixedPhases = "fixedPhases-(?<f>\\d+(\\.\\d+)?)";
    String fixedAutoPoses = "fixedAutoPoses-(?<stepT>\\d+(\\.\\d+)?)-(?<nRegions>\\d+)-(?<nUniquePoses>\\d+)-" +
        "(?<nPoses>\\d+)";
    String bodySin = "bodySin-(?<fullness>\\d+(\\.\\d+)?)-(?<minF>\\d+(\\.\\d+)?)-(?<maxF>\\d+(\\.\\d+)?)";
    String bodyAndHomoDistributed = "bodyAndHomoDist-(?<fullness>\\d+(\\.\\d+)?)-(?<nSignals>\\d+)-(?<nLayers>\\d+)";
    String sensorAndBodyAndHomoDistributed = "sensorAndBodyAndHomoDist-(?<fullness>\\d+(\\.\\d+)?)-(?<nSignals>\\d+)" +
        "-" + "(?<nLayers>\\d+)-(?<position>(t|f))";
    String sensorCentralized = "sensorCentralized-(?<nLayers>\\d+)";
    String mlp = "MLP-(?<ratio>\\d+(\\.\\d+)?)-(?<nLayers>\\d+)(-(?<actFun>(sin|tanh|sigmoid|relu)))?";
    String pruningMlp = "pMLP-(?<ratio>\\d+(\\.\\d+)?)-(?<nLayers>\\d+)-(?<actFun>(sin|tanh|sigmoid|relu))-" +
        "(?<pruningTime>\\d+(\\.\\d+)?)-(?<pruningRate>0(\\.\\d+)?)-(?<criterion>(weight|abs_signal_mean|random))";
    String directNumGrid = "directNumGrid";
    String functionNumGrid = "functionNumGrid";
    String fgraph = "fGraph";
    String functionGrid = "fGrid-(?<innerMapper>.*)";
    Map<String, String> params;
    //robot mappers
    //noinspection UnusedAssignment
    if ((params = params(fixedCentralized, name)) != null) {
      return new FixedCentralized();
    }
    if ((params = params(fixedHomoDistributed, name)) != null) {
      return new FixedHomoDistributed(Integer.parseInt(params.get("nSignals")));
    }
    if ((params = params(fixedHeteroDistributed, name)) != null) {
      return new FixedHeteroDistributed(Integer.parseInt(params.get("nSignals")));
    }
    if ((params = params(fixedPhasesFunction, name)) != null) {
      return new FixedPhaseFunction(Double.parseDouble(params.get("f")), 1d);
    }
    if ((params = params(fixedPhases, name)) != null) {
      return new FixedPhaseValues(Double.parseDouble(params.get("f")), 1d);
    }
    if ((params = params(fixedAutoPoses, name)) != null) {
      return new FixedAutoPoses(
          Integer.parseInt(params.get("nUniquePoses")),
          Integer.parseInt(params.get("nPoses")),
          Integer.parseInt(params.get("nRegions")),
          16,
          Double.parseDouble(params.get("stepT"))
      );
    }
    if ((params = params(bodyAndHomoDistributed, name)) != null) {
      return new BodyAndHomoDistributed(
          Integer.parseInt(params.get("nSignals")),
          Double.parseDouble(params.get("fullness"))
      ).compose(PrototypedFunctionBuilder.of(List.of(
          new MLP(2d, 3, MultiLayerPerceptron.ActivationFunction.SIN),
          new MLP(0.65d, Integer.parseInt(params.get("nLayers")))
      ))).compose(PrototypedFunctionBuilder.merger());
    }
    if ((params = params(sensorAndBodyAndHomoDistributed, name)) != null) {
      return new SensorAndBodyAndHomoDistributed(
          Integer.parseInt(params.get("nSignals")),
          Double.parseDouble(params.get("fullness")),
          params.get("position").equals("t")
      ).compose(PrototypedFunctionBuilder.of(List.of(
          new MLP(2d, 3, MultiLayerPerceptron.ActivationFunction.SIN),
          new MLP(1.5d, Integer.parseInt(params.get("nLayers")))
      ))).compose(PrototypedFunctionBuilder.merger());
    }
    if ((params = params(bodySin, name)) != null) {
      return new BodyAndSinusoidal(
          Double.parseDouble(params.get("minF")),
          Double.parseDouble(params.get("maxF")),
          Double.parseDouble(params.get("fullness")),
          Set.of(
              BodyAndSinusoidal.Component.FREQUENCY,
              BodyAndSinusoidal.Component.PHASE,
              BodyAndSinusoidal.Component.AMPLITUDE
          )
      );
    }
    if ((params = params(fixedHomoDistributed, name)) != null) {
      return new FixedHomoDistributed(Integer.parseInt(params.get("nSignals")));
    }
    if ((params = params(sensorCentralized, name)) != null) {
      return new SensorCentralized().compose(PrototypedFunctionBuilder.of(List.of(
          new MLP(
              2d,
              3,
              MultiLayerPerceptron.ActivationFunction.SIN
          ),
          new MLP(1.5d, Integer.parseInt(params.get("nLayers")))
      ))).compose(PrototypedFunctionBuilder.merger());
    }
    //function mappers
    if ((params = params(mlp, name)) != null) {
      return new MLP(
          Double.parseDouble(params.get("ratio")),
          Integer.parseInt(params.get("nLayers")),
          params.containsKey("actFun") ? MultiLayerPerceptron.ActivationFunction.valueOf(params.get("actFun")
              .toUpperCase()) : MultiLayerPerceptron.ActivationFunction.TANH
      );
    }
    if ((params = params(pruningMlp, name)) != null) {
      return new PruningMLP(
          Double.parseDouble(params.get("ratio")),
          Integer.parseInt(params.get("nLayers")),
          MultiLayerPerceptron.ActivationFunction.valueOf(params.get("actFun").toUpperCase()),
          Double.parseDouble(params.get("pruningTime")),
          Double.parseDouble(params.get("pruningRate")),
          PruningMultiLayerPerceptron.Context.NETWORK,
          PruningMultiLayerPerceptron.Criterion.valueOf(params.get("criterion").toUpperCase())

      );
    }
    //noinspection UnusedAssignment
    if ((params = params(fgraph, name)) != null) {
      return new FGraph();
    }
    //misc
    if ((params = params(functionGrid, name)) != null) {
      return new FunctionGrid((PrototypedFunctionBuilder) getMapperBuilderFromName(params.get("innerMapper")));
    }
    //noinspection UnusedAssignment
    if ((params = params(directNumGrid, name)) != null) {
      return new DirectNumbersGrid();
    }
    //noinspection UnusedAssignment
    if ((params = params(functionNumGrid, name)) != null) {
      return new FunctionNumbersGrid();
    }
    throw new IllegalArgumentException(String.format("Unknown mapper name: %s", name));
  }
  */

  public static SortedMap<Long, String> getSequence(String sequenceName) {
    return new TreeMap<>(Arrays.stream(sequenceName.split(SEQUENCE_SEPARATOR_CHAR))
        .collect(Collectors.toMap(
            s -> s.contains(SEQUENCE_ITERATION_CHAR) ? Long.parseLong(s.split(
                SEQUENCE_ITERATION_CHAR)[0]) : 0,
            s -> s.contains(SEQUENCE_ITERATION_CHAR) ? s.split(SEQUENCE_ITERATION_CHAR)[1] : s
        )));
  }

  public static void main(String[] args) {
    new Starter(args);
  }

  public static ValidationOutcome validate(
      Robot robot, String terrainName, String transformationName, int seed, double episodeTime, double transientTime
  ) {
    RandomGenerator random = new Random(seed);
    robot = RobotUtils.buildRobotTransformation(transformationName, random).apply(robot);
    return new ValidationOutcome(terrainName, transformationName, seed, Starter.buildLocomotionTask(
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
    List<String> targetShapeNames = l(a("shape", "worm-4x2"));
    List<String> targetSensorConfigNames = l(a("sensorConfig", "uniform-a-0"));
    List<String> transformationNames = l(a("transformation", "identity"));
    List<String> solverNames = l(a("solver", "numGA-16-t-f"));
    List<String> mapperNames = l(a("mapper", "fixedPhases-1.0"));
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
        Map.entry("brainPhaseFun", new BrainPhaseFunction())
    ));
    //consumers
    List<NamedFunction<? super POSetPopulationState<?, Robot, Outcome>, ?>> basicFunctions =
        NamedFunctions.basicFunctions();
    List<NamedFunction<? super Individual<?, Robot, Outcome>, ?>> basicIndividualFunctions =
        NamedFunctions.individualFunctions(
            fitnessFunction);
    List<NamedFunction<? super POSetPopulationState<?, Robot, Outcome>, ?>> populationFunctions =
        NamedFunctions.populationFunctions(
            fitnessFunction);
    List<NamedFunction<? super POSetPopulationState<?, Robot, Outcome>, ?>> visualFunctions = Misc.concat(List.of(
        NamedFunctions.visualPopulationFunctions(fitnessFunction),
        detailedOutput ? NamedFunctions.best().then(NamedFunctions.visualIndividualFunctions()) : List.of()
    ));
    List<NamedFunction<? super Outcome, ?>> basicOutcomeFunctions = NamedFunctions.basicOutcomeFunctions();
    List<NamedFunction<? super Outcome, ?>> detailedOutcomeFunctions = NamedFunctions.detailedOutcomeFunctions(
        spectrumMinFreq,
        spectrumMaxFreq,
        spectrumSize
    );
    List<NamedFunction<? super Outcome, ?>> visualOutcomeFunctions = detailedOutput ?
        NamedFunctions.visualOutcomeFunctions(
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
          NamedFunctions.best().then(basicIndividualFunctions),
          basicOutcomeFunctions.stream().map(f -> f.of(fitness()).of(NamedFunctions.best())).toList(),
          visualOutcomeFunctions.stream().map(f -> f.of(fitness()).of(NamedFunctions.best())).toList()
      )), NamedFunctions.keysFunctions()));
    }
    //file listeners
    if (lastFileName != null) {
      factories.add(new CSVPrinter<>(Misc.concat(List.of(
          basicFunctions,
          populationFunctions,
          NamedFunctions.best().then(basicIndividualFunctions),
          basicOutcomeFunctions.stream().map(f -> f.of(fitness()).of(NamedFunctions.best())).toList(),
          detailedOutcomeFunctions.stream().map(f -> f.of(fitness()).of(NamedFunctions.best())).toList(),
          NamedFunctions.best().then(NamedFunctions.serializationFunction(serializationFlags.contains("last")))
      )), NamedFunctions.keysFunctions(), new File(lastFileName)).onLast());
    }
    if (bestFileName != null) {
      factories.add(new CSVPrinter<>(Misc.concat(List.of(
          basicFunctions,
          populationFunctions,
          NamedFunctions.best().then(basicIndividualFunctions),
          basicOutcomeFunctions.stream().map(f -> f.of(fitness()).of(NamedFunctions.best())).toList(),
          detailedOutcomeFunctions.stream().map(f -> f.of(fitness()).of(NamedFunctions.best())).toList(),
          NamedFunctions.best().then(NamedFunctions.serializationFunction(serializationFlags.contains("last")))
      )), NamedFunctions.keysFunctions(), new File(bestFileName)));
    }
    if (allFileName != null) {
      List<NamedFunction<? super Pair<POSetPopulationState<?, Robot, Outcome>, Individual<?, Robot, Outcome>>, ?>> functions = new ArrayList<>();
      functions.addAll(NamedFunctions.stateExtractor().then(basicFunctions));
      functions.addAll(NamedFunctions.individualExtractor().then(basicIndividualFunctions));
      functions.addAll(NamedFunctions.individualExtractor()
          .then(NamedFunctions.serializationFunction(serializationFlags.contains("final"))));
      factories.add(new CSVPrinter<>(
          functions,
          NamedFunctions.keysFunctions(),
          new File(allFileName)
      ).forEach(NamedFunctions.populationSplitter()));
    }
    if (finalFileName != null) {
      List<NamedFunction<? super Pair<POSetPopulationState<?, Robot, Outcome>, Individual<?, Robot, Outcome>>, ?>> functions = new ArrayList<>();
      functions.addAll(NamedFunctions.stateExtractor().then(basicFunctions));
      functions.addAll(NamedFunctions.individualExtractor().then(basicIndividualFunctions));
      functions.addAll(NamedFunctions.individualExtractor()
          .then(NamedFunctions.serializationFunction(serializationFlags.contains("final"))));
      factories.add(new CSVPrinter<>(
          functions,
          NamedFunctions.keysFunctions(),
          new File(finalFileName)
      ).forEach(NamedFunctions.populationSplitter()).onLast());
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
      factories.add(new CSVPrinter<>(functions, NamedFunctions.keysFunctions(), new File(validationFileName)).forEach(
          NamedFunctions.best()
              .andThen(NamedFunctions.validation(
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
          NamedFunctions.lastEventToString(fitnessFunction),
          NamedFunctions.fitnessPlot(fitnessFunction),
          NamedFunctions.centerPositionPlot(),
          NamedFunctions.bestVideo(videoEpisodeTransientTime, videoEpisodeTime, PHYSICS_SETTINGS)
      ), telegramBotId, telegramChatId));
      progressMonitor = progressMonitor.and(new TelegramProgressMonitor(telegramBotId, telegramChatId));
    }
    ListenerFactory<? super POSetPopulationState<?, Robot, Outcome>, Map<String, Object>> factory = ListenerFactory.all(
        factories);
    //summarize params
    L.info("Experiment name: " + experimentName);
    L.info("Evolvers: " + solverNames);
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
                  } catch (ClassCastException | IllegalArgumentException e) {
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
                        buildTaskFromName(
                            transformationName,
                            terrainName,
                            episodeTime,
                            random,
                            cacheOutcome
                        ).andThen(o -> o.subOutcome(episodeTransientTime, episodeTime)),
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
