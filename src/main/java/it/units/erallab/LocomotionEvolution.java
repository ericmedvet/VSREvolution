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

package it.units.erallab;

import com.google.common.base.Stopwatch;
import com.google.common.collect.Lists;
import it.units.erallab.builder.DirectNumbersGrid;
import it.units.erallab.builder.FunctionGrid;
import it.units.erallab.builder.FunctionNumbersGrid;
import it.units.erallab.builder.PrototypedFunctionBuilder;
import it.units.erallab.builder.evolver.CMAES;
import it.units.erallab.builder.evolver.DoublesSpeciated;
import it.units.erallab.builder.evolver.DoublesStandard;
import it.units.erallab.builder.evolver.EvolverBuilder;
import it.units.erallab.builder.phenotype.FGraph;
import it.units.erallab.builder.phenotype.MLP;
import it.units.erallab.builder.robot.*;
import it.units.erallab.hmsrobots.core.controllers.MultiLayerPerceptron;
import it.units.erallab.hmsrobots.core.objects.Robot;
import it.units.erallab.hmsrobots.tasks.locomotion.Locomotion;
import it.units.erallab.hmsrobots.tasks.locomotion.Outcome;
import it.units.erallab.hmsrobots.util.RobotUtils;
import it.units.erallab.hmsrobots.util.SerializationUtils;
import it.units.malelab.jgea.Worker;
import it.units.malelab.jgea.core.Individual;
import it.units.malelab.jgea.core.evolver.Event;
import it.units.malelab.jgea.core.evolver.Evolver;
import it.units.malelab.jgea.core.evolver.stopcondition.Births;
import it.units.malelab.jgea.core.listener.CSVPrinter;
import it.units.malelab.jgea.core.listener.Listener;
import it.units.malelab.jgea.core.listener.NamedFunction;
import it.units.malelab.jgea.core.listener.TabularPrinter;
import it.units.malelab.jgea.core.listener.telegram.TelegramUpdater;
import it.units.malelab.jgea.core.order.PartialComparator;
import it.units.malelab.jgea.core.util.Misc;
import it.units.malelab.jgea.core.util.Pair;
import it.units.malelab.jgea.core.util.SequentialFunction;
import it.units.malelab.jgea.core.util.TextPlotter;
import org.dyn4j.dynamics.Settings;

import java.io.File;
import java.util.*;
import java.util.concurrent.TimeUnit;
import java.util.function.Function;
import java.util.stream.Collectors;

import static it.units.erallab.hmsrobots.util.Utils.params;
import static it.units.malelab.jgea.core.listener.NamedFunctions.*;
import static it.units.malelab.jgea.core.util.Args.*;

/**
 * @author eric
 */
public class LocomotionEvolution extends Worker {

  private final static Settings PHYSICS_SETTINGS = new Settings();

  public static class ValidationOutcome {
    private final Event<?, ? extends Robot<?>, ? extends Outcome> event;
    private final Map<String, Object> keys;
    private final Outcome outcome;

    public ValidationOutcome(Event<?, ? extends Robot<?>, ? extends Outcome> event, Map<String, Object> keys, Outcome outcome) {
      this.event = event;
      this.keys = keys;
      this.outcome = outcome;
    }
  }

  public static final int CACHE_SIZE = 1000;
  public static final String MAPPER_PIPE_CHAR = "<";
  public static final String SEQUENCE_SEPARATOR_CHAR = ">";
  public static final String SEQUENCE_ITERATION_CHAR = ":";

  public LocomotionEvolution(String[] args) {
    super(args);
  }

  public static void main(String[] args) {
    new LocomotionEvolution(args);
  }

  @Override
  public void run() {
    int spectrumSize = 10;
    double spectrumMinFreq = 0d;
    double spectrumMaxFreq = 5d;
    double episodeTime = d(a("episodeTime", "10"));
    double episodeTransientTime = d(a("episodeTransientTime", "5"));
    double videoEpisodeTime = d(a("videoEpisodeTime", "10"));
    int nBirths = i(a("nBirths", "100"));
    int[] seeds = ri(a("seed", "0:1"));
    String experimentName = a("expName", "short");
    List<String> terrainNames = l(a("terrain", "hilly-1-10-0"));
    List<String> targetShapeNames = l(a("shape", "biped-7x4"));
    List<String> targetSensorConfigNames = l(a("sensorConfig", "spinedTouch-t-f"));
    List<String> transformationNames = l(a("transformation", "identity"));
    List<String> evolverNames = l(a("evolver", "CMAES"));
    List<String> mapperNames = l(a("mapper", "fixedPhases-1"));//bodySin-50-0.1-1<functionNumGrid<MLP-4-4"));//""sensorAndBodyAndHomoDist-50-3-3-t"));
    String bestFileName = a("bestFile", null);
    String allFileName = a("allFile", null);
    String validationFileName = a("validationFile", null);
    String telegramBotId = a("telegramBotId", null);
    long telegramChatId = Long.parseLong(a("telegramChatId", "207490209"));
    boolean serialization = a("serialization", "false").startsWith("t");
    boolean output = a("output", "false").startsWith("t");
    List<String> validationTransformationNames = l(a("validationTransformation", "")).stream().filter(s -> !s.isEmpty()).collect(Collectors.toList());
    List<String> validationTerrainNames = l(a("validationTerrain", "flat,downhill-30")).stream().filter(s -> !s.isEmpty()).collect(Collectors.toList());
    Function<Outcome, Double> fitnessFunction = Outcome::getVelocity;
    //consumers
    Map<String, Object> keys = new HashMap<>();
    List<NamedFunction<Event<?, ? extends Robot<?>, ? extends Outcome>, ?>> keysFunctions = Utils.keysFunctions(keys);
    List<NamedFunction<Event<?, ? extends Robot<?>, ? extends Outcome>, ?>> basicFunctions = Utils.basicFunctions();
    List<NamedFunction<Individual<?, ? extends Robot<?>, ? extends Outcome>, ?>> basicIndividualFunctions = Utils.individualFunctions(fitnessFunction);
    List<NamedFunction<Individual<?, ? extends Robot<?>, ? extends Outcome>, ?>> detailedIndividualFunctions = serialization ? List.of(
        f("serialized", r -> SerializationUtils.serialize(r, SerializationUtils.Mode.GZIPPED_JSON)).of(solution())
    ) : List.of();
    List<NamedFunction<Event<?, ? extends Robot<?>, ? extends Outcome>, ?>> populationFunctions = Utils.populationFunctions();
    List<NamedFunction<Event<?, ? extends Robot<?>, ? extends Outcome>, ?>> visualFunctions = Utils.visualFunctions(fitnessFunction);
    List<NamedFunction<Outcome, ?>> basicOutcomeFunctions = Utils.basicOutcomeFunctions();
    List<NamedFunction<Outcome, ?>> detailedOutcomeFunctions = Utils.detailedOutcomeFunctions(spectrumMinFreq, spectrumMaxFreq, spectrumSize);
    List<NamedFunction<Outcome, ?>> visualOutcomeFunctions = Utils.visualOutcomeFunctions(spectrumMinFreq, spectrumMaxFreq);
    Listener.Factory<Event<?, ? extends Robot<?>, ? extends Outcome>> factory = Listener.Factory.deaf();
    if (bestFileName == null || output) {
      factory = factory.and(new TabularPrinter<>(Misc.concat(List.of(
          basicFunctions,
          populationFunctions,
          visualFunctions,
          NamedFunction.then(best(), basicIndividualFunctions),
          NamedFunction.then(as(Outcome.class).of(fitness()).of(best()), basicOutcomeFunctions),
          NamedFunction.then(as(Outcome.class).of(fitness()).of(best()), visualOutcomeFunctions)
      ))));
    }
    if (bestFileName != null) {
      factory = factory.and(new CSVPrinter<>(Misc.concat(List.of(
          keysFunctions,
          basicFunctions,
          populationFunctions,
          NamedFunction.then(best(), basicIndividualFunctions),
          NamedFunction.then(best(), detailedIndividualFunctions),
          NamedFunction.then(as(Outcome.class).of(fitness()).of(best()), basicOutcomeFunctions),
          NamedFunction.then(as(Outcome.class).of(fitness()).of(best()), detailedOutcomeFunctions)
      )), new File(bestFileName)
      ));
    }
    if (validationFileName != null) {
      if (!validationTerrainNames.isEmpty() && validationTransformationNames.isEmpty()) {
        validationTransformationNames.add("identity");
      }
      if (validationTerrainNames.isEmpty() && !validationTransformationNames.isEmpty()) {
        validationTerrainNames.add(terrainNames.get(0));
      }
      Listener.Factory<Event<?, ? extends Robot<?>, ? extends Outcome>> validationFactory = Listener.Factory.forEach(
          Utils.validation(validationTerrainNames, validationTransformationNames, List.of(0), episodeTime),
          new CSVPrinter<>(
              Misc.concat(List.of(
                  NamedFunction.then(f("event", (ValidationOutcome vo) -> vo.event), basicFunctions),
                  NamedFunction.then(f("event", (ValidationOutcome vo) -> vo.event), keysFunctions),
                  NamedFunction.then(f("keys", (ValidationOutcome vo) -> vo.keys), List.of(
                      f("validation.terrain", (Map<String, Object> map) -> map.get("validation.terrain")),
                      f("validation.transformation", (Map<String, Object> map) -> map.get("validation.transformation")),
                      f("validation.seed", "%2d", (Map<String, Object> map) -> map.get("validation.seed"))
                  )),
                  NamedFunction.then(f("outcome", (ValidationOutcome vo) -> vo.outcome), basicOutcomeFunctions),
                  NamedFunction.then(f("outcome", (ValidationOutcome vo) -> vo.outcome), detailedOutcomeFunctions)
              )),
              new File(validationFileName)
          )
      ).onLast();
      factory = factory.and(validationFactory);
    }
    if (allFileName != null) {
      factory = factory.and(Listener.Factory.forEach(
          event -> event.getOrderedPopulation().all().stream()
              .map(i -> Pair.of(event, i))
              .collect(Collectors.toList()),
          new CSVPrinter<>(
              Misc.concat(List.of(
                  NamedFunction.then(f("event", Pair::first), keysFunctions),
                  NamedFunction.then(f("event", Pair::first), basicFunctions),
                  NamedFunction.then(f("individual", Pair::second), basicIndividualFunctions),
                  NamedFunction.then(f("individual", Pair::second), detailedIndividualFunctions)
              )),
              new File(allFileName)
          )
      ));
    }
    if (telegramBotId != null && telegramChatId != 0) {
      factory = factory.and(new TelegramUpdater<>(List.of(
          Utils.lastEventToString(keys, fitnessFunction),
          Utils.fitnessPlot(fitnessFunction),
          Utils.centerPositionPlot(),
          Utils.bestVideo(episodeTransientTime, videoEpisodeTime, validationTerrainNames.get(0))
      ), telegramBotId, telegramChatId));
    }
    //summarize params
    L.info("Experiment name: " + experimentName);
    L.info("Evolvers: " + evolverNames);
    L.info("Mappers: " + mapperNames);
    L.info("Shapes: " + targetShapeNames);
    L.info("Sensor configs: " + targetSensorConfigNames);
    L.info("Terrains: " + terrainNames);
    L.info("Transformations: " + transformationNames);
    L.info("Validations: " + Lists.cartesianProduct(validationTerrainNames, validationTransformationNames));
    //start iterations
    int nOfRuns = seeds.length * terrainNames.size() * targetShapeNames.size() * targetSensorConfigNames.size() * mapperNames.size() * transformationNames.size() * evolverNames.size();
    int counter = 0;
    for (int seed : seeds) {
      for (String terrainName : terrainNames) {
        for (String targetShapeName : targetShapeNames) {
          for (String targetSensorConfigName : targetSensorConfigNames) {
            for (String mapperName : mapperNames) {
              for (String transformationName : transformationNames) {
                for (String evolverName : evolverNames) {
                  counter = counter + 1;
                  final Random random = new Random(seed);
                  //prepare consumers
                  keys.putAll(Map.of(
                      "experiment.name", experimentName,
                      "seed", seed,
                      "terrain", terrainName,
                      "shape", targetShapeName,
                      "sensor.config", targetSensorConfigName,
                      "mapper", mapperName,
                      "transformation", transformationName,
                      "evolver", evolverName
                  ));
                  Robot<?> target = new Robot<>(
                      null,
                      RobotUtils.buildSensorizingFunction(targetSensorConfigName).apply(RobotUtils.buildShape(targetShapeName))
                  );
                  //build evolver
                  Evolver<?, Robot<?>, Outcome> evolver;
                  try {
                    evolver = buildEvolver(evolverName, mapperName, target, fitnessFunction);
                  } catch (ClassCastException | IllegalArgumentException e) {
                    L.warning(String.format(
                        "Cannot instantiate %s for %s: %s",
                        evolverName,
                        mapperName,
                        e.toString()
                    ));
                    continue;
                  }
                  //optimize
                  Stopwatch stopwatch = Stopwatch.createStarted();
                  L.info(String.format("Progress %s (%d/%d); Starting %s",
                      TextPlotter.horizontalBar(counter - 1, 0, nOfRuns, 8),
                      counter, nOfRuns,
                      keys
                  ));
                  try {
                    Collection<Robot<?>> solutions = evolver.solve(
                        buildTaskFromName(transformationName, terrainName, episodeTime, random).andThen(o -> o.subOutcome(episodeTransientTime, episodeTime)),
                        new Births(nBirths),
                        random,
                        executorService,
                        factory.build() //TODO could be deferred, but problems arise with the (possibly concurrent) modification of the keys
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
      }
    }
    factory.shutdown();
  }

  private static EvolverBuilder<?> getEvolverBuilderFromName(String name) {
    String numGA = "numGA-(?<nPop>\\d+)";
    String numGASpeciated = "numGASpec-(?<nPop>\\d+)-(?<dT>\\d+(\\.\\d+)?)";
    String cmaES = "CMAES";
    Map<String, String> params;
    if ((params = params(numGA, name)) != null) {
      return new DoublesStandard(
          Integer.parseInt(params.get("nPop")),
          (int) Math.max(Math.round((double) Integer.parseInt(params.get("nPop")) / 10d), 3),
          0.75d
      );
    }
    if ((params = params(numGASpeciated, name)) != null) {
      return new DoublesSpeciated(
          Integer.parseInt(params.get("nPop")),
          (int) Math.max(Math.round((double) Integer.parseInt(params.get("nPop")) / 10d), 3),
          0.75d,
          Double.parseDouble(params.get("dT"))
      );
    }
    if ((params = params(cmaES, name)) != null) {
      return new CMAES();
    }
    throw new IllegalArgumentException(String.format("Unknown evolver builder name: %s", name));
  }

  @SuppressWarnings({"unchecked", "rawtypes"})
  private static PrototypedFunctionBuilder<?, ?> getMapperBuilderFromName(String name) {
    String fixedCentralized = "fixedCentralized";
    String fixedHomoDistributed = "fixedHomoDist-(?<nSignals>\\d+)";
    String fixedHeteroDistributed = "fixedHeteroDist-(?<nSignals>\\d+)";
    String fixedPhasesFunction = "fixedPhasesFunct-(?<f>\\d+)";
    String fixedPhases = "fixedPhases-(?<f>\\d+)";
    String bodySin = "bodySin-(?<fullness>\\d+(\\.\\d+)?)-(?<minF>\\d+(\\.\\d+)?)-(?<maxF>\\d+(\\.\\d+)?)";
    String bodyAndHomoDistributed = "bodyAndHomoDist-(?<fullness>\\d+(\\.\\d+)?)-(?<nSignals>\\d+)-(?<nLayers>\\d+)";
    String sensorAndBodyAndHomoDistributed = "sensorAndBodyAndHomoDist-(?<fullness>\\d+(\\.\\d+)?)-(?<nSignals>\\d+)-(?<nLayers>\\d+)-(?<position>(t|f))";
    String mlp = "MLP-(?<ratio>\\d+(\\.\\d+)?)-(?<nLayers>\\d+)(-(?<actFun>(sin|tanh|sigmoid|relu)))?";
    String directNumGrid = "directNumGrid";
    String functionNumGrid = "functionNumGrid";
    String fgraph = "fGraph";
    String functionGrid = "fGrid-(?<innerMapper>.*)";
    Map<String, String> params;
    //robot mappers
    if ((params = params(fixedCentralized, name)) != null) {
      return new FixedCentralized();
    }
    if ((params = params(fixedHomoDistributed, name)) != null) {
      return new FixedHomoDistributed(
          Integer.parseInt(params.get("nSignals"))
      );
    }
    if ((params = params(fixedHeteroDistributed, name)) != null) {
      return new FixedHeteroDistributed(
          Integer.parseInt(params.get("nSignals"))
      );
    }
    if ((params = params(fixedPhasesFunction, name)) != null) {
      return new FixedPhaseFunction(
          Double.parseDouble(params.get("f")),
          1d
      );
    }
    if ((params = params(fixedPhases, name)) != null) {
      return new FixedPhaseValues(
          Double.parseDouble(params.get("f")),
          1d
      );
    }
    if ((params = params(bodyAndHomoDistributed, name)) != null) {
      return new BodyAndHomoDistributed(
          Integer.parseInt(params.get("nSignals")),
          Double.parseDouble(params.get("fullness"))
      )
          .compose(PrototypedFunctionBuilder.of(List.of(
              new MLP(2d, 3, MultiLayerPerceptron.ActivationFunction.SIN),
              new MLP(0.65d, Integer.parseInt(params.get("nLayers")))
          )))
          .compose(PrototypedFunctionBuilder.merger());
    }
    if ((params = params(sensorAndBodyAndHomoDistributed, name)) != null) {
      return new SensorAndBodyAndHomoDistributed(
          Integer.parseInt(params.get("nSignals")),
          Double.parseDouble(params.get("fullness")),
          params.get("position").equals("t")
      )
          .compose(PrototypedFunctionBuilder.of(List.of(
              new MLP(2d, 3, MultiLayerPerceptron.ActivationFunction.SIN),
              new MLP(1.5d, Integer.parseInt(params.get("nLayers")))
          )))
          .compose(PrototypedFunctionBuilder.merger());
    }
    if ((params = params(bodySin, name)) != null) {
      return new BodyAndSinusoidal(
          Double.parseDouble(params.get("minF")),
          Double.parseDouble(params.get("maxF")),
          Double.parseDouble(params.get("fullness")),
          Set.of(BodyAndSinusoidal.Component.FREQUENCY, BodyAndSinusoidal.Component.PHASE, BodyAndSinusoidal.Component.AMPLITUDE)
      );
    }
    if ((params = params(fixedHomoDistributed, name)) != null) {
      return new FixedHomoDistributed(
          Integer.parseInt(params.get("nSignals"))
      );
    }
    //function mappers
    if ((params = params(mlp, name)) != null) {
      return new MLP(
          Double.parseDouble(params.get("ratio")),
          Integer.parseInt(params.get("nLayers")),
          params.containsKey("actFun") ? MultiLayerPerceptron.ActivationFunction.valueOf(params.get("actFun").toUpperCase()) : MultiLayerPerceptron.ActivationFunction.TANH
      );
    }
    if ((params = params(fgraph, name)) != null) {
      return new FGraph();
    }
    //misc
    if ((params = params(functionGrid, name)) != null) {
      return new FunctionGrid((PrototypedFunctionBuilder) getMapperBuilderFromName(params.get("innerMapper")));
    }
    if ((params = params(directNumGrid, name)) != null) {
      return new DirectNumbersGrid();
    }
    if ((params = params(functionNumGrid, name)) != null) {
      return new FunctionNumbersGrid();
    }
    throw new IllegalArgumentException(String.format("Unknown mapper name: %s", name));
  }

  @SuppressWarnings({"unchecked", "rawtypes"})
  private static Evolver<?, Robot<?>, Outcome> buildEvolver(String evolverName, String robotMapperName, Robot<?>
      target, Function<Outcome, Double> outcomeMeasure) {
    PrototypedFunctionBuilder<?, ?> mapperBuilder = null;
    for (String piece : robotMapperName.split(MAPPER_PIPE_CHAR)) {
      if (mapperBuilder == null) {
        mapperBuilder = getMapperBuilderFromName(piece);
      } else {
        mapperBuilder = mapperBuilder.compose((PrototypedFunctionBuilder) getMapperBuilderFromName(piece));
      }
    }
    return getEvolverBuilderFromName(evolverName).build(
        (PrototypedFunctionBuilder) mapperBuilder,
        target,
        PartialComparator.from(Double.class).comparing(outcomeMeasure).reversed()
    );
  }

  private static Function<Robot<?>, Outcome> buildTaskFromName(String transformationSequenceName, String
      terrainSequenceName, double episodeT, Random random) {
    //for sequence, assume format '99:name>99:name'
    //transformations
    Function<Robot<?>, Robot<?>> transformation;
    if (transformationSequenceName.contains(SEQUENCE_SEPARATOR_CHAR)) {
      transformation = new SequentialFunction<>(Arrays.stream(transformationSequenceName.split(SEQUENCE_SEPARATOR_CHAR))
          .collect(Collectors.toMap(
              p -> Long.parseLong(p.split(SEQUENCE_ITERATION_CHAR)[0]),
              p -> RobotUtils.buildRobotTransformation(p.split(SEQUENCE_ITERATION_CHAR)[1], random)
          )));
    } else {
      transformation = RobotUtils.buildRobotTransformation(transformationSequenceName, random);
    }
    //terrains
    Function<Robot<?>, Outcome> task;
    if (terrainSequenceName.contains(SEQUENCE_SEPARATOR_CHAR)) {
      task = new SequentialFunction<>(Arrays.stream(terrainSequenceName.split(SEQUENCE_SEPARATOR_CHAR))
          .collect(Collectors.toMap(
              p -> Long.parseLong(p.split(SEQUENCE_ITERATION_CHAR)[0]),
              p -> buildLocomotionTask(p.split(SEQUENCE_ITERATION_CHAR)[1], episodeT, random)
          )));
    } else {
      task = buildLocomotionTask(terrainSequenceName, episodeT, random);
    }
    return task.compose(transformation);
  }

  public static Function<Robot<?>, Outcome> buildLocomotionTask(String terrainName, double episodeT, Random random) {
    if (!terrainName.contains("-rnd")) {
      return Misc.cached(new Locomotion(
          episodeT,
          Locomotion.createTerrain(terrainName),
          PHYSICS_SETTINGS
      ), CACHE_SIZE);
    }
    return r -> new Locomotion(
        episodeT,
        Locomotion.createTerrain(terrainName.replace("-rnd", "-" + random.nextInt(10000))),
        PHYSICS_SETTINGS
    ).apply(r);
  }

}
