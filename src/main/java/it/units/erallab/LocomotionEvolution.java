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
import it.units.erallab.builder.*;
import it.units.erallab.builder.evolver.*;
import it.units.erallab.builder.phenotype.*;
import it.units.erallab.builder.robot.*;
import it.units.erallab.hmsrobots.core.controllers.Controller;
import it.units.erallab.hmsrobots.core.controllers.MultiLayerPerceptron;
import it.units.erallab.hmsrobots.core.controllers.PruningMultiLayerPerceptron;
import it.units.erallab.hmsrobots.core.controllers.snn.IzhikevicNeuron;
import it.units.erallab.hmsrobots.core.controllers.snn.LIFNeuron;
import it.units.erallab.hmsrobots.core.controllers.snn.LIFNeuronWithHomeostasis;
import it.units.erallab.hmsrobots.core.controllers.snn.SpikingFunction;
import it.units.erallab.hmsrobots.core.controllers.snn.converters.stv.AverageFrequencySpikeTrainToValueConverter;
import it.units.erallab.hmsrobots.core.controllers.snn.converters.stv.MovingAverageSpikeTrainToValueConverter;
import it.units.erallab.hmsrobots.core.controllers.snn.converters.stv.SpikeTrainToValueConverter;
import it.units.erallab.hmsrobots.core.controllers.snn.converters.vts.UniformValueToSpikeTrainConverter;
import it.units.erallab.hmsrobots.core.controllers.snn.converters.vts.UniformWithMemoryValueToSpikeTrainConverter;
import it.units.erallab.hmsrobots.core.controllers.snn.converters.vts.ValueToSpikeTrainConverter;
import it.units.erallab.hmsrobots.core.controllers.snndiscr.QuantizedIzhikevicNeuron;
import it.units.erallab.hmsrobots.core.controllers.snndiscr.QuantizedLIFNeuron;
import it.units.erallab.hmsrobots.core.controllers.snndiscr.QuantizedLIFNeuronWithHomeostasis;
import it.units.erallab.hmsrobots.core.controllers.snndiscr.QuantizedSpikingFunction;
import it.units.erallab.hmsrobots.core.controllers.snndiscr.converters.stv.QuantizedAverageFrequencySpikeTrainToValueConverter;
import it.units.erallab.hmsrobots.core.controllers.snndiscr.converters.stv.QuantizedMovingAverageSpikeTrainToValueConverter;
import it.units.erallab.hmsrobots.core.controllers.snndiscr.converters.stv.QuantizedSpikeTrainToValueConverter;
import it.units.erallab.hmsrobots.core.controllers.snndiscr.converters.vts.QuantizedUniformValueToSpikeTrainConverter;
import it.units.erallab.hmsrobots.core.controllers.snndiscr.converters.vts.QuantizedUniformWithMemoryValueToSpikeTrainConverter;
import it.units.erallab.hmsrobots.core.controllers.snndiscr.converters.vts.QuantizedValueToSpikeTrainConverter;
import it.units.erallab.hmsrobots.core.objects.Robot;
import it.units.erallab.hmsrobots.tasks.locomotion.Locomotion;
import it.units.erallab.hmsrobots.tasks.locomotion.Outcome;
import it.units.erallab.hmsrobots.util.RobotUtils;
import it.units.erallab.hmsrobots.viewers.drawers.Drawer;
import it.units.erallab.hmsrobots.viewers.drawers.Drawers;
import it.units.erallab.utils.Utils;
import it.units.malelab.jgea.Worker;
import it.units.malelab.jgea.core.Individual;
import it.units.malelab.jgea.core.evolver.Event;
import it.units.malelab.jgea.core.evolver.Evolver;
import it.units.malelab.jgea.core.evolver.stopcondition.FitnessEvaluations;
import it.units.malelab.jgea.core.listener.*;
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
import java.util.function.BiFunction;
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
    double episodeTransientTime = d(a("episodeTransientTime", "1"));
    double validationEpisodeTime = d(a("validationEpisodeTime", Double.toString(episodeTime)));
    double validationEpisodeTransientTime = d(a("validationEpisodeTransientTime", Double.toString(episodeTransientTime)));
    double videoEpisodeTime = d(a("videoEpisodeTime", "10"));
    double videoEpisodeTransientTime = d(a("videoEpisodeTransientTime", "0"));
    int nEvals = i(a("nEvals", "1000"));
    int[] seeds = ri(a("seed", "0:1"));
    String experimentName = a("expName", "short");
    List<String> terrainNames = l(a("terrain", "hilly-1-10-rnd"));
    List<String> targetShapeNames = l(a("shape", "biped-4x3"));
    List<String> targetSensorConfigNames = l(a("sensorConfig", "spinedTouchSighted-f-f-0.01"));
    List<String> transformationNames = l(a("transformation", "identity"));
    List<String> evolverNames = l(a("evolver", "ES-40-0.35"));
    List<String> mapperNames = l(a("mapper", "fixedCentralized<MLP-1-1-tanh"));
    String bestFileName = a("bestFile", null);
    String lastFileName = a("lastFile", null);
    String allFileName = a("allFile", null);
    String validationFileName = a("validationFile", null);
    boolean deferred = a("deferred", "true").startsWith("t");
    String telegramBotId = a("telegramBotId", null);
    long telegramChatId = Long.parseLong(a("telegramChatId", "0"));
    List<String> serializationFlags = l(a("serialization", "")); //last,best,all
    boolean output = a("output", "false").startsWith("t");
    List<String> validationTransformationNames = l(a("validationTransformation", "")).stream().filter(s -> !s.isEmpty()).collect(Collectors.toList());
    List<String> validationTerrainNames = l(a("validationTerrain", "flat")).stream().filter(s -> !s.isEmpty()).collect(Collectors.toList());
    String fitnessMetrics = a("fitness", "velocity");
    String videoConfiguration = a("videoConfiguration", "basicWithMiniWorldAndBrain");
    Pair<Pair<Integer, Integer>, Function<String,Drawer>> drawerSupplier = getDrawerSupplierFromName(videoConfiguration);
    Function<Outcome, Double> fitnessFunction = getFitnessFunctionFromName(fitnessMetrics);
    //consumers
    List<NamedFunction<Event<?, ? extends Robot<?>, ? extends Outcome>, ?>> keysFunctions = Utils.keysFunctions();
    List<NamedFunction<Event<?, ? extends Robot<?>, ? extends Outcome>, ?>> basicFunctions = Utils.basicFunctions();
    List<NamedFunction<Individual<?, ? extends Robot<?>, ? extends Outcome>, ?>> basicIndividualFunctions = Utils.individualFunctions(fitnessFunction);
    List<NamedFunction<Event<?, ? extends Robot<?>, ? extends Outcome>, ?>> populationFunctions = Utils.populationFunctions(fitnessFunction);
    List<NamedFunction<Event<?, ? extends Robot<?>, ? extends Outcome>, ?>> computationTimeFunctions = Utils.computationTimeFunctions();
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
          computationTimeFunctions,
          NamedFunction.then(best(), basicIndividualFunctions),
          NamedFunction.then(as(Outcome.class).of(fitness()).of(best()), basicOutcomeFunctions),
          NamedFunction.then(as(Outcome.class).of(fitness()).of(best()), visualOutcomeFunctions)
      ))));
    }
    if (lastFileName != null) {
      factory = factory.and(new CSVPrinter<>(Misc.concat(List.of(
          keysFunctions,
          basicFunctions,
          populationFunctions,
          NamedFunction.then(best(), basicIndividualFunctions),
          NamedFunction.then(as(Outcome.class).of(fitness()).of(best()), basicOutcomeFunctions),
          NamedFunction.then(as(Outcome.class).of(fitness()).of(best()), detailedOutcomeFunctions),
          NamedFunction.then(best(), Utils.serializationFunction(serializationFlags.contains("last")))
      )), new File(lastFileName)
      ).onLast());
    }
    if (bestFileName != null) {
      factory = factory.and(new CSVPrinter<>(Misc.concat(List.of(
          keysFunctions,
          basicFunctions,
          populationFunctions,
          NamedFunction.then(best(), basicIndividualFunctions),
          NamedFunction.then(as(Outcome.class).of(fitness()).of(best()), basicOutcomeFunctions),
          NamedFunction.then(as(Outcome.class).of(fitness()).of(best()), detailedOutcomeFunctions),
          NamedFunction.then(best(), Utils.serializationFunction(serializationFlags.contains("best")))
      )), new File(bestFileName)
      ));
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
                  NamedFunction.then(f("individual", Pair::second), Utils.serializationFunction(serializationFlags.contains("all")))
              )),
              new File(allFileName)
          )
      ));
    }
    //validation listener
    if (validationFileName != null) {
      if (!validationTerrainNames.isEmpty() && validationTransformationNames.isEmpty()) {
        validationTransformationNames.add("identity");
      }
      if (validationTerrainNames.isEmpty() && !validationTransformationNames.isEmpty()) {
        validationTerrainNames.add(terrainNames.get(0));
      }
      Listener.Factory<Event<?, ? extends Robot<?>, ? extends Outcome>> validationFactory = Listener.Factory.forEach(
          Utils.validation(validationTerrainNames, validationTransformationNames, List.of(0), validationEpisodeTime),
          new CSVPrinter<>(
              Misc.concat(List.of(
                  NamedFunction.then(f("event", (ValidationOutcome vo) -> vo.event), basicFunctions),
                  NamedFunction.then(f("event", (ValidationOutcome vo) -> vo.event), keysFunctions),
                  NamedFunction.then(f("keys", (ValidationOutcome vo) -> vo.keys), List.of(
                      f("validation.terrain", (Map<String, Object> map) -> map.get("validation.terrain")),
                      f("validation.transformation", (Map<String, Object> map) -> map.get("validation.transformation")),
                      f("validation.seed", "%2d", (Map<String, Object> map) -> map.get("validation.seed"))
                  )),
                  NamedFunction.then(
                      f("outcome", (ValidationOutcome vo) -> vo.outcome.subOutcome(validationEpisodeTransientTime, validationEpisodeTime)),
                      basicOutcomeFunctions
                  ),
                  NamedFunction.then(
                      f("outcome", (ValidationOutcome vo) -> vo.outcome.subOutcome(validationEpisodeTransientTime, validationEpisodeTime)),
                      detailedOutcomeFunctions
                  )
              )),
              new File(validationFileName)
          )
      ).onLast();
      factory = factory.and(validationFactory);
    }
    if (telegramBotId != null && telegramChatId != 0) {
      factory = factory.and(new TelegramUpdater<>(List.of(
          Utils.lastEventToString(fitnessFunction),
          Utils.fitnessPlot(fitnessFunction),
          Utils.centerPositionPlot(),
          Utils.bestVideo(videoEpisodeTransientTime, videoEpisodeTime, PHYSICS_SETTINGS, drawerSupplier)
      ), telegramBotId, telegramChatId));
    }
    //summarize params
    L.info("Experiment name: " + experimentName);
    L.info("N evaluations: " + nEvals);
    L.info("Evolvers: " + evolverNames);
    L.info("Mappers: " + mapperNames);
    L.info("Fitness metrics: " + fitnessMetrics);
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
                  //prepare keys
                  Map<String, Object> keys = Map.ofEntries(
                      Map.entry("experiment.name", experimentName),
                      Map.entry("seed", seed),
                      Map.entry("terrain", terrainName),
                      Map.entry("shape", targetShapeName),
                      Map.entry("sensor.config", targetSensorConfigName),
                      Map.entry("mapper", mapperName),
                      Map.entry("transformation", transformationName),
                      Map.entry("evolver", evolverName),
                      Map.entry("fitness.metrics", fitnessMetrics)
                  );
                  Robot<?> target = new Robot<>(
                      Controller.empty(),
                      RobotUtils.buildSensorizingFunction(targetSensorConfigName).apply(RobotUtils.buildShape(targetShapeName))
                  );
                  //build evolver
                  Evolver<?, Robot<?>, Outcome> evolver;
                  try {
                    evolver = buildEvolver(evolverName, mapperName, target, fitnessFunction);
                  } catch (ClassCastException | IllegalArgumentException e) {
                    e.printStackTrace();
                    L.warning(String.format(
                        "Cannot instantiate %s for %s: %s",
                        evolverName,
                        mapperName,
                        e
                    ));
                    continue;
                  }
                  Listener<Event<?, ? extends Robot<?>, ? extends Outcome>> listener = Listener.all(List.of(
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
                    Collection<Robot<?>> solutions = evolver.solve(
                        buildTaskFromName(transformationName, terrainName, episodeTime, random).andThen(o -> o.subOutcome(episodeTransientTime, episodeTime)),
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
      }
    }
    factory.shutdown();
  }

  private static EvolverBuilder<?> getEvolverBuilderFromName(String name) {
    String numGA = "numGA-(?<nPop>\\d+)-(?<diversity>(t|f))";
    String numGASpeciated = "numGASpec-(?<nPop>\\d+)-(?<nSpecies>\\d+)-(?<criterion>(" + Arrays.stream(DoublesSpeciated.SpeciationCriterion.values()).map(c -> c.name().toLowerCase(Locale.ROOT)).collect(Collectors.joining("|")) + "))";
    String cmaES = "CMAES";
    String eS = "ES-(?<nPop>\\d+)-(?<sigma>\\d+(\\.\\d+)?)";
    String sparseGA = "sparse-(?<nPop>\\d+)-(?<p>\\d+(\\.\\d+)?)";
    String STDPstandardGA = "STDP_GA-(?<nPop>\\d+)-(?<diversity>(t|f))";
    Map<String, String> params;
    if ((params = params(numGA, name)) != null) {
      return new DoublesStandard(
          Integer.parseInt(params.get("nPop")),
          (int) Math.max(Math.round((double) Integer.parseInt(params.get("nPop")) / 10d), 3),
          0.75d,
          params.get("diversity").equals("t")
      );
    }
    if ((params = params(numGASpeciated, name)) != null) {
      return new DoublesSpeciated(
          Integer.parseInt(params.get("nPop")),
          Integer.parseInt(params.get("nSpecies")),
          0.75d,
          DoublesSpeciated.SpeciationCriterion.valueOf(params.get("criterion").toUpperCase())
      );
    }
    if ((params = params(eS, name)) != null) {
      return new ES(
          Double.parseDouble(params.get("sigma")),
          Integer.parseInt(params.get("nPop"))
      );
    }
    if ((params = params(cmaES, name)) != null) {
      return new CMAES();
    }
    if ((params = params(sparseGA, name)) != null) {
      return new DoublesInitiallySparse(
          Integer.parseInt(params.get("nPop")),
          (int) Math.max(Math.round((double) Integer.parseInt(params.get("nPop")) / 10d), 3),
          Double.parseDouble(params.get("p"))
      );
    }
    if ((params = params(STDPstandardGA, name)) != null) {
      return new STDPStandard(
          Integer.parseInt(params.get("nPop")),
          (int) Math.max(Math.round((double) Integer.parseInt(params.get("nPop")) / 10d), 3),
          0d,
          params.get("diversity").equals("t")
      );
    }
    throw new IllegalArgumentException(String.format("Unknown evolver builder name: %s", name));
  }

  private static Function<Outcome, Double> getFitnessFunctionFromName(String name) {
    String velocity = "velocity";
    String efficiency = "efficiency";
    String weightsPenalizedVelocity = "velocity-(?<lambda>\\d+(\\.\\d+)?)\\*weights";
    Map<String,String> params;
    if (params(velocity, name) != null) {
      return Outcome::getVelocity;
    }
    if (params(efficiency, name) != null) {
      return Outcome::getCorrectedEfficiency;
    }
    if ((params = params(weightsPenalizedVelocity, name)) != null) {
      System.out.println(params.get("lambda"));
      return o -> (o.getVelocity() - Double.parseDouble(params.get("lambda")) * o.getInitialSumOfAbsoluteWeights());
    }
    throw new IllegalArgumentException(String.format("Unknown fitness function name: %s", name));
  }

  private static Pair<Pair<Integer, Integer>, Function<String,Drawer>> getDrawerSupplierFromName(String name) {
    String basic = "basic";
    String basicWithMiniWorld = "basicWithMiniWorld";
    String basicWithMiniWorldAndBrain = "basicWithMiniWorldAndBrain";
    String basicWithMiniWorldAndBrainUsage = "basicWithMiniWorldAndBrainUsage";
    if (params(basic, name) != null) {
      return Pair.of(Pair.of(1, 1), Drawers::basic);
    }
    if (params(basicWithMiniWorld, name) != null) {
      return Pair.of(Pair.of(1, 1), Drawers::basicWithMiniWorld);
    }
    if (params(basicWithMiniWorldAndBrain, name) != null) {
      return Pair.of(Pair.of(1, 2), Drawers::basicWithMiniWorldAndBrain);
    }
    if(params(basicWithMiniWorldAndBrain,name)!=null){
      return Pair.of(Pair.of(1,2), Drawers::basicWithMiniWorldAndBrainUsage);
    }
    throw new IllegalArgumentException(String.format("Unknown video configuration name: %s", name));
  }

  @SuppressWarnings({"unchecked", "rawtypes"})
  private static PrototypedFunctionBuilder<?, ?> getMapperBuilderFromName(String name) {
    String fixedCentralized = "fixedCentralized";
    String fixedHomoDistributed = "fixedHomoDist-(?<nSignals>\\d+)";
    String fixedHeteroDistributed = "fixedHeteroDist-(?<nSignals>\\d+)";
    String fixedHomoSpikingDistributed = "fixedHomoSpikeDist-(?<nSignals>\\d+)" +
        "-(?<iConv>(unif|unif_mem))-(?<iFreq>\\d+(\\.\\d+)?)-(?<oConv>(avg|avg_mem))(-(?<oMem>\\d+))?-(?<oFreq>\\d+(\\.\\d+)?)";
    String fixedHeteroSpikingDistributed = "fixedHeteroSpikeDist-(?<nSignals>\\d+)" +
        "-(?<iConv>(unif|unif_mem))-(?<iFreq>\\d+(\\.\\d+)?)-(?<oConv>(avg|avg_mem))(-(?<oMem>\\d+))?-(?<oFreq>\\d+(\\.\\d+)?)";
    String fixedHomoQuantizedSpikingDistributed = "fixedHomoQuantSpikeDist-(?<nSignals>\\d+)" +
        "-(?<iConv>(unif|unif_mem))-(?<iFreq>\\d+(\\.\\d+)?)-(?<oConv>(avg|avg_mem))(-(?<oMem>\\d+))?-(?<oFreq>\\d+(\\.\\d+)?)";
    String fixedHeteroQuantizedSpikingDistributed = "fixedHeteroQuantSpikeDist-(?<nSignals>\\d+)" +
        "-(?<iConv>(unif|unif_mem))-(?<iFreq>\\d+(\\.\\d+)?)-(?<oConv>(avg|avg_mem))(-(?<oMem>\\d+))?-(?<oFreq>\\d+(\\.\\d+)?)";
    String fixedPhasesFunction = "fixedPhasesFunct-(?<f>\\d+)";
    String fixedPhases = "fixedPhases-(?<f>\\d+)";
    String fixedPhasesAndFrequencies = "fixedPhasesAndFrequencies";
    String bodySin = "bodySin-(?<fullness>\\d+(\\.\\d+)?)-(?<minF>\\d+(\\.\\d+)?)-(?<maxF>\\d+(\\.\\d+)?)";
    String bodyAndHomoDistributed = "bodyAndHomoDist-(?<fullness>\\d+(\\.\\d+)?)-(?<nSignals>\\d+)-(?<nLayers>\\d+)";
    String sensorAndBodyAndHomoDistributed = "sensorAndBodyAndHomoDist-(?<fullness>\\d+(\\.\\d+)?)-(?<nSignals>\\d+)-(?<nLayers>\\d+)-(?<position>(t|f))";
    String sensorCentralized = "sensorCentralized-(?<nLayers>\\d+)";
    String mlp = "MLP-(?<ratio>\\d+(\\.\\d+)?)-(?<nLayers>\\d+)(-(?<actFun>(sin|tanh|sigmoid|relu)))?";
    String pruningMlp = "pMLP-(?<ratio>\\d+(\\.\\d+)?)-(?<nLayers>\\d+)-(?<actFun>(sin|tanh|sigmoid|relu))-(?<pruningTime>\\d+(\\.\\d+)?)-(?<pruningRate>0(\\.\\d+)?)-(?<criterion>(weight|abs_signal_mean|random))";
    String msn = "MSNd-(?<ratio>\\d+(\\.\\d+)?)-(?<nLayers>\\d+)-(?<spikeType>(lif|iz|lif_h))" +
        "(-(?<lRestPot>-?\\d+(\\.\\d+)?)-(?<lThreshPot>-?\\d+(\\.\\d+)?)-(?<lambda>\\d+(\\.\\d+)?)(-(?<theta>\\d+(\\.\\d+)?))?)?" +
        "(-(?<izParams>(regular_spiking_params)))?";
    String quantizedMsn = "QMSNd-(?<ratio>\\d+(\\.\\d+)?)-(?<nLayers>\\d+)-(?<spikeType>(lif|iz|lif_h))" +
        "(-(?<lRestPot>-?\\d+(\\.\\d+)?)-(?<lThreshPot>-?\\d+(\\.\\d+)?)-(?<lambda>\\d+(\\.\\d+)?)(-(?<theta>\\d+(\\.\\d+)?))?)?" +
        "(-(?<izParams>(regular_spiking_params)))?";
    String msnWithConverter = "MSN-(?<ratio>\\d+(\\.\\d+)?)-(?<nLayers>\\d+)-(?<spikeType>(lif|iz|lif_h|lif_h_output|lif_h_io))" +
        "(-(?<lRestPot>-?\\d+(\\.\\d+)?)-(?<lThreshPot>-?\\d+(\\.\\d+)?)-(?<lambda>\\d+(\\.\\d+)?)(-(?<theta>\\d+(\\.\\d+)?))?)?" +
        "(-(?<izParams>(regular_spiking_params)))?" +
        "-(?<iConv>(unif|unif_mem))-(?<iFreq>\\d+(\\.\\d+)?)-(?<oConv>(avg|avg_mem))(-(?<oMem>\\d+))?-(?<oFreq>\\d+(\\.\\d+)?)";
    String quantizedMsnWithConverter = "QMSN-(?<ratio>\\d+(\\.\\d+)?)-(?<nLayers>\\d+)-(?<spikeType>(lif|iz|lif_h|lif_h_output|lif_h_io))" +
        "(-(?<lRestPot>-?\\d+(\\.\\d+)?)-(?<lThreshPot>-?\\d+(\\.\\d+)?)-(?<lambda>\\d+(\\.\\d+)?)(-(?<theta>\\d+(\\.\\d+)?))?)?" +
        "(-(?<izParams>(regular_spiking_params)))?" +
        "-(?<iConv>(unif|unif_mem))-(?<iFreq>\\d+(\\.\\d+)?)-(?<oConv>(avg|avg_mem))(-(?<oMem>\\d+))?-(?<oFreq>\\d+(\\.\\d+)?)";
    String learningMsnWithConverter = "LMSN-(?<ratio>\\d+(\\.\\d+)?)-(?<nLayers>\\d+)-(?<spikeType>(lif|iz|lif_h))" +
        "(-(?<lRestPot>-?\\d+(\\.\\d+)?)-(?<lThreshPot>-?\\d+(\\.\\d+)?)-(?<lambda>\\d+(\\.\\d+)?)(-(?<theta>\\d+(\\.\\d+)?))?)?" +
        "(-(?<izParams>(regular_spiking_params)))?" +
        "-(?<iConv>(unif|unif_mem))-(?<iFreq>\\d+(\\.\\d+)?)-(?<oConv>(avg|avg_mem))(-(?<oMem>\\d+))?-(?<oFreq>\\d+(\\.\\d+)?)";
    String quantizedNumericLearningMsnWithConverter = "QNLMSN-(?<ratio>\\d+(\\.\\d+)?)-(?<nLayers>\\d+)-(?<spikeType>(lif|iz|lif_h))" +
        "(-(?<lRestPot>-?\\d+(\\.\\d+)?)-(?<lThreshPot>-?\\d+(\\.\\d+)?)-(?<lambda>\\d+(\\.\\d+)?)(-(?<theta>\\d+(\\.\\d+)?))?)?" +
        "(-(?<izParams>(regular_spiking_params)))?" +
        "-(?<iConv>(unif|unif_mem))-(?<iFreq>\\d+(\\.\\d+)?)-(?<oConv>(avg|avg_mem))(-(?<oMem>\\d+))?-(?<oFreq>\\d+(\\.\\d+)?)";
    String quantizedNumericLearningWithFixedRuleValuesAnd0WeightsMSNWithConverters = "QNLFVMSN-(?<ratio>\\d+(\\.\\d+)?)-(?<nLayers>\\d+)-(?<spikeType>(lif|iz|lif_h))" +
        "(-(?<lRestPot>-?\\d+(\\.\\d+)?)-(?<lThreshPot>-?\\d+(\\.\\d+)?)-(?<lambda>\\d+(\\.\\d+)?)(-(?<theta>\\d+(\\.\\d+)?))?)?" +
        "(-(?<izParams>(regular_spiking_params)))?" +
        "-(?<iConv>(unif|unif_mem))-(?<iFreq>\\d+(\\.\\d+)?)-(?<oConv>(avg|avg_mem))(-(?<oMem>\\d+))?-(?<oFreq>\\d+(\\.\\d+)?)" +
        "-(?<symmParam1>\\d+(\\.\\d+))-(?<symmParam2>\\d+(\\.\\d+))-(?<symmParam3>\\d+(\\.\\d+))-(?<symmParam4>\\d+(\\.\\d+))" +
        "-(?<asymmParam1>\\d+(\\.\\d+))-(?<asymmParam2>\\d+(\\.\\d+))-(?<asymmParam3>\\d+(\\.\\d+))-(?<asymmParam4>\\d+(\\.\\d+))";
    String quantizedNumericLearningFixedPoolMsnWithConverter = "QNLFMSN-(?<nRules>\\d+)-(?<ratio>\\d+(\\.\\d+)?)-(?<nLayers>\\d+)-(?<spikeType>(lif|iz|lif_h))" +
        "(-(?<lRestPot>-?\\d+(\\.\\d+)?)-(?<lThreshPot>-?\\d+(\\.\\d+)?)-(?<lambda>\\d+(\\.\\d+)?)(-(?<theta>\\d+(\\.\\d+)?))?)?" +
        "(-(?<izParams>(regular_spiking_params)))?" +
        "-(?<iConv>(unif|unif_mem))-(?<iFreq>\\d+(\\.\\d+)?)-(?<oConv>(avg|avg_mem))(-(?<oMem>\\d+))?-(?<oFreq>\\d+(\\.\\d+)?)";
    String quantizedHebbianNumericLearningMsnWithConverter = "QHLMSN-(?<ratio>\\d+(\\.\\d+)?)-(?<nLayers>\\d+)-(?<spikeType>(lif|iz|lif_h))" +
        "(-(?<lRestPot>-?\\d+(\\.\\d+)?)-(?<lThreshPot>-?\\d+(\\.\\d+)?)-(?<lambda>\\d+(\\.\\d+)?)(-(?<theta>\\d+(\\.\\d+)?))?)?" +
        "(-(?<izParams>(regular_spiking_params)))?" +
        "-(?<iConv>(unif|unif_mem))-(?<iFreq>\\d+(\\.\\d+)?)-(?<oConv>(avg|avg_mem))(-(?<oMem>\\d+))?-(?<oFreq>\\d+(\\.\\d+)?)";
    String quantizedHebbianNumericLearningWeightsMSNWithConverters = "QHLWMSN-(?<ratio>\\d+(\\.\\d+)?)-(?<nLayers>\\d+)-(?<spikeType>(lif|iz|lif_h))" +
        "(-(?<lRestPot>-?\\d+(\\.\\d+)?)-(?<lThreshPot>-?\\d+(\\.\\d+)?)-(?<lambda>\\d+(\\.\\d+)?)(-(?<theta>\\d+(\\.\\d+)?))?)?" +
        "(-(?<izParams>(regular_spiking_params)))?" +
        "-(?<iConv>(unif|unif_mem))-(?<iFreq>\\d+(\\.\\d+)?)-(?<oConv>(avg|avg_mem))(-(?<oMem>\\d+))?-(?<oFreq>\\d+(\\.\\d+)?)";
    String quantizedHebbianNumericLearningClippedWeightsMSNWithConverters = "QHLCWMSN-(?<maxWeight>\\d+(\\.\\d+)?)-(?<ratio>\\d+(\\.\\d+)?)-(?<nLayers>\\d+)-(?<spikeType>(lif|iz|lif_h))" +
        "(-(?<lRestPot>-?\\d+(\\.\\d+)?)-(?<lThreshPot>-?\\d+(\\.\\d+)?)-(?<lambda>\\d+(\\.\\d+)?)(-(?<theta>\\d+(\\.\\d+)?))?)?" +
        "(-(?<izParams>(regular_spiking_params)))?" +
        "-(?<iConv>(unif|unif_mem))-(?<iFreq>\\d+(\\.\\d+)?)-(?<oConv>(avg|avg_mem))(-(?<oMem>\\d+))?-(?<oFreq>\\d+(\\.\\d+)?)";

    String directNumGrid = "directNumGrid";
    String functionNumGrid = "functionNumGrid";
    String fgraph = "fGraph";
    String functionGrid = "fGrid-(?<innerMapper>.*)";
    String spikingFunctionGrid = "snnFuncGrid-(?<innerMapper>.*)";
    String spikingQuantizedFunctionGrid = "snnQuantFuncGrid-(?<innerMapper>.*)";
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
    if ((params = params(fixedHomoSpikingDistributed, name)) != null) {
      ValueToSpikeTrainConverter valueToSpikeTrainConverter = new UniformValueToSpikeTrainConverter();
      SpikeTrainToValueConverter spikeTrainToValueConverter = new AverageFrequencySpikeTrainToValueConverter();
      if (params.containsKey("iConv")) {
        switch (params.get("iConv")) {
          case "unif":
            if (params.containsKey("iFreq")) {
              valueToSpikeTrainConverter = new UniformValueToSpikeTrainConverter(Double.parseDouble(params.get("iFreq")));
            } else {
              valueToSpikeTrainConverter = new UniformValueToSpikeTrainConverter();
            }
            break;
          case "unif_mem":
            if (params.containsKey("iFreq")) {
              valueToSpikeTrainConverter = new UniformWithMemoryValueToSpikeTrainConverter(Double.parseDouble(params.get("iFreq")));
            } else {
              valueToSpikeTrainConverter = new UniformWithMemoryValueToSpikeTrainConverter();
            }
            break;
        }
      }
      if (params.containsKey("oConv")) {
        switch (params.get("oConv")) {
          case "avg":
            if (params.containsKey("oFreq")) {
              spikeTrainToValueConverter = new AverageFrequencySpikeTrainToValueConverter(Double.parseDouble(params.get("oFreq")));
            } else {
              spikeTrainToValueConverter = new AverageFrequencySpikeTrainToValueConverter();
            }
            break;
          case "avg_mem":
            if (params.containsKey("oFreq")) {
              if (params.containsKey("oMem")) {
                spikeTrainToValueConverter = new MovingAverageSpikeTrainToValueConverter(Double.parseDouble(params.get("oFreq")),
                    Integer.parseInt(params.get("oMem")));
              } else {
                spikeTrainToValueConverter = new MovingAverageSpikeTrainToValueConverter(Double.parseDouble(params.get("oFreq")));
              }
            } else {
              if (params.containsKey("oMem")) {
                spikeTrainToValueConverter = new MovingAverageSpikeTrainToValueConverter(Integer.parseInt(params.get("oMem")));
              } else {
                spikeTrainToValueConverter = new MovingAverageSpikeTrainToValueConverter();
              }
            }
            break;
        }
      }
      return new FixedHomoSpikingDistributed(
          Integer.parseInt(params.get("nSignals")),
          valueToSpikeTrainConverter,
          spikeTrainToValueConverter
      );
    }
    if ((params = params(fixedHeteroSpikingDistributed, name)) != null) {
      ValueToSpikeTrainConverter valueToSpikeTrainConverter = new UniformValueToSpikeTrainConverter();
      SpikeTrainToValueConverter spikeTrainToValueConverter = new AverageFrequencySpikeTrainToValueConverter();
      if (params.containsKey("iConv")) {
        switch (params.get("iConv")) {
          case "unif":
            if (params.containsKey("iFreq")) {
              valueToSpikeTrainConverter = new UniformValueToSpikeTrainConverter(Double.parseDouble(params.get("iFreq")));
            } else {
              valueToSpikeTrainConverter = new UniformValueToSpikeTrainConverter();
            }
            break;
          case "unif_mem":
            if (params.containsKey("iFreq")) {
              valueToSpikeTrainConverter = new UniformWithMemoryValueToSpikeTrainConverter(Double.parseDouble(params.get("iFreq")));
            } else {
              valueToSpikeTrainConverter = new UniformWithMemoryValueToSpikeTrainConverter();
            }
            break;
        }
      }
      if (params.containsKey("oConv")) {
        switch (params.get("oConv")) {
          case "avg":
            if (params.containsKey("oFreq")) {
              spikeTrainToValueConverter = new AverageFrequencySpikeTrainToValueConverter(Double.parseDouble(params.get("oFreq")));
            } else {
              spikeTrainToValueConverter = new AverageFrequencySpikeTrainToValueConverter();
            }
            break;
          case "avg_mem":
            if (params.containsKey("oFreq")) {
              if (params.containsKey("oMem")) {
                spikeTrainToValueConverter = new MovingAverageSpikeTrainToValueConverter(Double.parseDouble(params.get("oFreq")),
                    Integer.parseInt(params.get("oMem")));
              } else {
                spikeTrainToValueConverter = new MovingAverageSpikeTrainToValueConverter(Double.parseDouble(params.get("oFreq")));
              }
            } else {
              if (params.containsKey("oMem")) {
                spikeTrainToValueConverter = new MovingAverageSpikeTrainToValueConverter(Integer.parseInt(params.get("oMem")));
              } else {
                spikeTrainToValueConverter = new MovingAverageSpikeTrainToValueConverter();
              }
            }
            break;
        }
      }
      return new FixedHeteroSpikingDistributed(
          Integer.parseInt(params.get("nSignals")),
          valueToSpikeTrainConverter,
          spikeTrainToValueConverter
      );
    }
    if ((params = params(fixedHomoQuantizedSpikingDistributed, name)) != null) {
      QuantizedValueToSpikeTrainConverter valueToSpikeTrainConverter = new QuantizedUniformValueToSpikeTrainConverter();
      QuantizedSpikeTrainToValueConverter spikeTrainToValueConverter = new QuantizedAverageFrequencySpikeTrainToValueConverter();
      if (params.containsKey("iConv")) {
        switch (params.get("iConv")) {
          case "unif":
            if (params.containsKey("iFreq")) {
              valueToSpikeTrainConverter = new QuantizedUniformValueToSpikeTrainConverter(Double.parseDouble(params.get("iFreq")));
            } else {
              valueToSpikeTrainConverter = new QuantizedUniformValueToSpikeTrainConverter();
            }
            break;
          case "unif_mem":
            if (params.containsKey("iFreq")) {
              valueToSpikeTrainConverter = new QuantizedUniformWithMemoryValueToSpikeTrainConverter(Double.parseDouble(params.get("iFreq")));
            } else {
              valueToSpikeTrainConverter = new QuantizedUniformWithMemoryValueToSpikeTrainConverter();
            }
            break;
        }
      }
      if (params.containsKey("oConv")) {
        switch (params.get("oConv")) {
          case "avg":
            if (params.containsKey("oFreq")) {
              spikeTrainToValueConverter = new QuantizedAverageFrequencySpikeTrainToValueConverter(Double.parseDouble(params.get("oFreq")));
            } else {
              spikeTrainToValueConverter = new QuantizedAverageFrequencySpikeTrainToValueConverter();
            }
            break;
          case "avg_mem":
            if (params.containsKey("oFreq")) {
              if (params.containsKey("oMem")) {
                spikeTrainToValueConverter = new QuantizedMovingAverageSpikeTrainToValueConverter(Double.parseDouble(params.get("oFreq")),
                    Integer.parseInt(params.get("oMem")));
              } else {
                spikeTrainToValueConverter = new QuantizedMovingAverageSpikeTrainToValueConverter(Double.parseDouble(params.get("oFreq")));
              }
            } else {
              if (params.containsKey("oMem")) {
                spikeTrainToValueConverter = new QuantizedMovingAverageSpikeTrainToValueConverter(Integer.parseInt(params.get("oMem")));
              } else {
                spikeTrainToValueConverter = new QuantizedMovingAverageSpikeTrainToValueConverter();
              }
            }
            break;
        }
      }
      return new FixedHomoQuantizedSpikingDistributed(
          Integer.parseInt(params.get("nSignals")),
          valueToSpikeTrainConverter,
          spikeTrainToValueConverter
      );
    }
    if ((params = params(fixedHeteroQuantizedSpikingDistributed, name)) != null) {
      QuantizedValueToSpikeTrainConverter valueToSpikeTrainConverter = new QuantizedUniformValueToSpikeTrainConverter();
      QuantizedSpikeTrainToValueConverter spikeTrainToValueConverter = new QuantizedAverageFrequencySpikeTrainToValueConverter();
      if (params.containsKey("iConv")) {
        switch (params.get("iConv")) {
          case "unif":
            if (params.containsKey("iFreq")) {
              valueToSpikeTrainConverter = new QuantizedUniformValueToSpikeTrainConverter(Double.parseDouble(params.get("iFreq")));
            } else {
              valueToSpikeTrainConverter = new QuantizedUniformValueToSpikeTrainConverter();
            }
            break;
          case "unif_mem":
            if (params.containsKey("iFreq")) {
              valueToSpikeTrainConverter = new QuantizedUniformWithMemoryValueToSpikeTrainConverter(Double.parseDouble(params.get("iFreq")));
            } else {
              valueToSpikeTrainConverter = new QuantizedUniformWithMemoryValueToSpikeTrainConverter();
            }
            break;
        }
      }
      if (params.containsKey("oConv")) {
        switch (params.get("oConv")) {
          case "avg":
            if (params.containsKey("oFreq")) {
              spikeTrainToValueConverter = new QuantizedAverageFrequencySpikeTrainToValueConverter(Double.parseDouble(params.get("oFreq")));
            } else {
              spikeTrainToValueConverter = new QuantizedAverageFrequencySpikeTrainToValueConverter();
            }
            break;
          case "avg_mem":
            if (params.containsKey("oFreq")) {
              if (params.containsKey("oMem")) {
                spikeTrainToValueConverter = new QuantizedMovingAverageSpikeTrainToValueConverter(Double.parseDouble(params.get("oFreq")),
                    Integer.parseInt(params.get("oMem")));
              } else {
                spikeTrainToValueConverter = new QuantizedMovingAverageSpikeTrainToValueConverter(Double.parseDouble(params.get("oFreq")));
              }
            } else {
              if (params.containsKey("oMem")) {
                spikeTrainToValueConverter = new QuantizedMovingAverageSpikeTrainToValueConverter(Integer.parseInt(params.get("oMem")));
              } else {
                spikeTrainToValueConverter = new QuantizedMovingAverageSpikeTrainToValueConverter();
              }
            }
            break;
        }
      }
      return new FixedHeteroQuantizedSpikingDistributed(
          Integer.parseInt(params.get("nSignals")),
          valueToSpikeTrainConverter,
          spikeTrainToValueConverter
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
    if ((params = params(fixedPhasesAndFrequencies, name)) != null) {
      return new FixedPhaseAndFrequencyValues(
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
    if ((params = params(sensorCentralized, name)) != null) {
      return new SensorCentralized()
          .compose(PrototypedFunctionBuilder.of(List.of(
              new MLP(2d, 3, MultiLayerPerceptron.ActivationFunction.SIN),
              new MLP(1.5d, Integer.parseInt(params.get("nLayers")))
          )))
          .compose(PrototypedFunctionBuilder.merger());
    }
    //function mappers
    if ((params = params(mlp, name)) != null) {
      return new MLP(
          Double.parseDouble(params.get("ratio")),
          Integer.parseInt(params.get("nLayers")),
          params.containsKey("actFun") ? MultiLayerPerceptron.ActivationFunction.valueOf(params.get("actFun").toUpperCase()) : MultiLayerPerceptron.ActivationFunction.TANH
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
    if ((params = params(msn, name)) != null || (params = params(msnWithConverter, name)) != null || (params = params(learningMsnWithConverter, name)) != null) {
      BiFunction<Integer, Integer, SpikingFunction> neuronBuilder = null;
      if (params.containsKey("spikeType") && params.get("spikeType").equals("lif")) {
        if (params.containsKey("lRestPot") && params.containsKey("lThreshPot") && params.containsKey("lambda")) {
          double restingPotential = Double.parseDouble(params.get("lRestPot"));
          double thresholdPotential = Double.parseDouble(params.get("lThreshPot"));
          double lambda = Double.parseDouble(params.get("lambda"));
          neuronBuilder = (l, n) -> new LIFNeuron(restingPotential, thresholdPotential, lambda);
        } else {
          neuronBuilder = (l, n) -> new LIFNeuron();
        }
      }
      if (params.containsKey("spikeType") && params.get("spikeType").equals("lif_h")) {
        if (params.containsKey("lRestPot") && params.containsKey("lThreshPot") && params.containsKey("lambda") && params.containsKey("theta")) {
          double restingPotential = Double.parseDouble(params.get("lRestPot"));
          double thresholdPotential = Double.parseDouble(params.get("lThreshPot"));
          double lambda = Double.parseDouble(params.get("lambda"));
          double theta = Double.parseDouble(params.get("theta"));
          neuronBuilder = (l, n) -> new LIFNeuronWithHomeostasis(restingPotential, thresholdPotential, lambda, theta);
        } else {
          neuronBuilder = (l, n) -> new LIFNeuronWithHomeostasis();
        }
      }
      if (params.containsKey("spikeType") && params.get("spikeType").equals("lif_h_output")) {
        int outputLayerIndex = Integer.parseInt(params.get("nLayers")) + 1;
        if (params.containsKey("lRestPot") && params.containsKey("lThreshPot") && params.containsKey("lambda") && params.containsKey("theta")) {
          double restingPotential = Double.parseDouble(params.get("lRestPot"));
          double thresholdPotential = Double.parseDouble(params.get("lThreshPot"));
          double lambda = Double.parseDouble(params.get("lambda"));
          double theta = Double.parseDouble(params.get("theta"));
          neuronBuilder = (l, n) -> (l == outputLayerIndex) ? new LIFNeuronWithHomeostasis(restingPotential, thresholdPotential, lambda, theta) : new LIFNeuron(restingPotential, thresholdPotential, lambda);
        } else {
          neuronBuilder = (l, n) -> (l == outputLayerIndex) ? new LIFNeuronWithHomeostasis() : new LIFNeuron();
        }
      }
      if (params.containsKey("spikeType") && params.get("spikeType").equals("lif_h_io")) {
        int outputLayerIndex = Integer.parseInt(params.get("nLayers")) + 1;
        if (params.containsKey("lRestPot") && params.containsKey("lThreshPot") && params.containsKey("lambda") && params.containsKey("theta")) {
          double restingPotential = Double.parseDouble(params.get("lRestPot"));
          double thresholdPotential = Double.parseDouble(params.get("lThreshPot"));
          double lambda = Double.parseDouble(params.get("lambda"));
          double theta = Double.parseDouble(params.get("theta"));
          neuronBuilder = (l, n) -> (l == outputLayerIndex || l == 0) ? new LIFNeuronWithHomeostasis(restingPotential, thresholdPotential, lambda, theta) : new LIFNeuron(restingPotential, thresholdPotential, lambda);
        } else {
          neuronBuilder = (l, n) -> (l == outputLayerIndex || l == 0) ? new LIFNeuronWithHomeostasis() : new LIFNeuron();
        }
      }
      if (params.containsKey("spikeType") && params.get("spikeType").equals("iz")) {
        if (params.containsKey("izParams")) {
          IzhikevicNeuron.IzhikevicParameters izhikevicParameters = IzhikevicNeuron.IzhikevicParameters.valueOf(params.get("izParams").toUpperCase());
          neuronBuilder = (l, n) -> new IzhikevicNeuron(izhikevicParameters);
        } else {
          neuronBuilder = (l, n) -> new IzhikevicNeuron();
        }
      }
      if ((params = params(msn, name)) != null) {
        return new MSN(
            Double.parseDouble(params.get("ratio")),
            Integer.parseInt(params.get("nLayers")),
            neuronBuilder
        );
      }
      if ((params = params(msnWithConverter, name)) != null || (params = params(learningMsnWithConverter, name)) != null) {
        ValueToSpikeTrainConverter valueToSpikeTrainConverter = new UniformValueToSpikeTrainConverter();
        SpikeTrainToValueConverter spikeTrainToValueConverter = new AverageFrequencySpikeTrainToValueConverter();
        if (params.containsKey("iConv")) {
          switch (params.get("iConv")) {
            case "unif":
              if (params.containsKey("iFreq")) {
                valueToSpikeTrainConverter = new UniformValueToSpikeTrainConverter(Double.parseDouble(params.get("iFreq")));
              } else {
                valueToSpikeTrainConverter = new UniformValueToSpikeTrainConverter();
              }
              break;
            case "unif_mem":
              if (params.containsKey("iFreq")) {
                valueToSpikeTrainConverter = new UniformWithMemoryValueToSpikeTrainConverter(Double.parseDouble(params.get("iFreq")));
              } else {
                valueToSpikeTrainConverter = new UniformWithMemoryValueToSpikeTrainConverter();
              }
              break;
          }
        }
        if (params.containsKey("oConv")) {
          switch (params.get("oConv")) {
            case "avg":
              if (params.containsKey("oFreq")) {
                spikeTrainToValueConverter = new AverageFrequencySpikeTrainToValueConverter(Double.parseDouble(params.get("oFreq")));
              } else {
                spikeTrainToValueConverter = new AverageFrequencySpikeTrainToValueConverter();
              }
              break;
            case "avg_mem":
              if (params.containsKey("oFreq")) {
                if (params.containsKey("oMem")) {
                  spikeTrainToValueConverter = new MovingAverageSpikeTrainToValueConverter(Double.parseDouble(params.get("oFreq")),
                      Integer.parseInt(params.get("oMem")));
                } else {
                  spikeTrainToValueConverter = new MovingAverageSpikeTrainToValueConverter(Double.parseDouble(params.get("oFreq")));
                }
              } else {
                if (params.containsKey("oMem")) {
                  spikeTrainToValueConverter = new MovingAverageSpikeTrainToValueConverter(Integer.parseInt(params.get("oMem")));
                } else {
                  spikeTrainToValueConverter = new MovingAverageSpikeTrainToValueConverter();
                }
              }
              break;
          }
        }
        if ((params = params(msnWithConverter, name)) != null)
          return new MSNWithConverter(
              Double.parseDouble(params.get("ratio")),
              Integer.parseInt(params.get("nLayers")),
              neuronBuilder,
              valueToSpikeTrainConverter,
              spikeTrainToValueConverter
          );
        if ((params = params(learningMsnWithConverter, name)) != null) {
          return new LearningMSNWithConverters(
              Double.parseDouble(params.get("ratio")),
              Integer.parseInt(params.get("nLayers")),
              neuronBuilder,
              valueToSpikeTrainConverter,
              spikeTrainToValueConverter
          );
        }
      }
    }


    if ((params = params(quantizedMsn, name)) != null ||
        (params = params(quantizedMsnWithConverter, name)) != null ||
        (params = params(quantizedNumericLearningMsnWithConverter, name)) != null ||
        (params = params(quantizedNumericLearningFixedPoolMsnWithConverter, name)) != null ||
        (params = params(quantizedHebbianNumericLearningMsnWithConverter, name)) != null||
        (params = params(quantizedHebbianNumericLearningWeightsMSNWithConverters, name)) != null ||
        (params = params(quantizedHebbianNumericLearningClippedWeightsMSNWithConverters, name)) != null ||
        (params = params(quantizedNumericLearningWithFixedRuleValuesAnd0WeightsMSNWithConverters, name)) != null
    ) {
      BiFunction<Integer, Integer, QuantizedSpikingFunction> neuronBuilder = null;
      if (params.containsKey("spikeType") && params.get("spikeType").equals("lif")) {
        if (params.containsKey("lRestPot") && params.containsKey("lThreshPot") && params.containsKey("lambda")) {
          double restingPotential = Double.parseDouble(params.get("lRestPot"));
          double thresholdPotential = Double.parseDouble(params.get("lThreshPot"));
          double lambda = Double.parseDouble(params.get("lambda"));
          neuronBuilder = (l, n) -> new QuantizedLIFNeuron(restingPotential, thresholdPotential, lambda);
        } else {
          neuronBuilder = (l, n) -> new QuantizedLIFNeuron();
        }
      }
      if (params.containsKey("spikeType") && params.get("spikeType").equals("lif_h")) {
        if (params.containsKey("lRestPot") && params.containsKey("lThreshPot") && params.containsKey("lambda") && params.containsKey("theta")) {
          double restingPotential = Double.parseDouble(params.get("lRestPot"));
          double thresholdPotential = Double.parseDouble(params.get("lThreshPot"));
          double lambda = Double.parseDouble(params.get("lambda"));
          double theta = Double.parseDouble(params.get("theta"));
          neuronBuilder = (l, n) -> new QuantizedLIFNeuronWithHomeostasis(restingPotential, thresholdPotential, lambda, theta);
        } else {
          neuronBuilder = (l, n) -> new QuantizedLIFNeuronWithHomeostasis();
        }
      }
      if (params.containsKey("spikeType") && params.get("spikeType").equals("lif_h_output")) {
        int outputLayerIndex = Integer.parseInt(params.get("nLayers")) + 1;
        if (params.containsKey("lRestPot") && params.containsKey("lThreshPot") && params.containsKey("lambda") && params.containsKey("theta")) {
          double restingPotential = Double.parseDouble(params.get("lRestPot"));
          double thresholdPotential = Double.parseDouble(params.get("lThreshPot"));
          double lambda = Double.parseDouble(params.get("lambda"));
          double theta = Double.parseDouble(params.get("theta"));
          neuronBuilder = (l, n) -> (l == outputLayerIndex) ? new QuantizedLIFNeuronWithHomeostasis(restingPotential, thresholdPotential, lambda, theta) : new QuantizedLIFNeuron(restingPotential, thresholdPotential, lambda);
        } else {
          neuronBuilder = (l, n) -> (l == outputLayerIndex) ? new QuantizedLIFNeuronWithHomeostasis() : new QuantizedLIFNeuron();
        }
      }
      if (params.containsKey("spikeType") && params.get("spikeType").equals("lif_h_io")) {
        int outputLayerIndex = Integer.parseInt(params.get("nLayers")) + 1;
        if (params.containsKey("lRestPot") && params.containsKey("lThreshPot") && params.containsKey("lambda") && params.containsKey("theta")) {
          double restingPotential = Double.parseDouble(params.get("lRestPot"));
          double thresholdPotential = Double.parseDouble(params.get("lThreshPot"));
          double lambda = Double.parseDouble(params.get("lambda"));
          double theta = Double.parseDouble(params.get("theta"));
          neuronBuilder = (l, n) -> (l == outputLayerIndex || l == 0) ? new QuantizedLIFNeuronWithHomeostasis(restingPotential, thresholdPotential, lambda, theta) : new QuantizedLIFNeuron(restingPotential, thresholdPotential, lambda);
        } else {
          neuronBuilder = (l, n) -> (l == outputLayerIndex || l == 0) ? new QuantizedLIFNeuronWithHomeostasis() : new QuantizedLIFNeuron();
        }
      }
      if (params.containsKey("spikeType") && params.get("spikeType").equals("iz")) {
        if (params.containsKey("izParams")) {
          QuantizedIzhikevicNeuron.IzhikevicParameters izhikevicParameters = QuantizedIzhikevicNeuron.IzhikevicParameters.valueOf(params.get("izParams").toUpperCase());
          neuronBuilder = (l, n) -> new QuantizedIzhikevicNeuron(izhikevicParameters);
        } else {
          neuronBuilder = (l, n) -> new QuantizedIzhikevicNeuron();
        }
      }
      if ((params = params(quantizedMsn, name)) != null) {
        return new QuantizedMSN(
            Double.parseDouble(params.get("ratio")),
            Integer.parseInt(params.get("nLayers")),
            neuronBuilder
        );
      }
      if ((params = params(quantizedMsnWithConverter, name)) != null
          || (params = params(quantizedNumericLearningMsnWithConverter, name)) != null ||
          (params = params(quantizedNumericLearningFixedPoolMsnWithConverter, name)) != null ||
          (params = params(quantizedHebbianNumericLearningMsnWithConverter, name)) != null||
          (params = params(quantizedHebbianNumericLearningWeightsMSNWithConverters, name)) != null ||
          (params = params(quantizedHebbianNumericLearningClippedWeightsMSNWithConverters, name)) != null ||
          (params = params(quantizedNumericLearningWithFixedRuleValuesAnd0WeightsMSNWithConverters, name)) != null ) {
        QuantizedValueToSpikeTrainConverter valueToSpikeTrainConverter = new QuantizedUniformValueToSpikeTrainConverter();
        QuantizedSpikeTrainToValueConverter spikeTrainToValueConverter = new QuantizedAverageFrequencySpikeTrainToValueConverter();
        if (params.containsKey("iConv")) {
          switch (params.get("iConv")) {
            case "unif":
              if (params.containsKey("iFreq")) {
                valueToSpikeTrainConverter = new QuantizedUniformValueToSpikeTrainConverter(Double.parseDouble(params.get("iFreq")));
              } else {
                valueToSpikeTrainConverter = new QuantizedUniformValueToSpikeTrainConverter();
              }
              break;
            case "unif_mem":
              if (params.containsKey("iFreq")) {
                valueToSpikeTrainConverter = new QuantizedUniformWithMemoryValueToSpikeTrainConverter(Double.parseDouble(params.get("iFreq")));
              } else {
                valueToSpikeTrainConverter = new QuantizedUniformWithMemoryValueToSpikeTrainConverter();
              }
              break;
          }
        }
        if (params.containsKey("oConv")) {
          switch (params.get("oConv")) {
            case "avg":
              if (params.containsKey("oFreq")) {
                spikeTrainToValueConverter = new QuantizedAverageFrequencySpikeTrainToValueConverter(Double.parseDouble(params.get("oFreq")));
              } else {
                spikeTrainToValueConverter = new QuantizedAverageFrequencySpikeTrainToValueConverter();
              }
              break;
            case "avg_mem":
              if (params.containsKey("oFreq")) {
                if (params.containsKey("oMem")) {
                  spikeTrainToValueConverter = new QuantizedMovingAverageSpikeTrainToValueConverter(Double.parseDouble(params.get("oFreq")),
                      Integer.parseInt(params.get("oMem")));
                } else {
                  spikeTrainToValueConverter = new QuantizedMovingAverageSpikeTrainToValueConverter(Double.parseDouble(params.get("oFreq")));
                }
              } else {
                if (params.containsKey("oMem")) {
                  spikeTrainToValueConverter = new QuantizedMovingAverageSpikeTrainToValueConverter(Integer.parseInt(params.get("oMem")));
                } else {
                  spikeTrainToValueConverter = new QuantizedMovingAverageSpikeTrainToValueConverter();
                }
              }
              break;
          }
        }
        if ((params = params(quantizedMsnWithConverter, name)) != null) {
          return new QuantizedMSNWithConverters(
              Double.parseDouble(params.get("ratio")),
              Integer.parseInt(params.get("nLayers")),
              neuronBuilder,
              valueToSpikeTrainConverter,
              spikeTrainToValueConverter
          );
        }
        if ((params = params(quantizedNumericLearningMsnWithConverter, name)) != null) {
          return new QuantizedNumericLearningMSNWithConverters(
              Double.parseDouble(params.get("ratio")),
              Integer.parseInt(params.get("nLayers")),
              neuronBuilder,
              valueToSpikeTrainConverter,
              spikeTrainToValueConverter
          );
        }
        if ((params = params(quantizedHebbianNumericLearningMsnWithConverter, name)) != null) {
          return new QuantizedHebbianNumericLearningMSNWithConverters(
              Double.parseDouble(params.get("ratio")),
              Integer.parseInt(params.get("nLayers")),
              neuronBuilder,
              valueToSpikeTrainConverter,
              spikeTrainToValueConverter
          );
        }
        if ((params = params(quantizedNumericLearningFixedPoolMsnWithConverter, name)) != null) {
          return new QuantizedNumericLearningFixedPoolMSNWithConverters(
              Integer.parseInt(params.get("nRules")),
              Double.parseDouble(params.get("ratio")),
              Integer.parseInt(params.get("nLayers")),
              neuronBuilder,
              valueToSpikeTrainConverter,
              spikeTrainToValueConverter
          );
        }
        if ((params = params(quantizedHebbianNumericLearningWeightsMSNWithConverters, name)) != null) {
          return new QuantizedHebbianNumericLearningWeightsMSNWithConverters(
              Double.parseDouble(params.get("ratio")),
              Integer.parseInt(params.get("nLayers")),
              neuronBuilder,
              valueToSpikeTrainConverter,
              spikeTrainToValueConverter
          );
        }
        if ((params = params(quantizedHebbianNumericLearningClippedWeightsMSNWithConverters, name)) != null) {
          return new QuantizedHebbianNumericLearningClippedWeightsMSNWithConverters(
              Double.parseDouble(params.get("ratio")),
              Integer.parseInt(params.get("nLayers")),
              neuronBuilder,
              Double.parseDouble(params.get("maxWeight")),
              valueToSpikeTrainConverter,
              spikeTrainToValueConverter
          );
        }
        if ((params = params(quantizedNumericLearningWithFixedRuleValuesAnd0WeightsMSNWithConverters, name)) != null) {
          double[] symmetricParams = {Double.parseDouble(params.get("symmParam1")), Double.parseDouble(params.get("symmParam2")),
              Double.parseDouble(params.get("symmParam3")), Double.parseDouble(params.get("symmParam4"))};
          double[] asymmetricParams = {Double.parseDouble(params.get("asymmParam1")), Double.parseDouble(params.get("asymmParam2")),
              Double.parseDouble(params.get("asymmParam3")), Double.parseDouble(params.get("asymmParam4"))};
          return new QuantizedNumericLearningWithFixedRuleValuesAnd0WeightsMSNWithConverters(
              Double.parseDouble(params.get("ratio")),
              Integer.parseInt(params.get("nLayers")),
              neuronBuilder,
              valueToSpikeTrainConverter,
              spikeTrainToValueConverter,
              symmetricParams,
              asymmetricParams
          );
        }
      }
    }

    if ((params = params(fgraph, name)) != null) {
      return new FGraph();
    }
    //misc
    if ((params = params(functionGrid, name)) != null) {
      return new FunctionGrid((PrototypedFunctionBuilder) getMapperBuilderFromName(params.get("innerMapper")));
    }
    if ((params = params(spikingFunctionGrid, name)) != null) {
      return new FunctionGrid((PrototypedFunctionBuilder) getMapperBuilderFromName(params.get("innerMapper")));
    }
    if ((params = params(spikingQuantizedFunctionGrid, name)) != null) {
      return new SpikingQuantizedFunctionGrid((PrototypedFunctionBuilder) getMapperBuilderFromName(params.get("innerMapper")));
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
      transformation = new SequentialFunction<>(getSequence(transformationSequenceName).entrySet().stream()
          .collect(Collectors.toMap(
              Map.Entry::getKey,
              e -> RobotUtils.buildRobotTransformation(e.getValue(), random)
              )
          ));
    } else {
      transformation = RobotUtils.buildRobotTransformation(transformationSequenceName, random);
    }
    //terrains
    Function<Robot<?>, Outcome> task;
    if (terrainSequenceName.contains(SEQUENCE_SEPARATOR_CHAR)) {
      task = new SequentialFunction<>(getSequence(terrainSequenceName).entrySet().stream()
          .collect(Collectors.toMap(
              Map.Entry::getKey,
              e -> buildLocomotionTask(e.getValue(), episodeT, random)
              )
          ));
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

  public static SortedMap<Long, String> getSequence(String sequenceName) {
    return new TreeMap<>(Arrays.stream(sequenceName.split(SEQUENCE_SEPARATOR_CHAR)).collect(Collectors.toMap(
        s -> s.contains(SEQUENCE_ITERATION_CHAR) ? Long.parseLong(s.split(SEQUENCE_ITERATION_CHAR)[0]) : 0,
        s -> s.contains(SEQUENCE_ITERATION_CHAR) ? s.split(SEQUENCE_ITERATION_CHAR)[1] : s
    )));
  }

}
