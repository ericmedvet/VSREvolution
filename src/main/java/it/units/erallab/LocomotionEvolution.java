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
import it.units.erallab.builder.PrototypedFunctionBuilder;
import it.units.erallab.builder.evolver.CMAES;
import it.units.erallab.builder.evolver.DoublesSpeciated;
import it.units.erallab.builder.evolver.DoublesStandard;
import it.units.erallab.builder.evolver.EvolverBuilder;
import it.units.erallab.builder.phenotype.FGraph;
import it.units.erallab.builder.phenotype.FunctionGrid;
import it.units.erallab.builder.phenotype.MLP;
import it.units.erallab.builder.robot.*;
import it.units.erallab.hmsrobots.core.controllers.MultiLayerPerceptron;
import it.units.erallab.hmsrobots.core.objects.Robot;
import it.units.erallab.hmsrobots.tasks.locomotion.Footprint;
import it.units.erallab.hmsrobots.tasks.locomotion.Locomotion;
import it.units.erallab.hmsrobots.tasks.locomotion.Outcome;
import it.units.erallab.hmsrobots.util.Grid;
import it.units.erallab.hmsrobots.util.Point2;
import it.units.erallab.hmsrobots.util.RobotUtils;
import it.units.erallab.hmsrobots.util.SerializationUtils;
import it.units.malelab.jgea.Worker;
import it.units.malelab.jgea.core.Individual;
import it.units.malelab.jgea.core.evolver.Evolver;
import it.units.malelab.jgea.core.evolver.stopcondition.Births;
import it.units.malelab.jgea.core.listener.FileListenerFactory;
import it.units.malelab.jgea.core.listener.Listener;
import it.units.malelab.jgea.core.listener.ListenerFactory;
import it.units.malelab.jgea.core.listener.collector.*;
import it.units.malelab.jgea.core.order.PartialComparator;
import it.units.malelab.jgea.core.util.Misc;
import it.units.malelab.jgea.core.util.SequentialFunction;
import it.units.malelab.jgea.core.util.TextPlotter;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVPrinter;
import org.dyn4j.dynamics.Settings;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintStream;
import java.util.*;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.TimeUnit;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static it.units.erallab.hmsrobots.util.Utils.params;
import static it.units.malelab.jgea.core.util.Args.*;

/**
 * @author eric
 */
public class LocomotionEvolution extends Worker {

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
    int spectrumSize = 20;
    double spectrumMinFreq = 0d;
    double spectrumMaxFreq = 5d;
    Settings physicsSettings = new Settings();
    double episodeTime = d(a("episodeTime", "10"));
    double episodeTransientTime = d(a("episodeTransientTime", "5"));
    int nBirths = i(a("nBirths", "500"));
    int[] seeds = ri(a("seed", "0:1"));
    String experimentName = a("expName", "short");
    List<String> terrainNames = l(a("terrain", "hilly-1-10-0"));
    List<String> targetShapeNames = l(a("shape", "box-4x4"));
    List<String> targetSensorConfigNames = l(a("sensorConfig", "uniform-t+ax+ay+r+l1+a-0"));
    List<String> transformationNames = l(a("transformation", "identity"));
    List<String> evolverNames = l(a("evolver", "CMAES"));
    List<String> mapperNames = l(a("mapper", "sensorAndBodyAndHomoDist-50-3-3-t"));
    String statsFileName = a("statsFile", null) == null ? null : a("dir", ".") + File.separator + a("statsFile", null);
    boolean serialization = a("serialization", "false").startsWith("t");
    Function<Outcome, Double> fitnessFunction = Outcome::getVelocity;
    //collectors
    Function<Outcome, List<Item>> outcomeTransformer = new OutcomeItemizer(
        spectrumMinFreq,
        spectrumMaxFreq,
        spectrumSize
    );
    List<DataCollector<? super Object, ? super Robot<?>, ? super Outcome>> dataCollectors = List.of(
        new Basic(),
        new Population(),
        new Diversity(),
        new FunctionOfOneBest<>(i -> List.of(new Item("fitness", fitnessFunction.apply(i.getFitness()), "%5.3f"))),
        new Histogram<>(fitnessFunction.compose(Individual::getFitness), "fitness", 8),
        new FunctionOfOneBest<>(i -> List.of(new Item(
            "shape.minimap",
            TextPlotter.binaryMap(
                i.getSolution().getVoxels().toArray(Objects::nonNull),
                (int) Math.min(Math.ceil((float) i.getSolution().getVoxels().getW() / (float) i.getSolution().getVoxels().getH() * 2f), 4)
            ),
            "%4s"
        ))),
        new FunctionOfOneBest<>(i -> List.of(new Item(
            "shape.size",
            i.getSolution().getVoxels().getW() + "x" + i.getSolution().getVoxels().getH(),
            "%3s"
        ))),
        new FunctionOfOneBest<>(i -> List.of(new Item(
            "shape.num.voxel",
            i.getSolution().getVoxels().values().stream().filter(Objects::nonNull).count(),
            "%2d"
        ))),
        new Histogram<>(i -> i.getSolution().getVoxels().values().stream().filter(Objects::nonNull).count(), "shape.num.voxel", 4),
        new FunctionOfOneBest<>(i -> List.of(new Item(
            "average.posture.minimap",
            TextPlotter.binaryMap(i.getFitness().getAveragePosture().toArray(b -> b), 2),
            "%2s"
        ))),
        new FunctionOfOneBest<>(outcomeTransformer.compose(Individual::getFitness)),
        new FunctionOfOneBest<>(i -> List.of(new Item(
            "serialized.robot",
            serialization ? SerializationUtils.serialize(i.getSolution(), SerializationUtils.Mode.GZIPPED_JSON) : "",
            "%s"
        )))
    );
    //validation
    List<String> validationOutcomeHeaders = outcomeTransformer.apply(prototypeOutcome()).stream().map(Item::getName).collect(Collectors.toList());
    List<String> validationTransformationNames = l(a("validationTransformation", "")).stream().filter(s -> !s.isEmpty()).collect(Collectors.toList());
    List<String> validationTerrainNames = l(a("validationTerrain", "flat")).stream().filter(s -> !s.isEmpty()).collect(Collectors.toList());
    if (!validationTerrainNames.isEmpty() && validationTransformationNames.isEmpty()) {
      validationTransformationNames.add("identity");
    }
    if (validationTerrainNames.isEmpty() && !validationTransformationNames.isEmpty()) {
      validationTerrainNames.add(terrainNames.get(0));
    }
    //prepare file listeners
    ListenerFactory<Object, Robot<?>, Outcome> statsListenerFactory = new FileListenerFactory<>(statsFileName);
    CSVPrinter validationPrinter;
    List<String> validationKeyHeaders = List.of("experiment.name", "seed", "terrain", "shape", "sensor.config", "mapper", "transformation", "evolver");
    try {
      if (a("validationFile", null) != null) {
        validationPrinter = new CSVPrinter(new FileWriter(
            a("dir", ".") + File.separator + a("validationFile", null)
        ), CSVFormat.DEFAULT.withDelimiter(';'));
      } else {
        validationPrinter = new CSVPrinter(new PrintStream(PrintStream.nullOutputStream()), CSVFormat.DEFAULT.withDelimiter(';'));
      }
      List<String> headers = new ArrayList<>();
      headers.addAll(validationKeyHeaders);
      headers.addAll(List.of("validation.transformation", "validation.terrain"));
      headers.addAll(validationOutcomeHeaders.stream().map(n -> "validation." + n).collect(Collectors.toList()));
      validationPrinter.printRecord(headers);
    } catch (IOException e) {
      L.severe(String.format("Cannot create printer for validation results due to %s", e));
      return;
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
                  Map<String, String> keys = new TreeMap<>(Map.of(
                      "experiment.name", experimentName,
                      "seed", Integer.toString(seed),
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
                  //build main data collectors for listener
                  Listener<? super Object, ? super Robot<?>, ? super Outcome> listener;
                  if (statsFileName == null) {
                    listener = listener(dataCollectors);
                  } else {
                    listener = statsListenerFactory.build(Utils.concat(List.of(new Static(keys)), dataCollectors));
                  }
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
                        Listener.onExecutor(
                            listener,
                            executorService
                        )
                    );
                    L.info(String.format("Progress %s (%d/%d); Done: %d solutions in %4ds",
                        TextPlotter.horizontalBar(counter, 0, nOfRuns, 8),
                        counter, nOfRuns,
                        solutions.size(),
                        stopwatch.elapsed(TimeUnit.SECONDS)
                    ));
                    //do validation
                    Robot<?> bestSolution = solutions.stream().findFirst().orElse(null);
                    if (bestSolution != null) {
                      for (String validationTransformationName : validationTransformationNames) {
                        for (String validationTerrainName : validationTerrainNames) {
                          //build validation task
                          Function<Robot<?>, Outcome> validationTask = new Locomotion(
                              episodeTime,
                              Locomotion.createTerrain(validationTerrainName),
                              physicsSettings
                          );
                          validationTask = RobotUtils.buildRobotTransformation(validationTransformationName, new Random(0))
                              .andThen(SerializationUtils::clone)
                              .andThen(validationTask);
                          try {
                            Outcome validationOutcome = validationTask.apply(bestSolution);
                            L.info(String.format(
                                "Validation %s/%s of \"first\" best done in %ss",
                                validationTransformationName,
                                validationTerrainName,
                                validationOutcome.getComputationTime()
                            ));
                            List<Object> values = new ArrayList<>();
                            values.addAll(validationKeyHeaders.stream().map(keys::get).collect(Collectors.toList()));
                            values.addAll(List.of(validationTransformationName, validationTerrainName));
                            List<Item> validationItems = outcomeTransformer.apply(validationOutcome);
                            values.addAll(validationOutcomeHeaders.stream()
                                .map(n -> validationItems.stream()
                                    .filter(i -> i.getName().equals(n) && i.getValue() != null)
                                    .map(Item::getValue)
                                    .findFirst()
                                    .orElse(null))
                                .collect(Collectors.toList())
                            );
                            validationPrinter.printRecord(values);
                            validationPrinter.flush();
                          } catch (Throwable e) {
                            L.severe(String.format("Cannot do or save validation results due to %s", e));
                            e.printStackTrace(); // TODO possibly to be removed
                          }
                        }
                      }
                    } else {
                      L.warning("No solution, cannot do validation");
                    }
                  } catch (InterruptedException | ExecutionException e) {
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
    try {
      validationPrinter.close(true);
    } catch (
        IOException e) {
      L.severe(String.format("Cannot close printer for validation results due to %s", e));
    }

  }

  private static Outcome prototypeOutcome() {
    double dT = new Settings().getStepFrequency();
    return new Outcome(IntStream.range(0, 1000)
        .mapToObj(i -> new Outcome.Observation(
            (double) i * dT,
            Point2.build(Math.sin((double) i / dT), Math.sin((double) i * dT / 5d)),
            new Footprint(new boolean[]{true, false, true}),
            Grid.create(1, 1, true),
            0.1,
            0.1,
            (double) i * dT / 10d
        ))
        .collect(Collectors.toList())
    );
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
    String bodyAndHomoDistributed = "bodyAndHomoDist-(?<fullness>\\d+(\\.\\d+)?)-(?<nSignals>\\d+)-(?<nLayers>\\d+)";
    String sensorAndBodyAndHomoDistributed = "sensorAndBodyAndHomoDist-(?<fullness>\\d+(\\.\\d+)?)-(?<nSignals>\\d+)-(?<nLayers>\\d+)-(?<position>(t|f))";
    String mlp = "MLP-(?<nLayers>\\d+)(-(?<actFun>(sin|tanh|sigmoid|relu)))?";
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
    if ((params = params(fixedHomoDistributed, name)) != null) {
      return new FixedHomoDistributed(
          Integer.parseInt(params.get("nSignals"))
      );
    }
    //function mappers
    if ((params = params(mlp, name)) != null) {
      return new MLP(
          1d,
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
    throw new IllegalArgumentException(String.format("Unknown mapper name: %s", name));
  }

  @SuppressWarnings({"unchecked", "rawtypes"})
  private static Evolver<?, Robot<?>, Outcome> buildEvolver(String evolverName, String robotMapperName, Robot<?> target, Function<Outcome, Double> outcomeMeasure) {
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

  private static Function<Robot<?>, Outcome> buildTaskFromName(String transformationSequenceName, String terrainSequenceName, double episodeT, Random random) {
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

  private static Function<Robot<?>, Outcome> buildLocomotionTask(String terrainName, double episodeT, Random random) {
    if (!terrainName.contains("-rnd")) {
      return Misc.cached(new Locomotion(
          episodeT,
          Locomotion.createTerrain(terrainName),
          new Settings()
      ), CACHE_SIZE);
    }
    return r -> new Locomotion(
        episodeT,
        Locomotion.createTerrain(terrainName.replace("-rnd", "-" + random.nextInt(10000))),
        new Settings()
    ).apply(r);
  }

}
