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
import it.units.erallab.hmsrobots.core.objects.Robot;
import it.units.erallab.hmsrobots.core.objects.SensingVoxel;
import it.units.erallab.hmsrobots.tasks.locomotion.Footprint;
import it.units.erallab.hmsrobots.tasks.locomotion.Locomotion;
import it.units.erallab.hmsrobots.tasks.locomotion.Outcome;
import it.units.erallab.hmsrobots.util.Grid;
import it.units.erallab.hmsrobots.util.Point2;
import it.units.erallab.hmsrobots.util.SerializationUtils;
import it.units.erallab.mapper.PrototypedFunctionBuilder;
import it.units.erallab.mapper.evolver.CMAES;
import it.units.erallab.mapper.evolver.DoublesSpeciated;
import it.units.erallab.mapper.evolver.DoublesStandard;
import it.units.erallab.mapper.evolver.EvolverBuilder;
import it.units.erallab.mapper.phenotype.FGraph;
import it.units.erallab.mapper.phenotype.MLP;
import it.units.erallab.mapper.robot.*;
import it.units.malelab.jgea.Worker;
import it.units.malelab.jgea.core.Individual;
import it.units.malelab.jgea.core.evolver.Evolver;
import it.units.malelab.jgea.core.evolver.stopcondition.Births;
import it.units.malelab.jgea.core.listener.FileListenerFactory;
import it.units.malelab.jgea.core.listener.Listener;
import it.units.malelab.jgea.core.listener.ListenerFactory;
import it.units.malelab.jgea.core.listener.collector.*;
import it.units.malelab.jgea.core.util.Misc;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVPrinter;
import org.dyn4j.dynamics.Settings;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.*;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.TimeUnit;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static it.units.erallab.hmsrobots.util.Utils.*;
import static it.units.malelab.jgea.core.util.Args.*;

/**
 * @author eric
 */
public class LocomotionEvolution extends Worker {

  public static final int CACHE_SIZE = 1000;

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
    List<String> terrainNames = l(a("terrain", "flat"));
    List<String> targetShapeNames = l(a("shape", "biped-4x2"));
    List<String> targetSensorConfigNames = l(a("sensorConfig", "uniform"));
    List<String> transformationNames = l(a("transformations", "identity"));
    List<String> evolverMapperNames = l(a("evolver", "CMAES"));
    List<String> robotMapperNames = l(a("robotMapper", "fixedHomoDist-2"));
    List<String> phenotypeMapperNames = l(a("phenotypeMapper", "MLP-2"));
    Function<Outcome, Double> fitnessFunction = Outcome::getVelocity;
    Function<Outcome, List<Item>> outcomeTransformer = new OutcomeItemizer(
        episodeTransientTime,
        episodeTime,
        spectrumMinFreq,
        spectrumMaxFreq,
        spectrumSize
    );
    List<String> validationOutcomeHeaders = outcomeTransformer.apply(prototypeOutcome()).stream().map(Item::getName).collect(Collectors.toList());
    List<String> validationTransformationNames = l(a("validationTransformations", "")).stream().filter(s -> !s.isEmpty()).collect(Collectors.toList());
    List<String> validationTerrainNames = l(a("validationTerrains", "flat")).stream().filter(s -> !s.isEmpty()).collect(Collectors.toList());
    if (!validationTerrainNames.isEmpty() && validationTransformationNames.isEmpty()) {
      validationTransformationNames.add("identity");
    }
    if (validationTerrainNames.isEmpty() && !validationTransformationNames.isEmpty()) {
      validationTerrainNames.add(terrainNames.get(0));
    }
    //prepare file listeners
    String statsFileName = a("statsFile", null) == null ? null : a("dir", ".") + File.separator + a("statsFile", null);
    String serializedFileName = a("serializedFile", null) == null ? null : a("dir", ".") + File.separator + a("serializedFile", null);
    ListenerFactory<Object, Robot<?>, Double> statsListenerFactory = new FileListenerFactory<>(statsFileName);
    ListenerFactory<Object, Robot<?>, Double> serializedListenerFactory = new FileListenerFactory<>(serializedFileName);
    CSVPrinter validationPrinter;
    List<String> validationKeyHeaders = List.of("experiment.name", "seed", "terrain", "shape", "sensor.config", "phenotype.mapper", "robot.mapper", "transformation", "evolver");
    try {
      if (a("validationFile", null) != null) {
        validationPrinter = new CSVPrinter(new FileWriter(
            a("dir", ".") + File.separator + a("validationFile", null)
        ), CSVFormat.DEFAULT.withDelimiter(';'));
      } else {
        validationPrinter = new CSVPrinter(System.out, CSVFormat.DEFAULT.withDelimiter(';'));
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
    L.info("Evolvers: " + evolverMapperNames);
    L.info("Robot mappers: " + robotMapperNames);
    L.info("Phenotype mappers: " + phenotypeMapperNames);
    L.info("Shapes: " + targetShapeNames);
    L.info("Sensor configs: " + targetSensorConfigNames);
    L.info("Terrains: " + terrainNames);
    L.info("Transformations: " + transformationNames);
    L.info("Validations: " + Lists.cartesianProduct(validationTerrainNames, validationTransformationNames));
    //start iterations
    for (int seed : seeds) {
      for (String terrainName : terrainNames) {
        for (String targetShapeName : targetShapeNames) {
          for (String targetSensorConfigName : targetSensorConfigNames) {
            for (String robotMapperName : robotMapperNames) {
              for (String phenotypeMapperName : phenotypeMapperNames) {
                for (String transformationName : transformationNames) {
                  for (String evolverName : evolverMapperNames) {
                    final Random random = new Random(seed);
                    Map<String, String> keys = new TreeMap<>(Map.of(
                        "experiment.name", experimentName,
                        "seed", Integer.toString(seed),
                        "terrain", terrainName,
                        "shape", targetShapeName,
                        "sensor.config", targetSensorConfigName,
                        "robot.mapper", robotMapperName,
                        "phenotype.mapper", phenotypeMapperName,
                        "transformation", transformationName,
                        "evolver", evolverName
                    ));
                    Robot<?> target = new Robot<>(
                        null,
                        buildSensorizingFunction(targetSensorConfigName).apply(buildShape(targetShapeName))
                    );
                    //build training task
                    Function<Robot<?>, Outcome> trainingTask = it.units.erallab.hmsrobots.util.Utils.buildRobotTransformation(
                        transformationName.replace("rnd", Integer.toString(random.nextInt(10000)))
                    ).andThen(new Locomotion(
                        episodeTime,
                        Locomotion.createTerrain(terrainName),
                        physicsSettings
                    ));
                    if (CACHE_SIZE > 0) {
                      trainingTask = Misc.cached(trainingTask, CACHE_SIZE);
                    }
                    //build main data collectors for listener
                    List<DataCollector<?, ? super Robot<SensingVoxel>, ? super Double>> collectors = new ArrayList<DataCollector<?, ? super Robot<SensingVoxel>, ? super Double>>(List.of(
                        new Static(keys),
                        new Basic(),
                        new Population(),
                        new Diversity(),
                        new FitnessHistogram(),
                        new BestInfo("%5.3f"),
                        new FunctionOfOneBest<>(
                            ((Function<Individual<?, ? extends Robot<SensingVoxel>, ? extends Double>, Robot<SensingVoxel>>) Individual::getSolution)
                                .andThen(SerializationUtils::clone)
                                .andThen(trainingTask)
                                .andThen(outcomeTransformer)
                        )
                    ));
                    Listener<? super Object, ? super Robot<?>, ? super Double> listener;
                    if (statsFileName == null) {
                      listener = listener(collectors.toArray(DataCollector[]::new));
                    } else {
                      listener = statsListenerFactory.build(collectors.toArray(DataCollector[]::new));
                    }
                    if (serializedFileName != null) {
                      listener = serializedListenerFactory.build(
                          new Static(keys),
                          new Basic(),
                          new FunctionOfOneBest<>(i -> List.of(
                              new Item("serialized.robot", SerializationUtils.serialize(i.getSolution(), SerializationUtils.Mode.GZIPPED_JSON), "%s")
                          ))
                      ).then(listener);
                    }
                    try {
                      Evolver<?, Robot<?>, Double> evolver;
                      try {
                        PrototypedFunctionBuilder<?, Robot<?>> mapperBuilder = getRobotMapperBuilderFromName(robotMapperName);
                        if (getPhenotypeMapperBuilderFromName(phenotypeMapperName) != null) {
                          mapperBuilder = mapperBuilder.compose((PrototypedFunctionBuilder) getPhenotypeMapperBuilderFromName(phenotypeMapperName));
                        }
                        evolver = getEvolverBuilderFromName(evolverName).build(
                            (PrototypedFunctionBuilder) mapperBuilder,
                            target
                        );
                      } catch (ClassCastException | IllegalArgumentException e) {
                        L.warning(String.format(
                            "Cannot instantiate %s for %s: %s",
                            evolverName,
                            robotMapperName,
                            phenotypeMapperName
                        ));
                        continue;
                      }
                      //optimize
                      Stopwatch stopwatch = Stopwatch.createStarted();
                      L.info(String.format("Starting %s", keys));
                      Collection<Robot<?>> solutions = evolver.solve(
                          trainingTask.andThen(fitnessFunction),
                          new Births(nBirths),
                          random,
                          executorService,
                          Listener.onExecutor(
                              listener,
                              executorService
                          )
                      );
                      L.info(String.format("Done %s: %d solutions in %4ds",
                          keys,
                          solutions.size(),
                          stopwatch.elapsed(TimeUnit.SECONDS)
                      ));
                      //do validation
                      for (String validationTransformationName : validationTransformationNames) {
                        for (String validationTerrainName : validationTerrainNames) {
                          //build validation task
                          Function<Robot<?>, Outcome> validationTask = new Locomotion(
                              episodeTime,
                              Locomotion.createTerrain(validationTerrainName),
                              physicsSettings
                          );
                          validationTask = it.units.erallab.hmsrobots.util.Utils.buildRobotTransformation(validationTransformationName)
                              .andThen(SerializationUtils::clone)
                              .andThen(validationTask);
                          Outcome validationOutcome = validationTask.apply(solutions.stream().findFirst().get());
                          L.info(String.format(
                              "Validation %s/%s of \"first\" best done",
                              validationTransformationName,
                              validationTerrainName
                          ));
                          try {
                            List<Object> values = new ArrayList<>();
                            values.addAll(validationKeyHeaders.stream().map(keys::get).collect(Collectors.toList()));
                            values.addAll(List.of(validationTransformationName, validationTerrainName));
                            List<Item> validationItems = outcomeTransformer.apply(validationOutcome);
                            values.addAll(validationOutcomeHeaders.stream()
                                .map(n -> validationItems.stream()
                                    .filter(i -> i.getName().equals(n))
                                    .map(Item::getValue)
                                    .findFirst()
                                    .orElse(null))
                                .collect(Collectors.toList())
                            );
                            validationPrinter.printRecord(values);
                            validationPrinter.flush();
                          } catch (IOException e) {
                            L.severe(String.format("Cannot save validation results due to %s", e));
                          }
                        }
                      }
                    } catch (InterruptedException | ExecutionException e) {
                      L.severe(String.format("Cannot complete %s due to %s",
                          keys,
                          e
                      ));
                      e.printStackTrace();
                    }
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
    } catch (IOException e) {
      L.severe(String.format("Cannot close printer for validation results due to %s", e));
    }
  }

  private static Outcome prototypeOutcome() {
    double dT = new Settings().getStepFrequency();
    return new Outcome(
        0d, 0d, 10d, 0, 0d, 0d,
        new TreeMap<>(IntStream.range(0, (int) Math.round(10d / dT)).boxed().collect(Collectors.toMap(
            i -> (double) i * dT,
            i -> Point2.build(Math.sin((double) i / dT), Math.sin((double) i * dT / 5d))
        ))),
        new TreeMap<>(IntStream.range(0, (int) Math.round(10d / dT)).boxed().collect(Collectors.toMap(
            i -> (double) i * dT,
            i -> new Footprint(new boolean[]{true, false, true}))
        )),
        new TreeMap<>(Map.of(0d, Grid.create(1, 1, true)))
    );
  }

  private EvolverBuilder<?> getEvolverBuilderFromName(String name) {
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

  private PrototypedFunctionBuilder<?, Robot<?>> getRobotMapperBuilderFromName(String name) {
    String fixedCentralized = "fixedCentralized";
    String fixedHomoDistributed = "fixedHomoDist-(?<nSignals>\\d+)";
    String fixedHeteroDistributed = "fixedHeteroDist-(?<nSignals>\\d+)";
    String fixedPhasesFunction = "fixedPhasesFunct-(?<f>\\d+)";
    String fixedPhases = "fixedPhases-(?<f>\\d+)";
    Map<String, String> params;
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
    throw new IllegalArgumentException(String.format("Unknown robot mapper name: %s", name));
  }

  private PrototypedFunctionBuilder<?, ?> getPhenotypeMapperBuilderFromName(String name) {
    String mlp = "MLP-(?<nLayers>\\d+)";
    String fgraph = "fGraph";
    String identity = "identity";
    Map<String, String> params;
    if ((params = params(mlp, name)) != null) {
      return new MLP(
          0.65d,
          Integer.parseInt(params.get("nLayers"))
      );
    }
    if ((params = params(identity, name)) != null) {
      return null;
    }
    if ((params = params(fgraph, name)) != null) {
      return new FGraph();
    }
    throw new IllegalArgumentException(String.format("Unknown phenotype mapper name: %s", name));
  }

}
