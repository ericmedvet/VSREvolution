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
import it.units.erallab.hmsrobots.core.controllers.CentralizedSensing;
import it.units.erallab.hmsrobots.core.controllers.MultiLayerPerceptron;
import it.units.erallab.hmsrobots.core.controllers.PhaseSin;
import it.units.erallab.hmsrobots.core.objects.Robot;
import it.units.erallab.hmsrobots.core.objects.SensingVoxel;
import it.units.erallab.hmsrobots.tasks.locomotion.Footprint;
import it.units.erallab.hmsrobots.tasks.locomotion.Locomotion;
import it.units.erallab.hmsrobots.tasks.locomotion.Outcome;
import it.units.erallab.hmsrobots.util.Grid;
import it.units.erallab.hmsrobots.util.Point2;
import it.units.erallab.hmsrobots.util.SerializationUtils;
import it.units.erallab.mapper.PrototypedFunctionBuilder;
import it.units.erallab.mapper.evolver.EvolverBuilder;
import it.units.malelab.jgea.Worker;
import it.units.malelab.jgea.core.Individual;
import it.units.malelab.jgea.core.evolver.*;
import it.units.malelab.jgea.core.evolver.stopcondition.Births;
import it.units.malelab.jgea.core.fitness.SequencedFitness;
import it.units.malelab.jgea.core.listener.FileListenerFactory;
import it.units.malelab.jgea.core.listener.Listener;
import it.units.malelab.jgea.core.listener.ListnerFactory;
import it.units.malelab.jgea.core.listener.collector.*;
import it.units.malelab.jgea.core.order.PartialComparator;
import it.units.malelab.jgea.core.selector.Tournament;
import it.units.malelab.jgea.core.selector.Worst;
import it.units.malelab.jgea.core.util.Misc;
import it.units.malelab.jgea.core.util.Pair;
import it.units.malelab.jgea.distance.Jaccard;
import it.units.malelab.jgea.representation.graph.*;
import it.units.malelab.jgea.representation.graph.numeric.Output;
import it.units.malelab.jgea.representation.graph.numeric.functiongraph.BaseFunction;
import it.units.malelab.jgea.representation.graph.numeric.functiongraph.FunctionGraph;
import it.units.malelab.jgea.representation.graph.numeric.functiongraph.FunctionNode;
import it.units.malelab.jgea.representation.graph.numeric.functiongraph.ShallowSparseFactory;
import it.units.malelab.jgea.representation.sequence.FixedLengthListFactory;
import it.units.malelab.jgea.representation.sequence.numeric.GaussianMutation;
import it.units.malelab.jgea.representation.sequence.numeric.GeometricCrossover;
import it.units.malelab.jgea.representation.sequence.numeric.UniformDoubleFactory;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVPrinter;
import org.dyn4j.dynamics.Settings;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.*;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.TimeUnit;
import java.util.function.BiFunction;
import java.util.function.Function;
import java.util.function.Predicate;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static it.units.malelab.jgea.core.util.Args.*;

/**
 * @author eric
 * @created 2020/08/18
 * @project VSREvolution
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
    double episodeTime = d(a("episodeTime", "30"));
    double episodeTransientTime = d(a("episodeTransientTime", "5"));
    int nBirths = i(a("nBirths", "500"));
    int[] seeds = ri(a("seed", "0:1"));
    String experimentName = a("expName", "");
    List<String> terrainNames = l(a("terrain", "flat"));
    List<String> evolverMapperNames = l(a("evolver", "mlp-0.65-cmaes"));
    List<String> bodyNames = l(a("body", "biped-4x2-f-f"));
    List<String> transformationNames = l(a("transformations", "identity"));
    List<String> robotMapperNames = l(a("mapper", "centralized"));
    Function<Outcome, Double> fitnessFunction = new SequencedFitness<>(Map.of(
        1000L, Outcome::getVelocity,
        50000L, Outcome::getCorrectedEfficiency
    ));
    Function<Outcome, List<Item>> outcomeTransformer = o -> Utils.concat(
        List.of(
            new Item("computation.time", o.getComputationTime(), "%4.1f"),
            new Item("time", o.getTime(), "%4.1f"),
            new Item("area.ratio.power", o.getAreaRatioPower(), "%5.1f"),
            new Item("control.power", o.getControlPower(), "%5.1f"),
            new Item("corrected.efficiency", o.getCorrectedEfficiency(), "%6.3f"),
            new Item("distance", o.getDistance(), "%5.1f"),
            new Item("velocity", o.getVelocity(), "%6.3f"),
            new Item(
                "average.posture",
                Grid.toString(o.getAveragePosture(episodeTransientTime, episodeTransientTime), (Predicate<Boolean>) b -> b, "|"),
                "%10.10s"
            )
        ),
        Utils.ifThenElse(
            (Predicate<Outcome.Gait>) Objects::isNull,
            g -> new ArrayList<Item>(),
            g -> List.of(
                new Item("gait.average.touch.area", g.getAvgTouchArea(), "%5.3f"),
                new Item("gait.coverage", g.getCoverage(), "%4.2f"),
                new Item("gait.mode.interval", g.getModeInterval(), "%3.1f"),
                new Item("gait.purity", g.getPurity(), "%4.2f"),
                new Item("gait.num.unique.footprints", g.getFootprints().stream().distinct().count(), "%2d"),
                new Item("gait.num.footprints", g.getFootprints().size(), "%2d"),
                new Item("gait.footprints", g.getFootprints().stream().map(Footprint::toString).collect(Collectors.joining("|")), "%10.10s")
            )
        ).apply(o.getMainGait(episodeTransientTime, episodeTransientTime)),
        Utils.index(o.getCenterPowerSpectrum(episodeTransientTime, episodeTransientTime, Outcome.Component.X, spectrumMinFreq, spectrumMaxFreq, spectrumSize)).entrySet().stream()
            .sorted(Map.Entry.comparingByKey())
            .map(e -> List.of(
                new Item(String.format("spectrum.x.%d.f", e.getKey() + 1), e.getValue().getFrequency(), "%3.1f"),
                new Item(String.format("spectrum.x.%d.s", e.getKey() + 1), e.getValue().getStrength(), "%4.1f")
            ))
            .reduce(Utils::concat)
            .orElse(List.of()),
        Utils.index(o.getCenterPowerSpectrum(episodeTransientTime, episodeTransientTime, Outcome.Component.X, spectrumMinFreq, spectrumMaxFreq, spectrumSize)).entrySet().stream()
            .sorted(Map.Entry.comparingByKey())
            .map(e -> List.of(
                new Item(String.format("spectrum.y.%d.f", e.getKey() + 1), e.getValue().getFrequency(), "%3.1f"),
                new Item(String.format("spectrum.y.%d.s", e.getKey() + 1), e.getValue().getStrength(), "%4.1f")
            ))
            .reduce(Utils::concat)
            .orElse(List.of())
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
    ListnerFactory<Object, Robot<?>, Double> statsListenerFactory = new FileListenerFactory<>(statsFileName);
    ListnerFactory<Object, Robot<?>, Double> serializedListenerFactory = new FileListenerFactory<>(serializedFileName);
    CSVPrinter validationPrinter;
    List<String> validationKeyHeaders = List.of("seed", "terrain", "body", "mapper", "transformation", "evolver");
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
    L.info("Mappers: " + robotMapperNames);
    L.info("Bodies: " + bodyNames);
    L.info("Terrains: " + terrainNames);
    L.info("Transformations: " + transformationNames);
    L.info("Validations: " + Lists.cartesianProduct(validationTerrainNames, validationTransformationNames));
    //start iterations
    for (int seed : seeds) {
      for (String terrainName : terrainNames) {
        for (String bodyName : bodyNames) {
          for (String robotMapperName : robotMapperNames) {
            for (String transformationName : transformationNames) {
              for (String evolverName : evolverMapperNames) {
                final Random random = new Random(seed);
                Map<String, String> keys = new TreeMap<>(Map.of(
                    "experiment.name", experimentName,
                    "seed", Integer.toString(seed),
                    "terrain", terrainName,
                    "body", bodyName,
                    "mapper", robotMapperName,
                    "transformation", transformationName,
                    "evolver", evolverName
                ));
                Grid<? extends SensingVoxel> body = it.units.erallab.hmsrobots.util.Utils.buildBody(bodyName);
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
                  Stopwatch stopwatch = Stopwatch.createStarted();
                  L.info(String.format("Starting %s", keys));
                  Evolver<?, Robot<?>, Double> evolver = getEvolverBuilderFromName(evolverName).build(
                      getMapperBuilderFromName(robotMapperName),
                      new Robot<>(null, body)
                  );
                  getMapperBuilderFromName(robotMapperName).;
                  //optimize
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
    try {
      validationPrinter.close(true);
    } catch (IOException e) {
      L.severe(String.format("Cannot close printer for validation results due to %s", e));
    }
  }

  private static Outcome prototypeOutcome() {
    return new Outcome(
        0d, 0d, 10d, 0, 0d, 0d,
        new TreeMap<>(IntStream.range(0, 100).boxed().collect(Collectors.toMap(
            i -> (double) i / 10d,
            i -> Point2.build(Math.sin((double) i / 10d), Math.sin((double) i / 5d))
        ))),
        new TreeMap<>(IntStream.range(0, 100).boxed().collect(Collectors.toMap(
            i -> (double) i / 10d,
            i -> new Footprint(new boolean[]{true, false, true}))
        )),
        new TreeMap<>(Map.of(0d, Grid.create(1, 1, true)))
    );
  }

  private EvolverBuilder<?> getEvolverBuilderFromName(String name) {
    String numGA = "numGA-(?<nPop>\\d+)";
    String numGASpeciated = "numGASpec-(?<nPop>\\d+)-(?<dT>\\d+(\\.\\d+)?)";
    String cmaEs = "CMAES";
    //TODO
    return null;
  }

  private PrototypedFunctionBuilder<?,Robot<?>> getMapperBuilderFromName(String name) {
    String fixedCentralizedMLP = "fixedCentralized-MLP-(?<nLayers>\\d+)";
    String fixedHomoDistributedMLP = "fixedHomoDist-(?<nSignals>\\d+)-MLP-(?<nLayers>\\d+)";
    String fixedHeteroDistributedMLP = "fixedHeteroDist-(?<nSignals>\\d+)-MLP-(?<n>\\d+)";
    String fixedPhasesFunction = "fixedPhasesFunct-(?<f>\\d+)";
    String fixedPhases = "fixedPhases-(?<f>\\d+)";
    //TODO
    return null;
  }

}
