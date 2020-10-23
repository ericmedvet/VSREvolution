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
import it.units.malelab.jgea.Worker;
import it.units.malelab.jgea.core.Individual;
import it.units.malelab.jgea.core.evolver.*;
import it.units.malelab.jgea.core.evolver.stopcondition.Births;
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
import java.io.Serializable;
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
public class ControllerComparison extends Worker {

  public static final int CACHE_SIZE = 10000;

  public ControllerComparison(String[] args) {
    super(args);
  }

  public static void main(String[] args) {
    new ControllerComparison(args);
  }

  private interface IODimMapper extends Function<Grid<? extends SensingVoxel>, Pair<Integer, Integer>> {
  }

  private interface RobotMapper extends Function<Grid<? extends SensingVoxel>, Function<Function<double[], double[]>, Robot<?>>> {
  }

  private interface EvolverMapper extends BiFunction<Pair<IODimMapper, RobotMapper>, Grid<? extends SensingVoxel>, Evolver<?, Robot<?>, Double>> {
  }

  @Override
  public void run() {
    int nOfModes = 5;
    Settings physicsSettings = new Settings();
    double episodeTime = d(a("episodeT", "30"));
    int nBirths = i(a("nBirths", "500"));
    int[] seeds = ri(a("seed", "0:1"));
    List<String> terrainNames = l(a("terrain", "flat"));
    List<String> evolverMapperNames = l(a("evolver", "mlp-0.65-cmaes"));
    List<String> bodyNames = l(a("body", "biped-4x2-f-f"));
    List<String> transformationNames = l(a("transformations", "identity"));
    List<String> robotMapperNames = l(a("mapper", "centralized"));
    Function<Outcome, Double> fitnessFunction = Outcome::getVelocity;
    Function<Outcome, List<Item>> outcomeTransformer = o -> Utils.concat(
        List.of(
            new Item("area.ratio.power", o.getAreaRatioPower(), "%5.1f"),
            new Item("control.power", o.getControlPower(), "%5.1f"),
            new Item("corrected.efficiency", o.getCorrectedEfficiency(), "%6.3f"),
            new Item("distance", o.getDistance(), "%5.1f"),
            new Item("velocity", o.getVelocity(), "%6.3f"),
            new Item(
                "average.posture",
                Grid.toString(o.getAveragePosture(), (Predicate<Boolean>) b -> b, "|"),
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
        ).apply(o.getMainGait()),
        Utils.index(o.getCenterModes(Outcome.Component.X)).entrySet().stream()
            .sorted(Map.Entry.comparingByKey())
            .limit(nOfModes)
            .map(e -> List.of(
                new Item(String.format("mode.x.%d.f", e.getKey() + 1), e.getValue().getFrequency(), "%3.1f"),
                new Item(String.format("mode.x.%d.s", e.getKey() + 1), e.getValue().getStrength(), "%4.1f")
            ))
            .reduce(Utils::concat)
            .orElse(List.of()),
        Utils.index(o.getCenterModes(Outcome.Component.Y)).entrySet().stream()
            .sorted(Map.Entry.comparingByKey())
            .limit(nOfModes)
            .map(e -> List.of(
                new Item(String.format("mode.y.%d.f", e.getKey() + 1), e.getValue().getFrequency(), "%3.1f"),
                new Item(String.format("mode.y.%d.s", e.getKey() + 1), e.getValue().getStrength(), "%4.1f")
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
              for (String evolverMapperName : evolverMapperNames) {
                final Random random = new Random(seed);
                Map<String, String> keys = new TreeMap<>(Map.of(
                    "seed", Integer.toString(seed),
                    "terrain", terrainName,
                    "body", bodyName,
                    "mapper", robotMapperName,
                    "transformation", transformationName,
                    "evolver", evolverMapperName
                ));
                Grid<? extends SensingVoxel> body = it.units.erallab.hmsrobots.util.Utils.buildBody(bodyName);
                //build training task
                Function<Robot<?>, Outcome> trainingTask = Misc.cached(
                    it.units.erallab.hmsrobots.util.Utils.buildRobotTransformation(
                        transformationName.replace("rnd", Integer.toString(random.nextInt(10000)))
                    ).andThen(new Locomotion(
                        episodeTime,
                        Locomotion.createTerrain(terrainName),
                        physicsSettings
                    )), CACHE_SIZE);
                //build main data collectors for listener
                List<DataCollector<?, ? super Robot<SensingVoxel>, ? super Double>> collectors = new ArrayList<DataCollector<?, ? super Robot<SensingVoxel>, ? super Double>>(List.of(
                    new Static(keys),
                    new Basic(),
                    new Population(),
                    new Diversity(),
                    new BestInfo("%6.4f"),
                    new FunctionOfOneBest<>(
                        ((Function<Individual<?, ? extends Robot<SensingVoxel>, ? extends Double>, Robot<SensingVoxel>>) Individual::getSolution)
                            .andThen(org.apache.commons.lang3.SerializationUtils::clone)
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
                          new Item("fitness.value", i.getFitness(), "%7.5f"),
                          new Item("serialized.robot", Utils.safelySerialize(i.getSolution()), "%s"),
                          new Item("serialized.genotype", Utils.safelySerialize((Serializable) i.getGenotype()), "%s")
                      ))
                  ).then(listener);
                }
                try {
                  Stopwatch stopwatch = Stopwatch.createStarted();
                  L.info(String.format("Starting %s", keys));
                  Evolver<?, Robot<?>, Double> evolver = buildEvolverMapper(evolverMapperName).apply(buildRobotMapper(robotMapperName), body);
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
                          .andThen(org.apache.commons.lang3.SerializationUtils::clone)
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

  private static Pair<IODimMapper, RobotMapper> buildRobotMapper(String name) {
    String centralized = "centralized";
    String phases = "phases-(?<f>\\d+(\\.\\d+)?)";
    if (name.matches(centralized)) {
      return Pair.of(
          body -> Pair.of(CentralizedSensing.nOfInputs(body), CentralizedSensing.nOfOutputs(body)),
          body -> f -> new Robot<>(
              new CentralizedSensing(body, f),
              org.apache.commons.lang3.SerializationUtils.clone(body)
          )
      );
    }
    if (name.matches(phases)) {
      return Pair.of(
          body -> Pair.of(2, 1),
          body -> f -> new Robot<>(
              new PhaseSin(
                  -Double.parseDouble(it.units.erallab.hmsrobots.util.Utils.paramValue(phases, name, "f")),
                  1d,
                  Grid.create(
                      body.getW(),
                      body.getH(),
                      (x, y) -> f.apply(new double[]{(double) x / (double) body.getW(), (double) y / (double) body.getH()})[0]
                  )),
              org.apache.commons.lang3.SerializationUtils.clone(body)
          )
      );
    }
    throw new IllegalArgumentException(String.format("Unknown mapper name: %s", name));
  }

  private static EvolverMapper buildEvolverMapper(String name) {
    PartialComparator<Individual<?, Robot<?>, Double>> comparator = PartialComparator.from(Double.class).reversed().comparing(Individual::getFitness);
    String mlpGa = "mlp-(?<h>\\d+(\\.\\d+)?)-ga-(?<nPop>\\d+)";
    String mlpGaDiv = "mlp-(?<h>\\d+(\\.\\d+)?)-gadiv-(?<nPop>\\d+)";
    String mlpCmaEs = "mlp-(?<h>\\d+(\\.\\d+)?)-cmaes";
    String graphea = "fgraph-hash\\+-speciated-(?<nPop>\\d+)";
    String grapheaNoXOver = "fgraph-seq-noxover-(?<nPop>\\d+)";
    if (name.matches(mlpGa)) {
      double ratioOfFirstLayer = Double.parseDouble(it.units.erallab.hmsrobots.util.Utils.paramValue(mlpGa, name, "h"));
      int nPop = Integer.parseInt(it.units.erallab.hmsrobots.util.Utils.paramValue(mlpGa, name, "nPop"));
      return (p, body) -> new StandardEvolver<>(
          ((Function<List<Double>, MultiLayerPerceptron>) ws -> new MultiLayerPerceptron(
              MultiLayerPerceptron.ActivationFunction.TANH,
              p.first().apply(body).first(),
              ratioOfFirstLayer == 0 ? new int[0] : new int[]{(int) Math.round(p.first().apply(body).first() * ratioOfFirstLayer)},
              p.first().apply(body).second(),
              ws.stream().mapToDouble(d -> d).toArray()
          )).andThen(mlp -> p.second().apply(body).apply(mlp)),
          new FixedLengthListFactory<>(
              MultiLayerPerceptron.countWeights(
                  p.first().apply(body).first(),
                  ratioOfFirstLayer == 0 ? new int[0] : new int[]{(int) Math.round(p.first().apply(body).first() * ratioOfFirstLayer)},
                  p.first().apply(body).second()
              ),
              new UniformDoubleFactory(-1, 1)
          ),
          comparator,
          nPop,
          Map.of(
              new GaussianMutation(1d), 0.2d,
              new GeometricCrossover(), 0.8d
          ),
          new Tournament(5),
          new Worst(),
          nPop,
          true
      );
    }
    if (name.matches(mlpGaDiv)) {
      double ratioOfFirstLayer = Double.parseDouble(it.units.erallab.hmsrobots.util.Utils.paramValue(mlpGaDiv, name, "h"));
      int nPop = Integer.parseInt(it.units.erallab.hmsrobots.util.Utils.paramValue(mlpGaDiv, name, "nPop"));
      return (p, body) -> new StandardWithEnforcedDiversityEvolver<>(
          ((Function<List<Double>, MultiLayerPerceptron>) ws -> new MultiLayerPerceptron(
              MultiLayerPerceptron.ActivationFunction.TANH,
              p.first().apply(body).first(),
              ratioOfFirstLayer == 0 ? new int[0] : new int[]{(int) Math.round(p.first().apply(body).first() * ratioOfFirstLayer)},
              p.first().apply(body).second(),
              ws.stream().mapToDouble(d -> d).toArray()
          )).andThen(mlp -> p.second().apply(body).apply(mlp)),
          new FixedLengthListFactory<>(
              MultiLayerPerceptron.countWeights(
                  p.first().apply(body).first(),
                  ratioOfFirstLayer == 0 ? new int[0] : new int[]{(int) Math.round(p.first().apply(body).first() * ratioOfFirstLayer)},
                  p.first().apply(body).second()
              ),
              new UniformDoubleFactory(-1, 1)
          ),
          comparator,
          nPop,
          Map.of(
              new GaussianMutation(1d), 0.2d,
              new GeometricCrossover(), 0.8d
          ),
          new Tournament(5),
          new Worst(),
          nPop,
          true,
          100
      );
    }
    if (name.matches(mlpCmaEs)) {
      double ratioOfFirstLayer = Double.parseDouble(it.units.erallab.hmsrobots.util.Utils.paramValue(mlpCmaEs, name, "h"));
      return (p, body) -> new CMAESEvolver<>(
          ((Function<List<Double>, MultiLayerPerceptron>) ws -> new MultiLayerPerceptron(
              MultiLayerPerceptron.ActivationFunction.TANH,
              p.first().apply(body).first(),
              ratioOfFirstLayer == 0 ? new int[0] : new int[]{(int) Math.round(p.first().apply(body).first() * ratioOfFirstLayer)},
              p.first().apply(body).second(),
              ws.stream().mapToDouble(d -> d).toArray()
          )).andThen(mlp -> p.second().apply(body).apply(mlp)),
          new FixedLengthListFactory<>(
              MultiLayerPerceptron.countWeights(
                  p.first().apply(body).first(),
                  ratioOfFirstLayer == 0 ? new int[0] : new int[]{(int) Math.round(p.first().apply(body).first() * ratioOfFirstLayer)},
                  p.first().apply(body).second()
              ),
              new UniformDoubleFactory(-1, 1)
          ),
          comparator
      );
    }
    if (name.matches(graphea)) {
      int nPop = Integer.parseInt(it.units.erallab.hmsrobots.util.Utils.paramValue(graphea, name, "nPop"));
      return (p, body) -> {
        Function<Graph<IndexedNode<Node>, Double>, Graph<Node, Double>> graphMapper = GraphUtils.mapper(
            IndexedNode::content,
            Misc::first
        );
        Predicate<Graph<Node, Double>> checker = FunctionGraph.checker();
        return new SpeciatedEvolver<>(
            graphMapper
                .andThen(FunctionGraph.builder())
                .andThen(fg -> p.second().apply(body).apply(fg)),
            new ShallowSparseFactory(
                0d, 0d, 1d,
                p.first().apply(body).first(),
                p.first().apply(body).second()
            ).then(GraphUtils.mapper(IndexedNode.incrementerMapper(Node.class), Misc::first)),
            comparator,
            nPop,
            Map.of(
                new IndexedNodeAddition<FunctionNode, Node, Double>(
                    FunctionNode.sequentialIndexFactory(BaseFunction.TANH),
                    n -> (n instanceof FunctionNode) ? ((FunctionNode) n).getFunction().hashCode() : 0,
                    p.first().apply(body).first() + p.first().apply(body).second() + 1,
                    (w, r) -> w,
                    (w, r) -> r.nextGaussian()
                ).withChecker(g -> checker.test(graphMapper.apply(g))), 1d,
                new ArcModification<IndexedNode<Node>, Double>((w, r) -> w + r.nextGaussian(), 1d).withChecker(g -> checker.test(graphMapper.apply(g))), 1d,
                new ArcAddition<IndexedNode<Node>, Double>(Random::nextGaussian, false).withChecker(g -> checker.test(graphMapper.apply(g))), 3d,
                new AlignedCrossover<IndexedNode<Node>, Double>(
                    (w1, w2, r) -> w1 + (w2 - w1) * (r.nextDouble() * 3d - 1d),
                    node -> node.content() instanceof Output,
                    false
                ).withChecker(g -> checker.test(graphMapper.apply(g))), 1d
            ),
            5,
            (new Jaccard()).on(i -> i.getGenotype().nodes()),
            0.25,
            individuals -> {
              double[] fitnesses = individuals.stream().mapToDouble(Individual::getFitness).toArray();
              Individual<Graph<IndexedNode<Node>, Double>, Robot<?>, Double> r = Misc.first(individuals);
              return new Individual<>(
                  r.getGenotype(),
                  r.getSolution(),
                  Misc.median(fitnesses),
                  r.getBirthIteration()
              );
            },
            0.75
        );
      };
    }
    if (name.matches(grapheaNoXOver)) {
      int nPop = Integer.parseInt(it.units.erallab.hmsrobots.util.Utils.paramValue(grapheaNoXOver, name, "nPop"));
      return (p, body) -> new SpeciatedEvolver<>(
          FunctionGraph.builder().andThen(fg -> p.second().apply(body).apply(fg)),
          new ShallowSparseFactory(
              0d, 0d, 1d,
              p.first().apply(body).first(),
              p.first().apply(body).second()
          ),
          comparator,
          nPop,
          Map.of(
              new NodeAddition<Node, Double>(
                  FunctionNode.sequentialIndexFactory(BaseFunction.TANH),
                  (w, r) -> w,
                  (w, r) -> r.nextGaussian()
              ).withChecker(FunctionGraph.checker()), 1d,
              new ArcModification<Node, Double>((w, r) -> w + r.nextGaussian(), 1d).withChecker(FunctionGraph.checker()), 1d,
              new ArcAddition<Node, Double>(Random::nextGaussian, false).withChecker(FunctionGraph.checker()), 3d
          ),
          5,
          (new Jaccard()).on(i -> i.getGenotype().nodes()),
          0.25,
          individuals -> {
            double[] fitnesses = individuals.stream().mapToDouble(Individual::getFitness).toArray();
            Individual<Graph<Node, Double>, Robot<?>, Double> r = Misc.first(individuals);
            return new Individual<>(
                r.getGenotype(),
                r.getSolution(),
                Misc.median(fitnesses),
                r.getBirthIteration()
            );
          },
          0.75
      );
    }
    throw new IllegalArgumentException(String.format("Unknown evolver name: %s", name));
  }

  private static Outcome prototypeOutcome() {
    return new Outcome(
        0d, 10d, 0d, 0d, 0d,
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

}
