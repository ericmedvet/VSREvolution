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
import it.units.erallab.hmsrobots.core.objects.BreakableVoxel;
import it.units.erallab.hmsrobots.core.objects.Robot;
import it.units.erallab.hmsrobots.core.objects.SensingVoxel;
import it.units.erallab.hmsrobots.core.sensors.*;
import it.units.erallab.hmsrobots.tasks.Locomotion;
import it.units.erallab.hmsrobots.util.Grid;
import it.units.malelab.jgea.Worker;
import it.units.malelab.jgea.core.Individual;
import it.units.malelab.jgea.core.evolver.*;
import it.units.malelab.jgea.core.evolver.stopcondition.Births;
import it.units.malelab.jgea.core.listener.Listener;
import it.units.malelab.jgea.core.listener.MultiFileListenerFactory;
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
import org.apache.commons.lang3.SerializationUtils;
import org.dyn4j.dynamics.Settings;

import java.io.Serializable;
import java.util.*;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.TimeUnit;
import java.util.function.BiFunction;
import java.util.function.Function;
import java.util.function.UnaryOperator;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

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
    double episodeTime = d(a("episodeT", "10.0"));
    int nBirths = i(a("nBirths", "1000"));
    int[] seeds = ri(a("seed", "0:1"));
    //int validationBirthsInterval = i(a("validationBirthsInterval", "100"));
    List<String> terrainNames = l(a("terrain", "flat"));
    List<String> evolverMapperNames = l(a("evolver", "mlp-0.65-cmaes"));
    List<String> bodyNames = l(a("body", "biped-4x3-f-f"));
    List<String> transformationNames = l(a("transformations", "identity"));
    List<String> robotMapperNames = l(a("mapper", "centralized"));
    Locomotion.Metric fitnessMetric = Locomotion.Metric.valueOf(a("fitnessMetric", Locomotion.Metric.X_DISTANCE_CORRECTED_EFFICIENCY.name().toLowerCase()).toUpperCase());
    List<Locomotion.Metric> allMetrics = l(a("metrics", List.of(Locomotion.Metric.values()).stream().map(m -> m.name().toLowerCase()).collect(Collectors.joining(",")))).stream()
        .map(String::toUpperCase)
        .map(Locomotion.Metric::valueOf)
        .collect(Collectors.toList());
    if (!allMetrics.contains(fitnessMetric)) {
      allMetrics.add(fitnessMetric);
    }
    List<String> validationTransformationNames = l(a("validationTransformations", "")).stream().filter(s -> !s.isEmpty()).collect(Collectors.toList());
    List<String> validationTerrainNames = l(a("validationTerrains", "")).stream().filter(s -> !s.isEmpty()).collect(Collectors.toList());
    if (!validationTerrainNames.isEmpty() && validationTransformationNames.isEmpty()) {
      validationTransformationNames.add("identity");
    }
    if (validationTerrainNames.isEmpty() && !validationTransformationNames.isEmpty()) {
      validationTerrainNames.add(terrainNames.get(0));
    }
    Settings physicsSettings = new Settings();
    //prepare file listeners
    MultiFileListenerFactory<Object, Robot<?>, Double> statsListenerFactory = new MultiFileListenerFactory<>(
        a("dir", "."),
        a("statsFile", null)
    );
    MultiFileListenerFactory<Object, Robot<?>, Double> serializedListenerFactory = new MultiFileListenerFactory<>(
        a("dir", "."),
        a("serializedFile", null)
    );
    L.info("Evolvers: " + evolverMapperNames);
    L.info("Mappers: " + robotMapperNames);
    L.info("Bodies: " + bodyNames);
    L.info("Terrains: " + terrainNames);
    L.info("Transformations: " + transformationNames);
    L.info("Validations: " + Lists.cartesianProduct(validationTerrainNames, validationTransformationNames));
    for (int seed : seeds) {
      for (String terrainName : terrainNames) {
        for (String bodyName : bodyNames) {
          for (String robotMapperName : robotMapperNames) {
            for (String transformationName : transformationNames) {
              for (String evolverMapperName : evolverMapperNames) {
                Map<String, String> keys = new TreeMap<>(Map.of(
                    "seed", Integer.toString(seed),
                    "terrain", terrainName,
                    "body", bodyName,
                    "mapper", robotMapperName,
                    "transformation", transformationName,
                    "evolver", evolverMapperName
                ));
                Grid<? extends SensingVoxel> body = buildBody(bodyName);
                //build training task
                Function<Robot<?>, List<Double>> trainingTask = Misc.cached(
                    buildRobotTransformation(transformationName).andThen(new Locomotion(
                        episodeTime,
                        Locomotion.createTerrain(terrainName),
                        allMetrics,
                        physicsSettings
                    )), CACHE_SIZE);
                //build main data collectors for listener
                List<DataCollector<?, ? super Robot<SensingVoxel>, ? super Double>> collectors = new ArrayList<DataCollector<?, ? super Robot<SensingVoxel>, ? super Double>>(List.of(
                    new Static(keys),
                    new Basic(),
                    new Population(),
                    new Diversity(),
                    new BestInfo("%5.2f"),
                    new FunctionOfOneBest<>(
                        ((Function<Individual<?, ? extends Robot<SensingVoxel>, ? extends Double>, Robot<SensingVoxel>>) Individual::getSolution)
                            .andThen(SerializationUtils::clone)
                            .andThen(metrics(allMetrics, "training", trainingTask, "%6.2f"))
                    )
                ));
                //build body validation tasks
                if (!validationTransformationNames.isEmpty() && !validationTerrainNames.isEmpty()) {
                  for (String validationTransformationName : validationTransformationNames) {
                    for (String validationTerrainName : validationTerrainNames) {
                      //build validation task
                      Function<Robot<?>, List<Double>> validationTask = new Locomotion(
                          episodeTime,
                          Locomotion.createTerrain(validationTerrainName),
                          allMetrics,
                          physicsSettings
                      );
                      validationTask = Misc.cached(
                          buildRobotTransformation(
                              validationTransformationName)
                              .andThen(SerializationUtils::clone)
                              .andThen(validationTask),
                          CACHE_SIZE);
                      //add to collector
                      collectors.add(new FunctionOfOneBest<>(
                              ((Function<Individual<?, ? extends Robot<SensingVoxel>, ? extends Double>, Robot<SensingVoxel>>) Individual::getSolution)
                                  .andThen(metrics(allMetrics, "validation." + validationTerrainName + "." + validationTransformationName, validationTask, "%6.2f"))
                          )
                      );
                    }
                  }
                }
                Listener<? super Object, ? super Robot<?>, ? super Double> listener;
                if (statsListenerFactory.getBaseFileName() == null) {
                  listener = listener(collectors.toArray(DataCollector[]::new));
                } else {
                  listener = statsListenerFactory.build(collectors.toArray(DataCollector[]::new));
                }
                if (serializedListenerFactory.getBaseFileName() != null) {
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
                  Collection<Robot<?>> solutions = evolver.solve(
                      trainingTask.andThen(values -> values.get(allMetrics.indexOf(fitnessMetric))),
                      new Births(nBirths),
                      new Random(seed),
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

  private static Pair<IODimMapper, RobotMapper> buildRobotMapper(String name) {
    String centralized = "centralized";
    String phases = "phases-(?<f>\\d+(\\.\\d+)?)";
    if (name.matches(centralized)) {
      return Pair.of(
          body -> Pair.of(CentralizedSensing.nOfInputs(body), CentralizedSensing.nOfOutputs(body)),
          body -> f -> new Robot<>(
              new CentralizedSensing(body, f),
              SerializationUtils.clone(body)
          )
      );
    }
    if (name.matches(phases)) {
      return Pair.of(
          body -> Pair.of(2, 1),
          body -> f -> new Robot<>(
              new PhaseSin(
                  -Double.parseDouble(paramValue(phases, name, "f")),
                  1d,
                  Grid.create(
                      body.getW(),
                      body.getH(),
                      (x, y) -> f.apply(new double[]{(double) x / (double) body.getW(), (double) y / (double) body.getH()})[0]
                  )),
              SerializationUtils.clone(body)
          )
      );
    }
    throw new IllegalArgumentException(String.format("Unknown mapper name: %s", name));
  }

  private static Function<Robot<?>, List<Item>> metrics(List<Locomotion.Metric> metrics, String prefix, Function<Robot<?>, List<Double>> task, String format) {
    return individual -> {
      List<Double> values = task.apply(individual);
      List<Item> items = new ArrayList<>(metrics.size());
      for (int i = 0; i < metrics.size(); i++) {
        items.add(new Item(
            prefix + "." + metrics.get(i).name().toLowerCase(),
            values.get(i),
            format
        ));
      }
      return items;
    };
  }

  private static UnaryOperator<Robot<?>> buildRobotTransformation(String name) {
    String areaBreakable = "areaBreak-(?<rate>\\d+(\\.\\d+)?)-(?<threshold>\\d+(\\.\\d+)?)-(?<seed>\\d+)";
    String timeBreakable = "timeBreak-(?<time>\\d+(\\.\\d+)?)-(?<seed>\\d+)";
    String identity = "identity";
    if (name.matches(identity)) {
      return UnaryOperator.identity();
    }
    if (name.matches(areaBreakable)) {
      double rate = Double.parseDouble(paramValue(areaBreakable, name, "rate"));
      double threshold = Double.parseDouble(paramValue(areaBreakable, name, "threshold"));
      Random random = new Random(Integer.parseInt(paramValue(areaBreakable, name, "seed")));
      return robot -> new Robot<>(
          ((Robot<SensingVoxel>) robot).getController(),
          Grid.create((Grid<SensingVoxel>) robot.getVoxels(), v -> v == null ? null : (random.nextDouble() > rate ? v : new BreakableVoxel(
              v.getSensors(),
              random,
              Map.of(
                  BreakableVoxel.ComponentType.ACTUATOR, Set.of(BreakableVoxel.MalfunctionType.FROZEN),
                  BreakableVoxel.ComponentType.SENSORS, Set.of(BreakableVoxel.MalfunctionType.ZERO)
              ),
              Map.of(BreakableVoxel.MalfunctionTrigger.AREA, threshold)
          )))
      );
    }
    if (name.matches(timeBreakable)) {
      double time = Double.parseDouble(paramValue(timeBreakable, name, "time"));
      Random random = new Random(Integer.parseInt(paramValue(timeBreakable, name, "seed")));
      return robot -> {
        List<Pair<Integer, Integer>> coords = robot.getVoxels().stream()
            .filter(e -> e.getValue() != null)
            .map(e -> Pair.of(e.getX(), e.getY()))
            .collect(Collectors.toList());
        Collections.shuffle(coords, random);
        Grid<SensingVoxel> body = SerializationUtils.clone((Grid<SensingVoxel>) robot.getVoxels());
        for (int i = 0; i < coords.size(); i++) {
          int x = coords.get(i).first();
          int y = coords.get(i).second();
          body.set(x, y, new BreakableVoxel(
              body.get(x, y).getSensors(),
              random,
              Map.of(
                  BreakableVoxel.ComponentType.ACTUATOR, Set.of(BreakableVoxel.MalfunctionType.FROZEN),
                  BreakableVoxel.ComponentType.SENSORS, Set.of(BreakableVoxel.MalfunctionType.ZERO)
              ),
              Map.of(BreakableVoxel.MalfunctionTrigger.TIME, time * ((double) (i + 1) / (double) coords.size()))
          ));
        }
        return new Robot<>(
            ((Robot<SensingVoxel>) robot).getController(),
            body
        );
      };
    }
    throw new IllegalArgumentException(String.format("Unknown body name: %s", name));
  }

  private static Grid<? extends SensingVoxel> buildBody(String name) {
    String wbt = "(?<shape>worm|biped|tripod)-(?<w>\\d+)x(?<h>\\d+)-(?<cgp>[tf])-(?<malfunction>[tf])";
    if (name.matches(wbt)) {
      String shape = paramValue(wbt, name, "shape");
      int w = Integer.parseInt(paramValue(wbt, name, "w"));
      int h = Integer.parseInt(paramValue(wbt, name, "h"));
      boolean withCentralPatternGenerator = paramValue(wbt, name, "cgp").equals("t");
      boolean withMalfunctionSensor = paramValue(wbt, name, "malfunction").equals("t");
      Grid<? extends SensingVoxel> body = Grid.create(
          w, h,
          (x, y) -> new SensingVoxel(ofNonNull(
              new AreaRatio(),
              withMalfunctionSensor ? new Malfunction() : null,
              (y == 0) ? new Touch() : null,
              (y == h - 1) ? new Velocity(true, 3d, Velocity.Axis.X, Velocity.Axis.Y) : null,
              (x == w - 1 && y == h - 1 && withCentralPatternGenerator) ? new TimeFunction(t -> Math.sin(2 * Math.PI * -1 * t), -1, 1) : null
          ).stream().filter(Objects::nonNull).collect(Collectors.toList())));
      if (shape.equals("biped")) {
        final Grid<? extends SensingVoxel> finalBody = body;
        body = Grid.create(w, h, (x, y) -> (y == 0 && x > 0 && x < w - 1) ? null : finalBody.get(x, y));
      } else if (shape.equals("tripod")) {
        final Grid<? extends SensingVoxel> finalBody = body;
        body = Grid.create(w, h, (x, y) -> (y != h - 1 && x != 0 && x != w - 1 && x != w / 2) ? null : finalBody.get(x, y));
      }
      return body;
    }
    throw new IllegalArgumentException(String.format("Unknown body name: %s", name));
  }

  private static EvolverMapper buildEvolverMapper(String name) {
    PartialComparator<Individual<?, Robot<?>, Double>> comparator = PartialComparator.from(Double.class).reversed().comparing(Individual::getFitness);
    String mlpGa = "mlp-(?<h>\\d+(\\.\\d+)?)-ga-(?<nPop>\\d+)";
    String mlpGaDiv = "mlp-(?<h>\\d+(\\.\\d+)?)-gadiv-(?<nPop>\\d+)";
    String mlpCmaEs = "mlp-(?<h>\\d+(\\.\\d+)?)-cmaes";
    String graphea = "fgraph-hash-speciated-(?<nPop>\\d+)";
    if (name.matches(mlpGa)) {
      double ratioOfFirstLayer = Double.parseDouble(paramValue(mlpGa, name, "h"));
      int nPop = Integer.parseInt(paramValue(mlpGa, name, "nPop"));
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
      double ratioOfFirstLayer = Double.parseDouble(paramValue(mlpGaDiv, name, "h"));
      int nPop = Integer.parseInt(paramValue(mlpGaDiv, name, "nPop"));
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
      double ratioOfFirstLayer = Double.parseDouble(paramValue(mlpCmaEs, name, "h"));
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
          comparator,
          -1,
          1
      );
    }
    if (name.matches(graphea)) {
      int nPop = Integer.parseInt(paramValue(graphea, name, "nPop"));
      return (p, body) -> new SpeciatedEvolver<>(
          GraphUtils.mapper((Function<IndexedNode<Node>, Node>) IndexedNode::content, (Function<Collection<Double>, Double>) Misc::first)
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
              new IndexedNodeAddition<>(
                  FunctionNode.sequentialIndexFactory(BaseFunction.TANH),
                  n -> n.getFunction().hashCode(),
                  p.first().apply(body).first() + p.first().apply(body).second() + 1,
                  (w, r) -> w,
                  (w, r) -> r.nextGaussian()
              ), 1d,
              new ArcModification<>((w, r) -> w + r.nextGaussian(), 1d), 1d,
              new ArcAddition
                  <>(Random::nextGaussian, false), 3d,
              new AlignedCrossover<>(
                  (w1, w2, r) -> w1 + (w2 - w1) * (r.nextDouble() * 3d - 1d),
                  node -> node.content() instanceof Output,
                  false
              ), 1d
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
    }
    throw new IllegalArgumentException(String.format("Unknown evolver name: %s", name));
  }

  private static String paramValue(String pattern, String string, String paramName) {
    Matcher matcher = Pattern.compile(pattern).matcher(string);
    if (matcher.matches()) {
      return matcher.group(paramName);
    }
    throw new IllegalStateException(String.format("Param %s not found in %s with pattern %s", paramName, string, pattern));
  }

  private static <E> List<E> ofNonNull(E... es) {
    List<E> list = new ArrayList<>();
    for (E e : es) {
      if (e != null) {
        list.add(e);
      }
    }
    return list;
  }
}
