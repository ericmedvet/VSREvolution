package it.units.erallab;

import com.google.common.base.Stopwatch;
import com.google.common.collect.Lists;
import com.google.common.graph.ValueGraph;
import it.units.erallab.hmsrobots.core.controllers.CentralizedSensing;
import it.units.erallab.hmsrobots.core.controllers.MultiLayerPerceptron;
import it.units.erallab.hmsrobots.core.controllers.PhaseSin;
import it.units.erallab.hmsrobots.core.objects.Robot;
import it.units.erallab.hmsrobots.core.objects.SensingVoxel;
import it.units.erallab.hmsrobots.core.sensors.AreaRatio;
import it.units.erallab.hmsrobots.core.sensors.TimeFunction;
import it.units.erallab.hmsrobots.core.sensors.Touch;
import it.units.erallab.hmsrobots.core.sensors.Velocity;
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
import it.units.malelab.jgea.representation.graph.numeric.Node;
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

  public ControllerComparison(String[] args) {
    super(args);
  }

  public static void main(String[] args) {
    new ControllerComparison(args);
  }

  private interface BodyIOMapper extends Function<Grid<SensingVoxel>, Pair<Integer, Integer>> {
  }


  private interface BodyMapperMapper extends Function<Grid<SensingVoxel>, Function<Function<double[], double[]>, Robot<SensingVoxel>>> {
  }


  @Override
  public void run() {
    double episodeTime = d(a("episodeT", "10.0"));
    int nBirths = i(a("nBirths", "50"));
    int[] seeds = ri(a("seed", "0:1"));
    List<String> terrains = l(a("terrain", "flat"));
    List<String> evolverNames = l(a("evolver", "mlp-0.65-gadiv-10"));
    List<String> bodyNames = l(a("body", "biped-cpg-4x3"));
    List<String> mapperNames = l(a("mapper", "centralized"));
    Locomotion.Metric metric = Locomotion.Metric.TRAVEL_X_VELOCITY;
    //prepare file listeners
    MultiFileListenerFactory<Object, Robot<SensingVoxel>, Double> statsListenerFactory = new MultiFileListenerFactory<>(
        a("dir", "."),
        a("statsFile", null)
    );
    MultiFileListenerFactory<Object, Robot<SensingVoxel>, Double> serializedListenerFactory = new MultiFileListenerFactory<>(
        a("dir", "."),
        a("serializedFile", null)
    );
    Map<String, Grid<SensingVoxel>> bodies = bodyNames.stream()
        .collect(Collectors.toMap(n -> n, ControllerComparison::buildBodyFromName));
    Map<String, Pair<BodyIOMapper, BodyMapperMapper>> mappers = mapperNames.stream()
        .collect(Collectors.toMap(n -> n, ControllerComparison::buildMapperFromName));
    Map<String, BiFunction<Pair<BodyIOMapper, BodyMapperMapper>, Grid<SensingVoxel>, Evolver<?, Robot<SensingVoxel>, Double>>> evolvers = evolverNames.stream()
        .collect(Collectors.toMap(n -> n, ControllerComparison::buildEvolverBuilderFromName));
    L.info("Evolvers: " + evolvers.keySet());
    L.info("Mappers: " + mappers.keySet());
    L.info("Bodies: " + bodies.keySet());
    for (int seed : seeds) {
      for (String terrain : terrains) {
        for (Map.Entry<String, Grid<SensingVoxel>> bodyEntry : bodies.entrySet()) {
          for (Map.Entry<String, Pair<BodyIOMapper, BodyMapperMapper>> mapperEntry : mappers.entrySet()) {
            for (Map.Entry<String, BiFunction<Pair<BodyIOMapper, BodyMapperMapper>, Grid<SensingVoxel>, Evolver<?, Robot<SensingVoxel>, Double>>> evolverEntry : evolvers.entrySet()) {
              Map<String, String> keys = new TreeMap<>(Map.of(
                  "seed", Integer.toString(seed),
                  "terrain", terrain,
                  "body", bodyEntry.getKey(),
                  "mapper", mapperEntry.getKey(),
                  "evolver", evolverEntry.getKey()
              ));
              Grid<SensingVoxel> body = bodyEntry.getValue();
              Evolver<?, Robot<SensingVoxel>, Double> evolver = evolverEntry.getValue().apply(mapperEntry.getValue(), body);
              Locomotion locomotion = new Locomotion(
                  episodeTime,
                  Locomotion.createTerrain(terrain),
                  Lists.newArrayList(metric),
                  new Settings()
              );
              List<DataCollector<?, ? super Robot<SensingVoxel>, ? super Double>> collectors = List.of(
                  new Static(keys),
                  new Basic(),
                  new Population(),
                  new Diversity(),
                  new BestInfo("%7.5f")
              );
              Listener<? super Object, ? super Robot<SensingVoxel>, ? super Double> listener;
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
                Collection<Robot<SensingVoxel>> solutions = evolver.solve(
                    Misc.cached(robot -> locomotion.apply(robot).get(0), 10000),
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

  private static Pair<BodyIOMapper, BodyMapperMapper> buildMapperFromName(String name) {
    if (name.matches("centralized")) {
      return Pair.of(
          body -> Pair.of(CentralizedSensing.nOfInputs(body), CentralizedSensing.nOfOutputs(body)),
          body -> f -> new Robot<>(
              new CentralizedSensing<>(body, f),
              SerializationUtils.clone(body)
          )
      );
    }
    if (name.matches("phases(-[0-9]+(\\.[0-9]+)?)?")) {
      return Pair.of(
          body -> Pair.of(2, 1),
          body -> f -> new Robot<>(
              new PhaseSin(
                  -extractParamValueFromName(name, 0, 1),
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

  private static Grid<SensingVoxel> buildBodyFromName(String name) {
    if (name.matches("biped(-cpg)?-[0-9]+x[0-9]+")) {
      int w = (int) extractParamValueFromName(name, 0, 4);
      int h = (int) extractParamValueFromName(name, 1, 3);
      return Grid.create(
          w, h,
          (x, y) -> (y == 0 && x > 0 && x < w - 1) ? null : new SensingVoxel(ofNonNull(
              new AreaRatio(),
              (y == 0) ? new Touch() : null,
              (y == h - 1) ? new Velocity(true, 3d, Velocity.Axis.X, Velocity.Axis.Y) : null,
              (x == w - 1 && y == h - 1 && name.contains("-cpg")) ? new TimeFunction(t -> Math.sin(2 * Math.PI * -1 * t), -1, 1) : null
          ).stream().filter(Objects::nonNull).collect(Collectors.toList())));
    }
    throw new IllegalArgumentException(String.format("Unknown body name: %s", name));
  }

  private static BiFunction<Pair<BodyIOMapper, BodyMapperMapper>, Grid<SensingVoxel>, Evolver<?, Robot<SensingVoxel>, Double>> buildEvolverBuilderFromName(String name) {
    if (name.matches("mlp-[0-9]+(\\.[0-9]+)?-ga(-[0-9]+)?")) {
      double ratioOfFirstLayer = extractParamValueFromName(name, 0, 0);
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
          PartialComparator.from(Double.class).on(Individual::getFitness),
          (int) extractParamValueFromName(name, 1, 100),
          Map.of(
              new GaussianMutation(1d), 0.2d,
              new GeometricCrossover(), 0.8d
          ),
          new Tournament(5),
          new Worst(),
          (int) extractParamValueFromName(name, 1, 100),
          true
      );
    }
    if (name.matches("mlp-[0-9]+(\\.[0-9]+)?-gadiv(-[0-9]+)?")) {
      double ratioOfFirstLayer = extractParamValueFromName(name, 0, 0);
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
          PartialComparator.from(Double.class).on(Individual::getFitness),
          (int) extractParamValueFromName(name, 1, 100),
          Map.of(
              new GaussianMutation(1d), 0.2d,
              new GeometricCrossover(), 0.8d
          ),
          new Tournament(5),
          new Worst(),
          (int) extractParamValueFromName(name, 1, 100),
          true,
          100
      );
    }
    if (name.matches("mlp-[0-9]+(\\.[0-9]+)?-cmaes")) {
      double ratioOfFirstLayer = extractParamValueFromName(name, 0, 0);
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
          PartialComparator.from(Double.class).on(Individual::getFitness),
          -1,
          1
      );
    }
    if (name.matches("fgraph-hash-speciated(-[0-9]+)?")) {
      return (p, body) -> new SpeciatedEvolver<>(
          GraphUtils.mapper((Function<IndexedNode<Node>, Node>) IndexedNode::content, (Function<Collection<Double>, Double>) Misc::first)
              .andThen(FunctionGraph.builder())
              .andThen(fg -> p.second().apply(body).apply(fg)),
          new ShallowSparseFactory(
              0d, 0d, 1d,
              p.first().apply(body).first(),
              p.first().apply(body).second()
          ).then(GraphUtils.mapper(IndexedNode.incrementerMapper(Node.class), Misc::first)),
          PartialComparator.from(Double.class).on(Individual::getFitness),
          (int) extractParamValueFromName(name, 0, 100),
          Map.of(
              new IndexedNodeAddition<>(
                  FunctionNode.sequentialIndexFactory(BaseFunction.TANH),
                  n -> n.getFunction().hashCode(),
                  p.first().apply(body).first() + p.first().apply(body).second(),
                  (w, r) -> w,
                  (w, r) -> r.nextGaussian()
              ), 1d,
              new EdgeModification<>((w, r) -> w + r.nextGaussian(), 1d), 1d,
              new EdgeAddition<>(Random::nextGaussian, false), 3d,
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
            Individual<ValueGraph<IndexedNode<Node>, Double>, Robot<SensingVoxel>, Double> r = Misc.first(individuals);
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

  private static double extractParamValueFromName(String name, int index, double defaultValue) {
    Matcher matcher = Pattern.compile("[0-9]+(\\.[0-9]+)?").matcher(name);
    List<Double> numbers = new ArrayList<>();
    int s = 0;
    while (matcher.find(s)) {
      numbers.add(Double.parseDouble(name.substring(matcher.start(), matcher.end())));
      s = matcher.end();
    }
    if (numbers.size() > index) {
      return numbers.get(index);
    }
    return defaultValue;
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
