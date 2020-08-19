package it.units.erallab;

import com.google.common.base.Stopwatch;
import com.google.common.collect.Lists;
import it.units.erallab.hmsrobots.core.controllers.CentralizedSensing;
import it.units.erallab.hmsrobots.core.controllers.MultiLayerPerceptron;
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
import it.units.malelab.jgea.core.evolver.CMAESEvolver;
import it.units.malelab.jgea.core.evolver.Evolver;
import it.units.malelab.jgea.core.evolver.StandardEvolver;
import it.units.malelab.jgea.core.evolver.stopcondition.Births;
import it.units.malelab.jgea.core.listener.Listener;
import it.units.malelab.jgea.core.listener.MultiFileListenerFactory;
import it.units.malelab.jgea.core.listener.collector.*;
import it.units.malelab.jgea.core.order.PartialComparator;
import it.units.malelab.jgea.core.selector.Tournament;
import it.units.malelab.jgea.core.selector.Worst;
import it.units.malelab.jgea.core.util.Misc;
import it.units.malelab.jgea.core.util.Pair;
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
    String fileName = a("file", null);
    int nPop = i(a("nPop", "5"));
    double episodeTime = d(a("episodeT", "10.0"));
    int nTournament = 5;
    int nBirths = i(a("nBirths", "50"));
    int[] seeds = ri(a("seed", "0:1"));
    double frequency = 1d;
    Locomotion.Metric metric = Locomotion.Metric.TRAVEL_X_VELOCITY;
    //prepare file listeners
    MultiFileListenerFactory<Object, Robot<SensingVoxel>, Double> statsListenerFactory = new MultiFileListenerFactory<>(
        a("dir", "."),
        fileName
    );
    MultiFileListenerFactory<Object, Robot<SensingVoxel>, Double> serializedListenerFactory = new MultiFileListenerFactory<>(
        a("dir", "."),
        fileName != null ? fileName.replaceFirst("\\.", ".ser.") : null
    );
    Map<String, Grid<SensingVoxel>> bodies = Map.ofEntries(
        Map.entry(
            "biped-cpg-4x3",
            Grid.create(4, 3, (x, y) -> (y == 0 && x >= 1 && x <= 2) ? null : new SensingVoxel(ofNonNull(
                new AreaRatio(),
                (y == 0) ? new Touch() : null,
                (y == 2) ? new Velocity(true, 3d, Velocity.Axis.X, Velocity.Axis.Y) : null,
                (x == 0 && y == 2) ? new TimeFunction(t -> Math.sin(2 * Math.PI * frequency * t), -1, 1) : null
            ).stream().filter(Objects::nonNull).collect(Collectors.toList())))
        ),
        Map.entry(
            "biped-4x3",
            Grid.create(4, 3, (x, y) -> (y == 0 && x >= 1 && x <= 2) ? null : new SensingVoxel(ofNonNull(
                new AreaRatio(),
                (y == 2) ? new Velocity(true, 3d, Velocity.Axis.X, Velocity.Axis.Y) : null,
                (y == 0) ? new Touch() : null
            ).stream().filter(Objects::nonNull).collect(Collectors.toList())))
        )
    );
    List<String> terrains = List.of("uneven5");
    Map<String, Pair<BodyIOMapper, BodyMapperMapper>> mappers = Map.ofEntries(
        Map.entry(
            "centralized",
            Pair.of(
                body -> Pair.of(CentralizedSensing.nOfInputs(body), CentralizedSensing.nOfOutputs(body)),
                body -> f -> new Robot<>(new CentralizedSensing<>(body, f), SerializationUtils.clone(body))
            )
        )
    );
    Map<String, BiFunction<Pair<BodyIOMapper, BodyMapperMapper>, Grid<SensingVoxel>, Evolver<?, Robot<SensingVoxel>, Double>>> evolvers = Map.ofEntries(
        Map.entry(
            "mlp-0-ga",
            (p, body) -> new StandardEvolver<List<Double>, Robot<SensingVoxel>, Double>(
                ((Function<List<Double>, MultiLayerPerceptron>) ws -> new MultiLayerPerceptron(
                    MultiLayerPerceptron.ActivationFunction.TANH,
                    p.first().apply(body).first(),
                    new int[0],
                    p.first().apply(body).second(),
                    ws.stream().mapToDouble(d -> d).toArray()
                )).andThen(mlp -> p.second().apply(body).apply(mlp)),
                new FixedLengthListFactory<>(
                    MultiLayerPerceptron.countWeights(
                        p.first().apply(body).first(),
                        new int[0],
                        p.first().apply(body).second()
                    ),
                    new UniformDoubleFactory(-1, 1)
                ),
                PartialComparator.from(Double.class).on(Individual::getFitness),
                nPop,
                Map.of(
                    new GaussianMutation(1d), 0.2d,
                    new GeometricCrossover(), 0.8d
                ),
                new Tournament(nTournament),
                new Worst(),
                nPop,
                true
            )
        ),
        Map.entry(
            "mlp-0-cmaes",
            (p, body) -> new CMAESEvolver<Robot<SensingVoxel>, Double>(
                ((Function<List<Double>, MultiLayerPerceptron>) ws -> new MultiLayerPerceptron(
                    MultiLayerPerceptron.ActivationFunction.TANH,
                    p.first().apply(body).first(),
                    new int[0],
                    p.first().apply(body).second(),
                    ws.stream().mapToDouble(d -> d).toArray()
                )).andThen(mlp -> p.second().apply(body).apply(mlp)),
                new FixedLengthListFactory<>(
                    MultiLayerPerceptron.countWeights(
                        p.first().apply(body).first(),
                        new int[0],
                        p.first().apply(body).second()
                    ),
                    new UniformDoubleFactory(-1, 1)
                ),
                PartialComparator.from(Double.class).on(Individual::getFitness),
                -1,
                1
            )
        )
    );
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
