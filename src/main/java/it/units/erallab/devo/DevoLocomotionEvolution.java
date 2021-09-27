package it.units.erallab.devo;

import it.units.erallab.builder.PrototypedFunctionBuilder;
import it.units.erallab.builder.evolver.EvolverBuilder;
import it.units.erallab.hmsrobots.core.objects.Robot;
import it.units.erallab.hmsrobots.tasks.locomotion.Outcome;
import it.units.erallab.hmsrobots.util.Grid;
import it.units.malelab.jgea.Worker;
import it.units.malelab.jgea.core.Individual;
import it.units.malelab.jgea.core.evolver.Event;
import it.units.malelab.jgea.core.evolver.Evolver;
import it.units.malelab.jgea.core.listener.Listener;
import it.units.malelab.jgea.core.listener.NamedFunction;
import it.units.malelab.jgea.core.listener.TabularPrinter;
import it.units.malelab.jgea.core.util.Misc;
import it.units.malelab.jgea.core.util.Pair;

import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Random;
import java.util.function.Function;

import static it.units.malelab.jgea.core.listener.NamedFunctions.*;
import static it.units.malelab.jgea.core.util.Args.*;

/**
 * @author "Eric Medvet" on 2021/09/27 for VSREvolution
 */
public class DevoLocomotionEvolution extends Worker {

  public DevoLocomotionEvolution(String[] args) {
    super(args);
  }

  public static void main(String[] args) {
    new DevoLocomotionEvolution(args);
  }

  @Override
  public void run() {
    //main params
    double episodeTime = d(a("episodeTime", "10"));
    double episodeTransientTime = d(a("episodeTransientTime", "1"));
    int nEvals = i(a("nEvals", "100"));
    int[] seeds = ri(a("seed", "0:1"));
    String experimentName = a("expName", "short");
    int gridW = i(a("gridW", "10"));
    int gridH = i(a("gridH", "10"));
    List<String> terrainNames = l(a("terrain", "flat"));
    List<String> devoFunctionNames = l(a("devoFunction", "plain"));
    List<String> evolverNames = l(a("evolver", "numGA-16-f,ES-8-0.35"));
    String lastFileName = a("lastFile", null);
    String bestFileName = a("bestFile", null);
    String allFileName = a("allFile", null);
    //fitness function
    Function<List<Outcome>, Double> fitnessFunction = outcomes -> outcomes.stream().mapToDouble(Outcome::getDistance).sum();
    //consumers
    List<NamedFunction<Event<?, ? extends List<Robot<?>>, ? extends List<Outcome>>, ?>> keysFunctions = keyFunctions();
    List<NamedFunction<Event<?, ? extends List<Robot<?>>, ? extends List<Outcome>>, ?>> basicFunctions = basicFunctions();
    List<NamedFunction<Event<?, ? extends List<Robot<?>>, ? extends List<Outcome>>, ?>> populationFunctions = List.of();
    List<NamedFunction<Individual<?, ? extends List<Robot<?>>, ? extends List<Outcome>>, ?>> basicIndividualFunctions = basicIndividualFunctions(fitnessFunction);
    List<NamedFunction<List<Outcome>, ?>> outcomesFunctions = List.of();
    List<NamedFunction<Individual<?, ? extends List<Robot<?>>, ? extends List<Outcome>>, ?>> visualIndividualFunctions = List.of();
    Listener.Factory<Event<?, ? extends List<Robot<?>>, ? extends List<Outcome>>> factory = Listener.Factory.deaf();
    NamedFunction<Event<?, ? extends List<Robot<?>>, ? extends List<Outcome>>, List<Outcome>> bestFitness = f("best.fitness", event -> Misc.first(event.getOrderedPopulation().firsts()).getFitness());
    //screen listener
    if (bestFileName == null) {
      factory = factory.and(new TabularPrinter<>(Misc.concat(List.of(
          basicFunctions,
          populationFunctions,
          NamedFunction.then(best(), basicIndividualFunctions),
          NamedFunction.then(best(), visualIndividualFunctions),
          NamedFunction.then(bestFitness, outcomesFunctions)
      ))));
    }
    //summarize params
    L.info("Experiment name: " + experimentName);
    L.info("Evolvers: " + evolverNames);
    L.info("Devo functions: " + devoFunctionNames);
    L.info("Terrains: " + terrainNames);
    //start iterations
    int nOfRuns = seeds.length * terrainNames.size() * devoFunctionNames.size() * evolverNames.size();
    int counter = 0;
    for (int seed : seeds) {
      for (String terrainName : terrainNames) {
        for (String devoFunctionName : devoFunctionNames) {
          for (String evolverName : evolverNames) {
            counter = counter + 1;
            final Random random = new Random(seed);
            //prepare keys
            Map<String, Object> keys = Map.ofEntries(
                Map.entry("experiment.name", experimentName),
                Map.entry("seed", seed),
                Map.entry("terrain", terrainName),
                Map.entry("devo.function", devoFunctionName),
                Map.entry("evolver", evolverName)
            );
          }
        }
      }
    }
  }

  private static List<NamedFunction<Event<?, ? extends List<Robot<?>>, ? extends List<Outcome>>, ?>> keyFunctions() {
    return List.of(
        eventAttribute("experiment.name"),
        eventAttribute("seed", "%2d"),
        eventAttribute("terrain"),
        eventAttribute("shape"),
        eventAttribute("sensor.config"),
        eventAttribute("mapper"),
        eventAttribute("transformation"),
        eventAttribute("evolver")
    );
  }

  private static List<NamedFunction<Event<?, ? extends List<Robot<?>>, ? extends List<Outcome>>, ?>> basicFunctions() {
    return List.of(
        iterations(),
        births(),
        fitnessEvaluations(),
        elapsedSeconds()
    );
  }

  private static List<NamedFunction<Individual<?, ? extends List<Robot<?>>, ? extends List<Outcome>>, ?>> basicIndividualFunctions(Function<List<Outcome>, Double> fitnessFunction) {
    NamedFunction<Individual<?, ? extends List<Robot<?>>, ? extends List<Outcome>>, ?> size = size().of(genotype());
    NamedFunction<Individual<?, ? extends List<Robot<?>>, ? extends List<Outcome>>, ? extends Grid<?>> firstShape = f("shape", (Function<Robot<?>, Grid<?>>) Robot::getVoxels)
        .of(f("first", (Function<List<Robot<?>>, Robot<?>>) l -> l.get(0)))
        .of(solution());
    NamedFunction<Individual<?, ? extends List<Robot<?>>, ? extends List<Outcome>>, ? extends Grid<?>> lastShape = f("shape", (Function<Robot<?>, Grid<?>>) Robot::getVoxels)
        .of(f("last", (Function<List<Robot<?>>, Robot<?>>) l -> l.get(l.size()-1)))
        .of(solution());
    return List.of(
        f("w", "%2d", (Function<Grid<?>, Number>) Grid::getW).of(firstShape),
        f("h", "%2d", (Function<Grid<?>, Number>) Grid::getW).of(firstShape),
        f("num.voxel", "%2d", (Function<Grid<?>, Number>) g -> g.count(Objects::nonNull)).of(firstShape),
        f("w", "%2d", (Function<Grid<?>, Number>) Grid::getW).of(lastShape),
        f("h", "%2d", (Function<Grid<?>, Number>) Grid::getW).of(lastShape),
        f("num.voxel", "%2d", (Function<Grid<?>, Number>) g -> g.count(Objects::nonNull)).of(lastShape),
        f("num.stages", "%2d", i -> i.getSolution().size()),
        size.reformat("%5d"),
        genotypeBirthIteration(),
        f("fitness", "%5.1f", fitnessFunction).of(fitness())
    );
  }

  private static Evolver<?, List<Robot<?>>, List<Outcome>> buildEvolver(String evolverName, String devoFunctionName, Robot<?>
      target, Function<Outcome, Double> outcomeMeasure) {
    PrototypedFunctionBuilder<Pair<?, Robot<?>>, Robot<?>> devoFunctionBuilder = getDevelopmentFunctionByName(devoFunctionName);
    return null; // TODO fix
  }

  private static EvolverBuilder<?> getEvolverBuilderFromName(String name) {
    return null; // TODO fix
  }

  private static PrototypedFunctionBuilder<Pair<?, Robot<?>>, Robot<?>> getDevelopmentFunctionByName(String name) {
    return null; // TODO fix
  }

}
