package it.units.erallab.devolocomotion;

import it.units.erallab.hmsrobots.core.objects.Robot;
import it.units.erallab.hmsrobots.tasks.devolocomotion.DevoLocomotion;
import it.units.erallab.hmsrobots.tasks.locomotion.Locomotion;
import it.units.erallab.hmsrobots.tasks.locomotion.Outcome;
import it.units.erallab.hmsrobots.util.Grid;
import it.units.erallab.hmsrobots.viewers.GridFileWriter;
import it.units.erallab.hmsrobots.viewers.VideoUtils;
import it.units.erallab.locomotion.Starter;
import it.units.malelab.jgea.core.Individual;
import it.units.malelab.jgea.core.evolver.Event;
import it.units.malelab.jgea.core.listener.Accumulator;
import it.units.malelab.jgea.core.listener.NamedFunction;
import it.units.malelab.jgea.core.listener.TableBuilder;
import it.units.malelab.jgea.core.util.ImagePlotters;
import it.units.malelab.jgea.core.util.Misc;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.Objects;
import java.util.Random;
import java.util.SortedMap;
import java.util.function.Function;
import java.util.function.UnaryOperator;
import java.util.logging.Logger;
import java.util.stream.Collectors;

import static it.units.malelab.jgea.core.listener.NamedFunctions.*;

/**
 * @author "Eric Medvet" on 2021/10/03 for VSREvolution
 */
public class NamedFunctions {

  private static final Logger L = Logger.getLogger(NamedFunctions.class.getName());

  private NamedFunctions() {}

  public static List<NamedFunction<Individual<?, ? extends UnaryOperator<Robot<?>>, ? extends List<Outcome>>, ?>> basicIndividualFunctions(Function<List<Outcome>, Double> fitnessFunction) {
    NamedFunction<Individual<?, ? extends UnaryOperator<Robot<?>>, ? extends List<Outcome>>, ?> size = size().of(genotype());
    NamedFunction<Individual<?, ? extends UnaryOperator<Robot<?>>, ? extends List<Outcome>>, ? extends Grid<?>> firstShape =
        f("shape", (Function<Outcome, Grid<?>>) o -> o.getObservations().get(o.getObservations().firstKey()).getVoxelPolies())
            .of(f("first", (Function<List<Outcome>, Outcome>) l -> l.get(0)))
            .of(fitness());
    NamedFunction<Individual<?, ? extends UnaryOperator<Robot<?>>, ? extends List<Outcome>>, ? extends Grid<?>> lastShape =
        f("shape", (Function<Outcome, Grid<?>>) o -> o.getObservations().get(o.getObservations().firstKey()).getVoxelPolies())
            .of(f("last", (Function<List<Outcome>, Outcome>) l -> l.get(l.size() - 1)))
            .of(fitness());
    return List.of(
        f("w", "%2d", (Function<Grid<?>, Number>) Grid::getW).of(firstShape),
        f("h", "%2d", (Function<Grid<?>, Number>) Grid::getW).of(firstShape),
        f("num.voxel", "%2d", (Function<Grid<?>, Number>) g -> g.count(Objects::nonNull)).of(firstShape),
        f("w", "%2d", (Function<Grid<?>, Number>) Grid::getW).of(lastShape),
        f("h", "%2d", (Function<Grid<?>, Number>) Grid::getW).of(lastShape),
        f("num.voxel", "%2d", (Function<Grid<?>, Number>) g -> g.count(Objects::nonNull)).of(lastShape),
        f("num.stages", "%2d", i -> i.getFitness().size()),
        size.reformat("%5d"),
        genotypeBirthIteration(),
        f("fitness", "%5.1f", fitnessFunction).of(fitness())
    );
  }

  public static List<NamedFunction<Event<?, ? extends UnaryOperator<Robot<?>>, ? extends List<Outcome>>, ?>> keysFunctions() {
    return List.of(
        eventAttribute("experiment.name"),
        eventAttribute("seed", "%2d"),
        eventAttribute("terrain"),
        eventAttribute("devo.function"),
        eventAttribute("evolver")
    );
  }

  public static List<NamedFunction<Event<?, ? extends UnaryOperator<Robot<?>>, ? extends List<Outcome>>, ?>> basicFunctions() {
    return List.of(
        iterations(),
        births(),
        fitnessEvaluations(),
        elapsedSeconds()
    );
  }

  public static List<NamedFunction<Event<?, ? extends UnaryOperator<Robot<?>>, ? extends List<Outcome>>, ?>> populationFunctions(Function<List<Outcome>, Double> fitnessFunction) {
    NamedFunction<Event<?, ? extends UnaryOperator<Robot<?>>, ? extends List<Outcome>>, ?> min = min(Double::compare).of(each(f("fitness", fitnessFunction).of(fitness()))).of(all());
    NamedFunction<Event<?, ? extends UnaryOperator<Robot<?>>, ? extends List<Outcome>>, ?> median = median(Double::compare).of(each(f("fitness", fitnessFunction).of(fitness()))).of(all());
    return List.of(
        size().of(all()),
        size().of(firsts()),
        size().of(lasts()),
        uniqueness().of(each(genotype())).of(all()),
        uniqueness().of(each(solution())).of(all()),
        uniqueness().of(each(fitness())).of(all()),
        min.reformat("%+4.1f"),
        median.reformat("%5.1f")
    );
  }

  public static List<NamedFunction<List<Outcome>, ?>> outcomesFunctions() {
    return List.of();
  }

  public static Accumulator.Factory<Event<?, ? extends UnaryOperator<Robot<?>>, ? extends List<Outcome>>, String> lastEventToString(Function<List<Outcome>, Double> fitnessFunction) {
    NamedFunction<Event<?, ? extends UnaryOperator<Robot<?>>, ? extends List<Outcome>>, List<Outcome>> bestFitness = f("best.fitness", event -> Misc.first(event.getOrderedPopulation().firsts()).getFitness());
    final List<NamedFunction<Event<?, ? extends UnaryOperator<Robot<?>>, ? extends List<Outcome>>, ?>> functions = Misc.concat(List.of(
        keysFunctions(),
        basicFunctions(),
        populationFunctions(fitnessFunction),
        NamedFunction.then(best(), basicIndividualFunctions(fitnessFunction)),
        NamedFunction.then(bestFitness, outcomesFunctions())
    ));
    return Accumulator.Factory.<Event<?, ? extends UnaryOperator<Robot<?>>, ? extends List<Outcome>>>last().then(
        e -> functions.stream()
            .map(f -> f.getName() + ": " + f.applyAndFormat(e))
            .collect(Collectors.joining("\n"))
    );
  }

  public static Accumulator.Factory<Event<?, ? extends UnaryOperator<Robot<?>>, ? extends List<Outcome>>, BufferedImage> fitnessPlot(Function<List<Outcome>, Double> fitnessFunction) {
    return new TableBuilder<Event<?, ? extends UnaryOperator<Robot<?>>, ? extends List<Outcome>>, Number>(List.of(
        iterations(),
        f("fitness", fitnessFunction).of(fitness()).of(best()),
        min(Double::compare).of(each(f("fitness", fitnessFunction).of(fitness()))).of(all()),
        median(Double::compare).of(each(f("fitness", fitnessFunction).of(fitness()))).of(all())
    )).then(ImagePlotters.xyLines(600, 400));
  }

  public static Accumulator.Factory<Event<?, ? extends UnaryOperator<Robot<?>>, ? extends List<Outcome>>, File> bestVideo(double stageMinDistance, double stageMaxT, double maxT) {
    return Accumulator.Factory.<Event<?, ? extends UnaryOperator<Robot<?>>, ? extends List<Outcome>>>last().then(
        event -> {
          Random random = new Random(0);
          SortedMap<Long, String> terrainSequence = Starter.getSequence((String) event.getAttributes().get("terrain"));
          String terrainName = terrainSequence.get(terrainSequence.lastKey());
          UnaryOperator<Robot<?>> solution = Misc.first(event.getOrderedPopulation().firsts()).getSolution();
          DevoLocomotion devoLocomotion = new DevoLocomotion(
              stageMinDistance, stageMaxT, maxT,
              Locomotion.createTerrain(terrainName.replace("-rnd", "-" + random.nextInt(10000))),
              Starter.PHYSICS_SETTINGS
          );
          File file;
          try {
            file = File.createTempFile("robot-video", ".mp4");
            GridFileWriter.save(devoLocomotion, solution, 300, 200, 0, 25, VideoUtils.EncoderFacility.JCODEC, file);
            file.deleteOnExit();
          } catch (IOException ioException) {
            L.warning(String.format("Cannot save video of best: %s", ioException));
            return null;
          }
          return file;
        }
    );
  }

  public static List<NamedFunction<Individual<?,? extends UnaryOperator<Robot<?>>,? extends List<Outcome>>,?>> visualIndividualFunctions() {
    return List.of();
  }
}
