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

package it.units.erallab.locomotion;

import it.units.erallab.hmsrobots.behavior.BehaviorUtils;
import it.units.erallab.hmsrobots.core.objects.Robot;
import it.units.erallab.hmsrobots.core.objects.Voxel;
import it.units.erallab.hmsrobots.core.snapshots.VoxelPoly;
import it.units.erallab.hmsrobots.tasks.locomotion.Locomotion;
import it.units.erallab.hmsrobots.tasks.locomotion.Outcome;
import it.units.erallab.hmsrobots.util.Grid;
import it.units.erallab.hmsrobots.util.RobotUtils;
import it.units.erallab.hmsrobots.util.SerializationUtils;
import it.units.erallab.hmsrobots.viewers.GridFileWriter;
import it.units.erallab.hmsrobots.viewers.NamedValue;
import it.units.erallab.hmsrobots.viewers.VideoUtils;
import it.units.erallab.locomotion.Starter.ValidationOutcome;
import it.units.malelab.jgea.core.listener.Accumulator;
import it.units.malelab.jgea.core.listener.AccumulatorFactory;
import it.units.malelab.jgea.core.listener.NamedFunction;
import it.units.malelab.jgea.core.listener.TableBuilder;
import it.units.malelab.jgea.core.solver.Individual;
import it.units.malelab.jgea.core.solver.state.POSetPopulationState;
import it.units.malelab.jgea.core.util.*;
import org.dyn4j.dynamics.Settings;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.*;
import java.util.function.Function;
import java.util.logging.Logger;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static it.units.malelab.jgea.core.listener.NamedFunctions.*;

/**
 * @author eric
 */
public class NamedFunctions {

  private static final Logger L = Logger.getLogger(NamedFunctions.class.getName());

  private NamedFunctions() {
  }

  public static List<NamedFunction<? super POSetPopulationState<?, Robot, Outcome>, ?>> basicFunctions() {
    return List.of(iterations(), births(), fitnessEvaluations(), elapsedSeconds());
  }

  public static List<NamedFunction<? super Outcome, ?>> basicOutcomeFunctions() {
    return List.of(
        f("computation.time", "%4.2f", Outcome::getComputationTime),
        f("distance", "%5.1f", Outcome::getDistance),
        f("velocity", "%5.1f", Outcome::getVelocity)
    );
  }

  public static NamedFunction<POSetPopulationState<?, Robot, Outcome>, Individual<?, Robot, Outcome>> best() {
    return ((NamedFunction<POSetPopulationState<?, Robot, Outcome>, Individual<?, Robot, Outcome>>) state -> Misc.first(
        state.getPopulation().firsts())).rename("best");
  }

  public static AccumulatorFactory<POSetPopulationState<?, Robot, Outcome>, File, Map<String, Object>> bestVideo(
      double transientTime, double episodeTime, Settings settings
  ) {
    return AccumulatorFactory.last((state, keys) -> {
      Random random = new Random(0);
      SortedMap<Long, String> terrainSequence = Starter.getSequence((String) keys.get("terrain"));
      SortedMap<Long, String> transformationSequence = Starter.getSequence((String) keys.get("transformation"));
      String terrainName = terrainSequence.get(terrainSequence.lastKey());
      String transformationName = transformationSequence.get(transformationSequence.lastKey());
      Robot robot = SerializationUtils.clone(Misc.first(state.getPopulation().firsts()).solution());
      robot = RobotUtils.buildRobotTransformation(transformationName, random).apply(robot);
      Locomotion locomotion = new Locomotion(
          episodeTime,
          Locomotion.createTerrain(terrainName.replace("-rnd", "-" + random.nextInt(10000))),
          settings
      );
      File file;
      try {
        file = File.createTempFile("robot-video", ".mp4");
        String robotName = keys.get("sensor.config") + " " + keys.get("mapper") + " (" + keys.get("seed") + ")";
        GridFileWriter.save(
            locomotion,
            Grid.create(1, 1, new NamedValue<>(robotName, robot)),
            300,
            200,
            transientTime,
            25,
            VideoUtils.EncoderFacility.JCODEC,
            file
        );
        file.deleteOnExit();
      } catch (IOException ioException) {
        L.warning(String.format("Cannot save video of best: %s", ioException));
        return null;
      }
      return file;
    });
  }

  public static AccumulatorFactory<POSetPopulationState<?, Robot, Outcome>, BufferedImage, Map<String, Object>> centerPositionPlot() {
    return ((AccumulatorFactory<POSetPopulationState<?, Robot, Outcome>, POSetPopulationState<?, Robot, Outcome>, Map<String, Object>>) keys -> Accumulator.last()).then(
        state -> {
          Outcome o = Misc.first(state.getPopulation().firsts()).fitness();
          Table<Number> table = new ArrayTable<>(List.of("x", "y", "terrain.y"));
          o.getObservations().values().forEach(obs -> {
            VoxelPoly poly = BehaviorUtils.getCentralElement(obs.voxelPolies());
            table.addRow(List.of(poly.center().x(), poly.center().y(), obs.terrainHeight()));
          });
          return ImagePlotters.xyLines(600, 400).apply(table);
        });
  }

  public static List<NamedFunction<? super Outcome, ?>> detailedOutcomeFunctions(
      double spectrumMinFreq, double spectrumMaxFreq, int spectrumSize
  ) {
    return Misc.concat(List.of(
        List.of(
            f("corrected.efficiency", "%5.2f", Outcome::getCorrectedEfficiency),
            f("area.ratio.power", "%5.1f", Outcome::getAreaRatioPower),
            f("control.power", "%5.1f", Outcome::getControlPower)
        ),
        NamedFunction.then(
            cachedF(
                "center.x.spectrum",
                (Outcome o) -> new ArrayList<>(o.getCenterXVelocitySpectrum(
                        spectrumMinFreq,
                        spectrumMaxFreq,
                        spectrumSize
                    )
                    .values())
            ),
            IntStream.range(0, spectrumSize)
                .mapToObj(it.units.malelab.jgea.core.listener.NamedFunctions::nth)
                .collect(Collectors.toList())
        ),
        NamedFunction.then(
            cachedF(
                "center.y.spectrum",
                (Outcome o) -> new ArrayList<>(o.getCenterYVelocitySpectrum(
                        spectrumMinFreq,
                        spectrumMaxFreq,
                        spectrumSize
                    )
                    .values())
            ),
            IntStream.range(0, spectrumSize)
                .mapToObj(it.units.malelab.jgea.core.listener.NamedFunctions::nth)
                .collect(Collectors.toList())
        ),
        NamedFunction.then(
            cachedF(
                "center.angle.spectrum",
                (Outcome o) -> new ArrayList<>(o.getCenterAngleSpectrum(spectrumMinFreq, spectrumMaxFreq, spectrumSize)
                    .values())
            ),
            IntStream.range(0, spectrumSize)
                .mapToObj(it.units.malelab.jgea.core.listener.NamedFunctions::nth)
                .collect(Collectors.toList())
        ),
        NamedFunction.then(
            cachedF(
                "footprints.spectra",
                (Outcome o) -> o.getFootprintsSpectra(4, spectrumMinFreq, spectrumMaxFreq, spectrumSize)
                    .stream()
                    .map(SortedMap::values)
                    .flatMap(Collection::stream)
                    .collect(Collectors.toList())
            ),
            IntStream.range(0, 4 * spectrumSize)
                .mapToObj(it.units.malelab.jgea.core.listener.NamedFunctions::nth)
                .collect(Collectors.toList())
        )
    ));
  }

  public static AccumulatorFactory<POSetPopulationState<?, Robot, Outcome>, BufferedImage, Map<String, Object>> fitnessPlot(
      Function<Outcome, Double> fitnessFunction
  ) {
    return new TableBuilder<POSetPopulationState<?, Robot, Outcome>, Number, Map<String, Object>>(List.of(
        iterations(),
        f("fitness", fitnessFunction).of(fitness()).of(best()),
        min(Double::compare).of(each(f("fitness", fitnessFunction).of(fitness()))).of(all()),
        median(Double::compare).of(each(f("fitness", fitnessFunction).of(fitness()))).of(all())
    ), List.of()).then(t -> ImagePlotters.xyLines(600, 400).apply(t));
  }

  public static NamedFunction<Pair<POSetPopulationState<?, Robot, Outcome>, Individual<?, Robot, Outcome>>, Individual<?, Robot, Outcome>> individualExtractor() {
    return f(
        "individual",
        Pair::second
    );
  }

  public static List<NamedFunction<? super Individual<?, Robot, Outcome>, ?>> individualFunctions(Function<Outcome, Double> fitnessFunction) {
    NamedFunction<Individual<?, Robot, Outcome>, ?> size = size().of(genotype());
    NamedFunction<Robot, Grid<Voxel>> shape = f("shape", Robot::getVoxels);
    NamedFunction<Grid<Voxel>, Number> w = f("w", "%2d", Grid::getW);
    NamedFunction<Grid<Voxel>, Number> h = f("h", "%2d", Grid::getH);
    NamedFunction<Grid<Voxel>, Number> numVoxel = f("num.voxel", "%2d", g -> g.count(Objects::nonNull));
    return List.of(
        w.of(shape).of(solution()),
        h.of(shape).of(solution()),
        numVoxel.of(shape).of(solution()),
        size.reformat("%5d"),
        f("genotype.birth.iteration", "%4d", Individual::genotypeBirthIteration),
        f("fitness", "%5.1f", fitnessFunction).of(fitness())
    );
  }

  public static List<NamedFunction<? super Map<String, Object>, ?>> keysFunctions() {
    return List.of(
        attribute("experiment.name"),
        attribute("seed").reformat("%2d"),
        attribute("terrain"),
        attribute("shape"),
        attribute("sensor.config"),
        attribute("mapper"),
        attribute("transformation"),
        attribute("evolver"),
        attribute("episode.time"),
        attribute("episode.transient.time")
    );
  }

  public static AccumulatorFactory<POSetPopulationState<?, Robot, Outcome>, String, Map<String, Object>> lastEventToString(
      Function<Outcome, Double> fitnessFunction
  ) {
    final List<NamedFunction<? super POSetPopulationState<?, Robot, Outcome>, ?>> functions = Misc.concat(List.of(
        basicFunctions(),
        populationFunctions(fitnessFunction),
        best().then(individualFunctions(fitnessFunction)),
        basicOutcomeFunctions().stream().map(f -> f.of(fitness()).of(best())).toList()
    ));
    List<NamedFunction<? super Map<String, Object>, ?>> keysFunctions = keysFunctions();
    return AccumulatorFactory.last((state, keys) -> {
      String s = keysFunctions.stream()
          .map(f -> String.format(f.getName() + ": " + f.getFormat(), f.apply(keys)))
          .collect(Collectors.joining("\n"));
      s = s + functions.stream()
          .map(f -> String.format(f.getName() + ": " + f.getFormat(), f.apply(state)))
          .collect(Collectors.joining("\n"));
      return s;
    });
  }

  public static List<NamedFunction<? super POSetPopulationState<?, Robot, Outcome>, ?>> populationFunctions(
      Function<Outcome, Double> fitnessFunction
  ) {
    NamedFunction<? super POSetPopulationState<?, Robot, Outcome>, ?> min = min(Double::compare).of(each(f(
        "fitness",
        fitnessFunction
    ).of(fitness()))).of(all());
    NamedFunction<? super POSetPopulationState<?, Robot, Outcome>, ?> median = median(Double::compare).of(each(f(
        "fitness",
        fitnessFunction
    ).of(fitness()))).of(all());
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

  public static Function<POSetPopulationState<?,Robot,Outcome>,Collection<Pair<POSetPopulationState<?, Robot, Outcome>, Individual<?, Robot, Outcome>>>> populationSplitter() {
    return state -> {
      List<Pair<POSetPopulationState<?, Robot, Outcome>, Individual<?, Robot, Outcome>>> list = new ArrayList<>();
      state.getPopulation().all().forEach(i -> list.add(Pair.of(state, i)));
      return list;
    };
  }

  public static List<NamedFunction<? super Individual<?, Robot, Outcome>, ?>> serializationFunction(boolean flag) {
    if (!flag) {
      return List.of();
    }
    return List.of(f("serialized", r -> SerializationUtils.serialize(r, SerializationUtils.Mode.GZIPPED_JSON)).of(
        solution()));
  }

  public static NamedFunction<Pair<POSetPopulationState<?, Robot, Outcome>, Individual<?, Robot, Outcome>>, POSetPopulationState<?, Robot, Outcome>> stateExtractor() {
    return f(
        "state",
        Pair::first
    );
  }

  public static Function<? super POSetPopulationState<?, Robot, Outcome>, Collection<ValidationOutcome>> validation(
      List<String> validationTerrainNames,
      List<String> validationTransformationNames,
      List<Integer> seeds,
      double episodeTime
  ) {
    return state -> {
      Robot robot = SerializationUtils.clone(Misc.first(state.getPopulation().firsts()).solution());
      List<ValidationOutcome> validationOutcomes = new ArrayList<>();
      for (String validationTerrainName : validationTerrainNames) {
        for (String validationTransformationName : validationTransformationNames) {
          for (int seed : seeds) {
            Random random = new Random(seed);
            robot = RobotUtils.buildRobotTransformation(validationTransformationName, random).apply(robot);
            Function<Robot, Outcome> validationLocomotion = Starter.buildLocomotionTask(
                validationTerrainName,
                episodeTime,
                random,
                false
            );
            Outcome outcome = validationLocomotion.apply(robot);
            validationOutcomes.add(new ValidationOutcome(
                state,
                Map.ofEntries(
                    Map.entry("validation.terrain", validationTerrainName),
                    Map.entry("validation.transformation", validationTransformationName),
                    Map.entry("validation.seed", seed),
                    Map.entry("validation.episode.time", episodeTime)
                ),
                outcome
            ));
          }
        }
      }
      return validationOutcomes;
    };
  }

  public static List<NamedFunction<? super Individual<?, Robot, Outcome>, ?>> visualIndividualFunctions() {
    return List.of(f(
        "minimap",
        "%4s",
        (Function<Grid<?>, String>) g -> TextPlotter.binaryMap(
            g.toArray(Objects::nonNull),
            (int) Math.min(Math.ceil((float) g.getW() / (float) g.getH() * 2f), 4)
        )
    ).of(f(
        "shape",
        (Function<Robot, Grid<?>>) Robot::getVoxels
    )).of(solution()), f(
        "average.posture.minimap",
        "%2s",
        (Function<Outcome, String>) o -> TextPlotter.binaryMap(o.getAveragePosture(8).toArray(b -> b), 2)
    ).of(fitness()));
  }

  public static List<NamedFunction<? super Outcome, ?>> visualOutcomeFunctions(
      double spectrumMinFreq,
      double spectrumMaxFreq
  ) {
    return Misc.concat(List.of(
        List.of(
            cachedF(
                "center.x.spectrum",
                "%4.4s",
                o -> TextPlotter.barplot(new ArrayList<>(o.getCenterXVelocitySpectrum(
                    spectrumMinFreq,
                    spectrumMaxFreq,
                    4
                ).values()))
            ),
            cachedF(
                "center.y.spectrum",
                "%4.4s",
                o -> TextPlotter.barplot(new ArrayList<>(o.getCenterYVelocitySpectrum(
                    spectrumMinFreq,
                    spectrumMaxFreq,
                    4
                ).values()))
            ),
            cachedF(
                "center.angle.spectrum",
                "%4.4s",
                o -> TextPlotter.barplot(new ArrayList<>(o.getCenterAngleSpectrum(spectrumMinFreq, spectrumMaxFreq, 4)
                    .values()))
            )
        ),
        NamedFunction.then(
            cachedF("footprints", o -> o.getFootprintsSpectra(3, spectrumMinFreq, spectrumMaxFreq, 4)),
            List.of(
                cachedF("left.spectrum", "%4.4s", l -> TextPlotter.barplot(new ArrayList<>(l.get(0).values()))),
                cachedF("center.spectrum", "%4.4s", l -> TextPlotter.barplot(new ArrayList<>(l.get(1).values()))),
                cachedF("right.spectrum", "%4.4s", l -> TextPlotter.barplot(new ArrayList<>(l.get(2).values())))
            )
        )
    ));
  }

  public static List<NamedFunction<? super POSetPopulationState<?, Robot, Outcome>, ?>> visualPopulationFunctions(
      Function<Outcome, Double> fitnessFunction
  ) {
    return List.of(
        hist(8).of(each(f("fitness", fitnessFunction).of(fitness()))).of(all()),
        hist(8).of(each(f("num.voxels", (Function<Grid<?>, Number>) g -> g.count(Objects::nonNull)).of(f(
            "shape",
            (Function<Robot, Grid<?>>) Robot::getVoxels
        )).of(solution()))).of(all())
    );
  }

}
