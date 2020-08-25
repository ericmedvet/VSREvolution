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

import com.google.common.collect.Lists;
import com.google.common.collect.Sets;
import it.units.erallab.hmsrobots.core.objects.Robot;
import it.units.erallab.hmsrobots.tasks.Locomotion;
import it.units.erallab.hmsrobots.util.Grid;
import it.units.erallab.hmsrobots.viewers.*;
import it.units.erallab.hmsrobots.viewers.drawers.Ground;
import it.units.erallab.hmsrobots.viewers.drawers.SensorReading;
import it.units.erallab.hmsrobots.viewers.drawers.Voxel;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVRecord;
import org.apache.commons.lang3.tuple.Pair;
import org.dyn4j.dynamics.Settings;

import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.io.Reader;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.function.Predicate;
import java.util.logging.Logger;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import static it.units.malelab.jgea.core.util.Args.*;

/**
 * @author eric
 * @created 2020/08/19
 * @project VSREvolution
 */
public class VideoMaker {

  private static final String PREDICATE_DEF = "≡";
  private static final String PREDICATE_SEP = "\\^";
  private static final String PREDICATE_F_SEP = ";";
  private static final String[] PREDICATE_F_PARS = new String[]{"[", "]"};

  private static final Logger L = Logger.getLogger(VideoMaker.class.getName());

  /* example of invocation
    /usr/lib/jvm/jdk-14.0.1/bin/java -cp ~/IdeaProjects/VSREvolution/out/artifacts/VSREvolution_jar/VSREvolution.jar it.units.erallab.VideoMaker inputFile=vsrs-short-all-p10-t10-10.ser.txt globalPredicate=seed≡1^terrain≡uneven5^mapper≡centralized columnPredicates=evolver≡mlp-0-cmaes^body≡biped-4x3,evolver≡mlp-0-cmaes^body≡biped-cpg-4x3 rowPredicates=quant[births\;500\;2]≡0,quant[births\;500\;2]≡250,quant[births\;500\;2]≡500
   */
  public static void main(String[] args) throws IOException {
    //get params
    String inputFileName = a(args, "inputFile", null);
    String outputFileName = a(args, "outputFile", null);
    String serializedRobotColumn = a(args, "serializedRobotColumnName", "best.serialized.robot");
    String terrain = a(args, "terrain", "flat");
    double episodeTime = d(a(args, "episodeT", "10.0"));
    int w = i(a(args, "w", "1024"));
    int h = i(a(args, "g", "768"));
    int frameRate = i(a(args, "frameRate", "30"));
    //read predicates
    Map<String, Predicate<String>> globalPredicate = Arrays.stream(a(args, "globalPredicate", "").split(PREDICATE_SEP)).sequential()
        .map(VideoMaker::buildPredicate)
        .collect(Collectors.toMap(Map.Entry::getKey, Map.Entry::getValue));
    List<Map<String, Predicate<String>>> columnPredicates = l(a(args, "columnPredicates", ""))
        .stream()
        .map(pairs -> Arrays.stream(pairs.split(PREDICATE_SEP)).sequential()
            .map(VideoMaker::buildPredicate)
            .collect(Collectors.toMap(Map.Entry::getKey, Map.Entry::getValue)))
        .collect(Collectors.toList());
    List<Map<String, Predicate<String>>> rowPredicates = l(a(args, "rowPredicates", ""))
        .stream()
        .map(pairs -> Arrays.stream(pairs.split(PREDICATE_SEP)).sequential()
            .map(VideoMaker::buildPredicate)
            .collect(Collectors.toMap(Map.Entry::getKey, Map.Entry::getValue)))
        .collect(Collectors.toList());
    //read data from file
    Reader reader = new FileReader(inputFileName);
    List<Map<String, String>> records = CSVFormat.DEFAULT.withDelimiter(';').withFirstRecordAsHeader().parse(reader).getRecords().stream()
        .map(CSVRecord::toMap).collect(Collectors.toList());
    L.info(String.format("Read %d data lines from %s", records.size(), inputFileName));
    //filter data
    Grid<List<Map<String, String>>> recordGrid = Grid.create(
        columnPredicates.size(),
        rowPredicates.size(),
        (x, y) -> records.stream()
            .filter(r -> match(r, globalPredicate))
            .filter(r -> match(r, columnPredicates.get(x)))
            .filter(r -> match(r, rowPredicates.get(y)))
            .collect(Collectors.toList())
    );
    L.info(String.format(
        "Grid # of records: %s",
        recordGrid.stream()
            .map(e -> String.format("(%d,%d)->%d", e.getX(), e.getY(), e.getValue().size()))
            .collect(Collectors.joining(", "))
    ));
    //build named grid of robots
    Set<String> relevantKeys = Stream.concat(columnPredicates.stream(), rowPredicates.stream())
        .map(Map::keySet)
        .reduce(Sets::union)
        .orElse(Set.of());
    Grid<Pair<String, Robot<?>>> namedRobotGrid = Grid.create(
        recordGrid,
        r -> r.isEmpty() ? null : Pair.of(
            r.get(0).entrySet().stream()
                .filter(e -> relevantKeys.contains(e.getKey()))
                .map(e -> e.toString())
                .collect(Collectors.joining("\n")),
            Utils.safelyDeserialize(r.get(0).get(serializedRobotColumn), Robot.class)
        )
    );
    //prepare problem
    Locomotion locomotion = new Locomotion(
        episodeTime,
        Locomotion.createTerrain(terrain),
        Lists.newArrayList(Locomotion.Metric.TRAVELED_X_DISTANCE),
        new Settings()
    );
    //do simulations
    ScheduledExecutorService uiExecutor = Executors.newScheduledThreadPool(4);
    ExecutorService executor = Executors.newCachedThreadPool();
    GridSnapshotListener gridSnapshotListener;
    if (outputFileName == null) {
      gridSnapshotListener = new GridOnlineViewer(
          Grid.create(namedRobotGrid, Pair::getLeft),
          uiExecutor
      );
      ((GridOnlineViewer) gridSnapshotListener).start(5);
    } else {
      gridSnapshotListener = new GridFileWriter(
          w, h, frameRate,
          new File(outputFileName),
          Grid.create(namedRobotGrid, Pair::getLeft),
          uiExecutor,
          GraphicsDrawer.build().setConfigurable("drawers", List.of(
              it.units.erallab.hmsrobots.viewers.drawers.Robot.build(),
              Voxel.build(),
              Ground.build(),
              SensorReading.build()
          )).setConfigurable("generalRenderingModes", Set.of(
              GraphicsDrawer.GeneralRenderingMode.GRID_MAJOR,
              GraphicsDrawer.GeneralRenderingMode.TIME_INFO,
              GraphicsDrawer.GeneralRenderingMode.VOXEL_COMPOUND_CENTERS_INFO
          ))
      );
    }
    GridEpisodeRunner<Robot<?>> runner = new GridEpisodeRunner<>(
        namedRobotGrid,
        locomotion,
        gridSnapshotListener,
        executor
    );
    runner.run();
    if (outputFileName != null) {
      executor.shutdownNow();
      uiExecutor.shutdownNow();
    }
  }

  private static boolean match(Map<String, String> values, Map<String, Predicate<String>> predicates) {
    for (String key : predicates.keySet()) {
      if (!values.containsKey(key)) {
        return false;
      }
      if (!predicates.get(key).test(values.get(key))) {
        return false;
      }
    }
    return true;
  }

  //expects a ^-piece of something like: evolver≡e1^body≡b1^quant(births|1000|2)≡3
  private static Map.Entry<String, Predicate<String>> buildPredicate(String string) {
    String left = string.split(PREDICATE_DEF)[0];
    String right = string.split(PREDICATE_DEF)[1];
    if (!left.contains(PREDICATE_F_PARS[0])) {
      return Map.entry(left, s -> s.equals(right));
    }
    String fName = left.substring(0, left.indexOf(PREDICATE_F_PARS[0]));
    String[] fArgs = left.substring(left.indexOf(PREDICATE_F_PARS[0]) + 1, left.indexOf(PREDICATE_F_PARS[1])).split(PREDICATE_F_SEP);
    if (fName.equals("quant")) {
      double range = Double.parseDouble(fArgs[1]);
      int bins = Integer.parseInt(fArgs[2]);
      return Map.entry(
          fArgs[0],
          s -> Math.floor(Double.parseDouble(s) / range * bins) * range / bins == Double.parseDouble(right)
      );
    }
    return null;
  }

}
