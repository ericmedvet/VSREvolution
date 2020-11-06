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

import it.units.erallab.hmsrobots.core.objects.Robot;
import it.units.erallab.hmsrobots.tasks.locomotion.Locomotion;
import it.units.erallab.hmsrobots.util.Grid;
import it.units.erallab.hmsrobots.util.SerializationUtils;
import it.units.erallab.hmsrobots.viewers.*;
import it.units.erallab.hmsrobots.viewers.drawers.Ground;
import it.units.erallab.hmsrobots.viewers.drawers.SensorReading;
import it.units.erallab.hmsrobots.viewers.drawers.Voxel;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;
import org.apache.commons.lang3.tuple.Pair;
import org.dyn4j.dynamics.Settings;

import java.io.*;
import java.util.List;
import java.util.Set;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.logging.Logger;
import java.util.stream.Collectors;

import static it.units.malelab.jgea.core.util.Args.*;

/**
 * @author eric
 * @created 2020/08/19
 * @project VSREvolution
 */
public class VideoMaker {

  private static final Logger L = Logger.getLogger(VideoMaker.class.getName());

  /* example of invocation
    /usr/lib/jvm/jdk-14.0.1/bin/java -cp ~/IdeaProjects/VSREvolution/out/artifacts/VSREvolution_jar/VSREvolution.jar it.units.erallab.VideoMaker inputFile=vsrs-short-all-p10-t10-10.ser.txt globalPredicate=seed≡1^terrain≡uneven5^mapper≡centralized columnPredicates=evolver≡mlp-0-cmaes^body≡biped-4x3,evolver≡mlp-0-cmaes^body≡biped-cpg-4x3 rowPredicates=quant[births\;500\;2]≡0,quant[births\;500\;2]≡250,quant[births\;500\;2]≡500 transformation=identity
   */
  public static void main(String[] args) {
    //get params
    String inputFileName = a(args, "inputFile", "/home/eric/experiments/vsrs-stat.txt");
    String outputFileName = a(args, "outputFile", null);
    String serializedRobotColumn = a(args, "serializedRobotColumnName", "best.serialized.robot");
    String terrainName = a(args, "terrain", "flat");
    String transformationName = a(args, "transformation", "identity");
    double startTime = d(a(args, "startTime", "0.0"));
    double endTime = d(a(args, "endTime", "30.0"));
    int w = i(a(args, "w", "1024"));
    int h = i(a(args, "h", "768"));
    int frameRate = i(a(args, "frameRate", "30"));
    SerializationUtils.Mode mode = SerializationUtils.Mode.valueOf(a(args, "deserializationMode", SerializationUtils.Mode.GZIPPED_JSON.name()).toUpperCase());
    //read data
    Reader reader = null;
    List<CSVRecord> records = null;
    List<String> headers = null;
    try {
      if (inputFileName != null) {
        reader = new FileReader(inputFileName);
      } else {
        reader = new InputStreamReader(System.in);
      }
      CSVParser csvParser = CSVFormat.DEFAULT.withDelimiter(';').withFirstRecordAsHeader().parse(reader);
      records = csvParser.getRecords();
      headers = csvParser.getHeaderNames();
      reader.close();
    } catch (IOException e) {
      L.severe(String.format("Cannot read input data: %s", e));
      if (reader != null) {
        try {
          reader.close();
        } catch (IOException ioException) {
          //ignore
        }
      }
      System.exit(-1);
    }
    L.info(String.format("Read %d data lines from %s", records.size(), (inputFileName != null) ? inputFileName : "stdin"));
    //TODO maybe check that col nums is appropriate
    //find x- and y- values
    String xHeader = headers.get(0);
    String yHeader = headers.get(1);
    List<String> xValues = records.stream()
        .map(r -> r.get(xHeader))
        .distinct()
        .collect(Collectors.toList());
    List<String> yValues = records.stream()
        .map(r -> r.get(yHeader))
        .distinct()
        .collect(Collectors.toList());
    //build grid
    List<CSVRecord> finalRecords = records;
    Grid<List<String>> rawGrid = Grid.create(
        xValues.size(),
        yValues.size(),
        (x, y) -> finalRecords.stream()
            .filter(r -> r.get(xHeader).equals(xValues.get(x)) && r.get(yHeader).equals(yValues.get(y)))
            .map(r -> r.get(2))
            .collect(Collectors.toList())
    );
    //build named grid of robots
    Grid<Pair<String, Robot<?>>> namedRobotGrid = Grid.create(
        rawGrid.getW(),
        rawGrid.getH(),
        (x, y) -> Pair.of(
            xValues.get(x) + " " + yValues.get(y),
            it.units.erallab.hmsrobots.util.Utils.buildRobotTransformation(transformationName)
                .apply(SerializationUtils.deserialize(rawGrid.get(x, y).get(0), Robot.class, mode))
        )
    );
    //prepare problem
    Locomotion locomotion = new Locomotion(
        endTime,
        Locomotion.createTerrain(terrainName),
        new Settings()
    );
    //do simulations
    ScheduledExecutorService uiExecutor = Executors.newScheduledThreadPool(4);
    ExecutorService executor = Executors.newCachedThreadPool();
    GridSnapshotListener gridSnapshotListener = null;
    if (outputFileName == null) {
      gridSnapshotListener = new GridOnlineViewer(
          Grid.create(namedRobotGrid, Pair::getLeft),
          uiExecutor
      );
      ((GridOnlineViewer) gridSnapshotListener).start(3);
    } else {
      try {
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
                GraphicsDrawer.GeneralRenderingMode.TIME_INFO,
                GraphicsDrawer.GeneralRenderingMode.VOXEL_COMPOUND_CENTERS_INFO
            ))
        );
      } catch (IOException e) {
        L.severe(String.format("Cannot build grid file writer: %s", e));
        System.exit(-1);
      }
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

}
