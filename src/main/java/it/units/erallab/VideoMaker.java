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
import it.units.erallab.hmsrobots.tasks.Task;
import it.units.erallab.hmsrobots.tasks.locomotion.Locomotion;
import it.units.erallab.hmsrobots.util.Grid;
import it.units.erallab.hmsrobots.util.SerializationUtils;
import it.units.erallab.hmsrobots.viewers.*;
import it.units.erallab.hmsrobots.viewers.drawers.Drawer;
import it.units.erallab.hmsrobots.viewers.drawers.Drawers;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;
import org.apache.commons.lang3.tuple.Pair;
import org.dyn4j.dynamics.Settings;

import java.io.*;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.function.Function;
import java.util.logging.Logger;
import java.util.stream.IntStream;

import static it.units.malelab.jgea.core.util.Args.*;

/**
 * @author eric
 */
public class VideoMaker {

  private static final char DELIMITER = ';';

  private static final Logger L = Logger.getLogger(VideoMaker.class.getName());

  public static void main(String[] args) {

    String param = "springf";

    //get params
    List<String> terrainNames = l(a(args, "terrainNames", "flat"));
    String robotFileName = a(args, "robotFile", "biped-" + param + ".txt");
    String serializedRobotColumn = a(args, "serializedRobotColumnName", "best.solution.serialized");
    String descriptionColumn = a(args, "descriptionColumnName", "desc");
    String outputFileName = a(args, "outputFile", "D:\\Research\\Physical_parameters\\Video-def\\" + param + ".mov");
    SerializationUtils.Mode mode = SerializationUtils.Mode.valueOf(a(args, "deserializationMode", SerializationUtils.Mode.GZIPPED_JSON.name()).toUpperCase());
    boolean specifyRowsAndCols = a(args, "specifyRowsAndCols", "f").startsWith("t");
    String colNum = a(args, "nCol", "10");
    String rowNum = a(args, "nRow", "1");

    // video features
    double startTime = d(a(args, "startTime", "0.0"));
    double endTime = d(a(args, "endTime", "30.0"));
    int w = i(a(args, "w", "800"));
    int h = i(a(args, "h", "600"));
    int frameRate = i(a(args, "frameRate", "30"));
    String encoderName = a(args, "encoder", VideoUtils.EncoderFacility.JCODEC.name());

    // type of representation
    Function<String, Drawer> drawerSupplier = Drawers::basicWithMiniWorld;

    //read data
    Reader reader = null;
    List<CSVRecord> records = null;
    List<String> headers = null;
    try {
      if (robotFileName != null) {
        reader = new FileReader(robotFileName);
      } else {
        reader = new InputStreamReader(System.in);
      }
      CSVParser csvParser = CSVFormat.DEFAULT.withDelimiter(DELIMITER).withFirstRecordAsHeader().parse(reader);
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
    //check columns
    if (!headers.contains(serializedRobotColumn)) {
      L.severe(String.format("Cannot find serialized robot column %s in %s", serializedRobotColumn, headers));
      System.exit(-1);
    }

    int nCols, nRows;
    if (specifyRowsAndCols) {
      nCols = Integer.parseInt(colNum);
      nRows = Integer.parseInt(rowNum);
      if (nCols * nRows != terrainNames.size() * records.size()) {
        L.severe(String.format("Cannot use the specified grid size of %d cells for creating %d videos", nCols * nRows, terrainNames.size() * records.size()));
        System.exit(-1);
      }
    } else {
      if (records.size() == 1) {
        nRows = (int) Math.floor(Math.sqrt(terrainNames.size()));
        nCols = (int) Math.ceil((double) terrainNames.size() / nRows);
      } else if (terrainNames.size() == 1) {
        nRows = (int) Math.floor(Math.sqrt(records.size()));
        nCols = (int) Math.ceil((double) records.size() / nRows);
      } else {
        nCols = records.size();
        nRows = terrainNames.size();
      }
    }
    w = w * nCols;
    h = h * nRows;

    List<Robot> readRobots = new ArrayList<>();
    List<String> readRobotDescriptions = new ArrayList<>();
    records.forEach(
        record -> {
          readRobots.add(SerializationUtils.deserialize(record.get(serializedRobotColumn), Robot.class, mode));
          readRobotDescriptions.add(record.get(descriptionColumn));
        }
    );
    List<Robot> robots = new ArrayList<>(readRobots);
    List<String> robotDescriptions = new ArrayList<>(readRobotDescriptions);
    IntStream.range(1, terrainNames.size()).forEach(x -> {
      robots.addAll(readRobots);
      robotDescriptions.addAll(readRobotDescriptions);
    });
    List<CSVRecord> finalRecords = records;
    List<String> terrainRepeatedNames = terrainNames.stream().map(terrainName ->
        Collections.nCopies(finalRecords.size(), terrainName)).flatMap(List::stream).toList();

    List<String> descriptions = IntStream.range(0, robotDescriptions.size()).mapToObj(i ->
        robotDescriptions.get(i) + "\n" + " ").toList();

    List<Task<Robot, ?>> locomotionList = new ArrayList<>();
    terrainRepeatedNames.forEach(terrainName ->
        locomotionList.add(new Locomotion(endTime, Locomotion.createTerrain(terrainName), new Settings())));
    List<Pair<NamedValue<Robot>, Task<Robot, ?>>> pairsList = IntStream.range(0, descriptions.size()).mapToObj(i -> {
          Pair<NamedValue<Robot>, Task<Robot, ?>> pair = Pair.of(new NamedValue<>("", robots.get(i)), locomotionList.get(i));
          return pair;
        }
    ).toList();

    Grid<String> descriptionsGrid = Grid.create(nCols, nRows);
    Grid<Pair<NamedValue<Robot>, Task<Robot, ?>>> pairsGrid = Grid.create(descriptionsGrid);

    int listIndex = 0;
    for (int x = 0; x < nCols; x++) {
      for (int y = 0; y < nRows; y++) {
        if (listIndex >= descriptions.size()) {
          break;
        }
        descriptionsGrid.set(x, y, descriptions.get(listIndex));
        pairsGrid.set(x, y, pairsList.get(listIndex));
        listIndex = listIndex + 1;
      }
    }

    ScheduledExecutorService uiExecutor = Executors.newScheduledThreadPool(4);
    ExecutorService executor = Executors.newCachedThreadPool();
    GridSnapshotListener gridSnapshotListener;
    if (outputFileName == null) {
      gridSnapshotListener = new GridOnlineViewer(
          Grid.create(pairsGrid, p -> null),
          Grid.create(descriptionsGrid, drawerSupplier),
          uiExecutor
      );
      ((GridOnlineViewer) gridSnapshotListener).start(3);
    } else {
      gridSnapshotListener = new GridFileWriter(
          w, h, startTime, frameRate, VideoUtils.EncoderFacility.valueOf(encoderName.toUpperCase()),
          new File(outputFileName),
          Grid.create(descriptionsGrid, p -> p),
          Grid.create(descriptionsGrid, drawerSupplier)
      );
    }
    GridEpisodeRunner<Robot> runner = new GridEpisodeRunner<>(
        pairsGrid,
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