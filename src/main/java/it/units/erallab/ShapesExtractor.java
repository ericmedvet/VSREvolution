package it.units.erallab;

import it.units.erallab.hmsrobots.core.objects.Robot;
import it.units.erallab.hmsrobots.util.Grid;
import it.units.erallab.hmsrobots.util.SerializationUtils;
import it.units.malelab.jgea.Worker;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVPrinter;
import org.apache.commons.csv.CSVRecord;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class ShapesExtractor extends Worker {

  public ShapesExtractor(String[] args) {
    super(args);
  }

  private List<CSVRecord> records;

  private void readRecordsFromFile(String inputFileName) {
    //read data
    Reader reader = null;
    try {
      if (inputFileName != null) {
        reader = new FileReader(inputFileName);
      } else {
        reader = new InputStreamReader(System.in);
      }
      CSVParser csvParser = CSVFormat.DEFAULT.withDelimiter(';').withFirstRecordAsHeader().parse(reader);
      records = csvParser.getRecords();
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
  }

  @Override
  public void run() {
    String inputFileName = "last.txt";
    String outputFileName = "shapes.txt";
    String lastShapeFileName = "last-shapes.txt";

    List<String> colNames = List.of("seed", "devo.function", "evolver", "stage.min.dist", "stage.max.time", "terrain");
    String robotColumn = "best.fitness→devo.robots";
    CSVPrinter allShapesPrinter;
    CSVPrinter lastShapesPrinter;
    try {
      allShapesPrinter = new CSVPrinter(new PrintStream(outputFileName), CSVFormat.DEFAULT.withDelimiter(';'));
      lastShapesPrinter = new CSVPrinter(new PrintStream(lastShapeFileName), CSVFormat.DEFAULT.withDelimiter(';'));
    } catch (IOException e) {
      e.printStackTrace();
      return;
    }
    readRecordsFromFile(inputFileName);
    try {
      List<String> newHeaders = new ArrayList<>(colNames);
      newHeaders.addAll(List.of("stage", "x", "y"));
      allShapesPrinter.printRecord(newHeaders);
      lastShapesPrinter.printRecord(newHeaders);
    } catch (IOException e) {
      e.printStackTrace();
    }
    System.out.println("Total records: " + records.size());
    int i = 1;
    for (CSVRecord record : records) {
      List<String> values = colNames.stream().map(record::get).collect(Collectors.toList());
      String[] serializedRobots = record.get(robotColumn).split(",");
      List<Grid> grid = Arrays.stream(serializedRobots)
          .parallel()
          .map(s -> SerializationUtils.deserialize(s, Robot.class, SerializationUtils.Mode.GZIPPED_JSON))
          .map(Robot::getVoxels)
          .collect(Collectors.toList());
      List<List<String>> newRecords = IntStream.range(0, grid.size())
          .parallel()
          .mapToObj(j -> shapeToListOfCoordinates(grid.get(j), j + "", values))
          .flatMap(List::stream)
          .collect(Collectors.toList());
      List<List<String>> lastShapeRecords = shapeToListOfCoordinates(
          SerializationUtils.deserialize(serializedRobots[serializedRobots.length - 1], Robot.class, SerializationUtils.Mode.GZIPPED_JSON).getVoxels(),
          "last",
          values
      );
      try {
        allShapesPrinter.printRecords(newRecords);
        lastShapesPrinter.printRecords(lastShapeRecords);
      } catch (IOException e) {
        e.printStackTrace();
      }
      System.out.println(i);
      i++;
    }
  }

  private static List<List<String>> shapeToListOfCoordinates(Grid<?> grid, String devoStage, List<String> headers) {
    return grid.stream().filter(e -> e.getValue() != null)
        .map(e -> {
          List<String> l = new ArrayList<>(headers);
          l.add(devoStage);
          l.add(e.getX() + "");
          l.add(e.getY() + "");
          return l;
        }).collect(Collectors.toList());
  }

  public static void main(String[] args) {
    new ShapesExtractor(args);
  }
}
