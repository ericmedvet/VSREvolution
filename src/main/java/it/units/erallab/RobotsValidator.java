package it.units.erallab;

import it.units.erallab.hmsrobots.core.controllers.DistributedSensing;
import it.units.erallab.hmsrobots.core.controllers.TimedRealFunction;
import it.units.erallab.hmsrobots.core.objects.Robot;
import it.units.erallab.hmsrobots.core.objects.Voxel;
import it.units.erallab.hmsrobots.tasks.locomotion.Locomotion;
import it.units.erallab.hmsrobots.tasks.locomotion.Outcome;
import it.units.erallab.hmsrobots.util.Grid;
import it.units.erallab.hmsrobots.util.RobotUtils;
import it.units.erallab.hmsrobots.util.SerializationUtils;
import it.units.erallab.locomotion.NamedFunctions;
import it.units.erallab.locomotion.Starter;
import it.units.malelab.jgea.Worker;
import it.units.malelab.jgea.core.listener.NamedFunction;
import it.units.malelab.jgea.core.util.Pair;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVPrinter;
import org.apache.commons.csv.CSVRecord;

import java.io.*;
import java.util.ArrayList;
import java.util.List;
import java.util.logging.Logger;
import java.util.stream.Collectors;

import static it.units.malelab.jgea.core.util.Args.d;

public class RobotsValidator extends Worker {

  private List<CSVRecord> records;
  private CSVPrinter printer;

  public RobotsValidator(String[] args) {
    super(args);
  }

  private List<String> readRecordsFromFile(String inputFileName) {
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
      List<String> oldHeaders = csvParser.getHeaderNames();
      reader.close();
      return oldHeaders;
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
      return null;
    }
  }

  @Override
  public void run() {
    Logger logger = Logger.getAnonymousLogger();
    System.setProperty("java.util.logging.SimpleFormatter.format", "[%1$tF %1$tT] [%4$-6s] %5$s %n");
    // params
    String inputFileName = a("inputFile", "C:\\Users\\giorg\\Documents\\UNITS\\PHD\\HomoHebbian\\hebbian-homo-new\\last.txt");
    String serializedRobotColumnName = a("serializedRobotColumn", "best→solution→serialized");
    String outputFileName = a("outputFile", "C:\\Users\\giorg\\Documents\\UNITS\\PHD\\HomoHebbian\\hebbian-homo-new\\validation-new-shapes.txt");
    double episodeTime = d(a("episodeTime", "60"));
    double episodeTransientTime = d(a("episodeTransientTime", "0"));
    List<String> headersToKeep = List.of("seed", "shape", "sensor.config", "mapper", "evolver");
    // create printer
    try {
      printer = new CSVPrinter(new PrintStream(outputFileName), CSVFormat.DEFAULT.withDelimiter(';'));
    } catch (IOException e) {
      e.printStackTrace();
      return;
    }
    // parse old file and print headers to new file
    List<String> oldHeaders = readRecordsFromFile(inputFileName);
    oldHeaders = oldHeaders.stream().filter(headersToKeep::contains).collect(Collectors.toList());
    List<NamedFunction<Outcome, ?>> basicOutcomeFunctions = NamedFunctions.basicOutcomeFunctions();
    List<String> basicOutcomeFunctionsNames = basicOutcomeFunctions.stream().map(NamedFunction::getName).toList();
    List<String> headers = oldHeaders.stream().map(h -> "event." + h).collect(Collectors.toList());
    headers.addAll(List.of("keys.validation.terrain", "keys.validation.shape"));
    headers.addAll(basicOutcomeFunctionsNames.stream().map(h -> "outcome." + h).toList());

    try {
      printer.printRecord(headers);
    } catch (IOException e) {
      e.printStackTrace();
    }
    int validationsCounter = 0;
    for (CSVRecord record : records) {
      // read robot and record
      String sensorConfig = record.get("sensor.config");
      String initialShape = record.get("shape");
      String robotString = record.get(serializedRobotColumnName);
      Robot robot = SerializationUtils.deserialize(robotString, Robot.class, SerializationUtils.Mode.GZIPPED_JSON);
      DistributedSensing distributedSensing = (DistributedSensing) robot.getController();

      List<String> oldRecord = oldHeaders.stream().map(record::get).toList();

      TimedRealFunction function = distributedSensing.getFunctions().get(0, 0);
      List<Pair<Grid<Boolean>, String>> shapes = buildShapes(initialShape);

      // validate robot with all new shapes
      List<List<Object>> rows = shapes.stream().parallel()
          .map(pair -> {
            Grid<Boolean> shape = pair.first();
            String shapeName = pair.second();
            Grid<Voxel> body = RobotUtils.buildSensorizingFunction(sensorConfig).apply(shape);
            DistributedSensing controller = new DistributedSensing(body, 2);
            for (Grid.Entry<Voxel> entry : body) {
              if (entry.value() != null) {
                controller.getFunctions().set(entry.key().x(), entry.key().y(), SerializationUtils.clone(function));
              }
            }

            List<Object> cells = new ArrayList<>(oldRecord);
            cells.addAll(List.of("flat", shapeName));
            Locomotion locomotion = new Locomotion(episodeTime, Locomotion.createTerrain("flat"), Starter.PHYSICS_SETTINGS);
            Outcome outcome = locomotion.apply(new Robot(controller, SerializationUtils.clone(body)));
            cells.addAll(basicOutcomeFunctions.stream().map(f -> f.apply(outcome.subOutcome(episodeTransientTime, episodeTime))).toList());
            return cells;
          }).toList();
      rows.forEach(row -> {
        try {
          printer.printRecord(row);
        } catch (IOException e) {
          e.printStackTrace();
        }
      });

      logger.info(String.format("%2d/%2d", ++validationsCounter, records.size()));
    }
    // close printer
    try {
      printer.flush();
      printer.close();
    } catch (IOException e) {
      e.printStackTrace();
    }
  }

  public static void main(String[] args) {
    new RobotsValidator(args);
  }

  private static List<Pair<Grid<Boolean>, String>> buildWorms() {
    Grid<Boolean> shape0 = RobotUtils.buildShape("worm-7x1");
    Grid<Boolean> shape1 = RobotUtils.buildShape("worm-7x2");
    Grid<Boolean> shape2 = RobotUtils.buildShape("worm-5x1");
    Grid<Boolean> shape3 = RobotUtils.buildShape("worm-5x2");
    Grid<Boolean> shape4 = RobotUtils.buildShape("worm-9x1");
    Grid<Boolean> shape5 = RobotUtils.buildShape("worm-9x2");

    List<Pair<Grid<Boolean>, String>> shapes = new ArrayList<>();
    shapes.add(Pair.of(shape0, "worm-0"));
    shapes.add(Pair.of(shape1, "worm-1"));
    shapes.add(Pair.of(shape2, "worm-2"));
    shapes.add(Pair.of(shape3, "worm-3"));
    shapes.add(Pair.of(shape4, "worm-4"));
    shapes.add(Pair.of(shape5, "worm-5"));
    return shapes;
  }

  private static List<Pair<Grid<Boolean>, String>> buildBipeds() {
    Grid<Boolean> shape0 = RobotUtils.buildShape("biped-4x3");
    Grid<Boolean> shape1 = RobotUtils.buildShape("biped-4x2");
    Grid<Boolean> shape2 = RobotUtils.buildShape("biped-3x2");
    shape2.set(0, 0, true);
    Grid<Boolean> shape3 = RobotUtils.buildShape("biped-4x4");
    Grid<Boolean> shape4 = RobotUtils.buildShape("biped-4x3");
    shape4.set(1, 1, false);
    shape4.set(2, 1, false);
    Grid<Boolean> shape5 = RobotUtils.buildShape("biped-6x4");
    shape5.set(1, 0, true);
    shape5.set(1, 1, true);

    List<Pair<Grid<Boolean>, String>> shapes = new ArrayList<>();
    shapes.add(Pair.of(shape0, "biped-0"));
    shapes.add(Pair.of(shape1, "biped-1"));
    shapes.add(Pair.of(shape2, "biped-2"));
    shapes.add(Pair.of(shape3, "biped-3"));
    shapes.add(Pair.of(shape4, "biped-4"));
    shapes.add(Pair.of(shape5, "biped-5"));
    return shapes;
  }

  private static List<Pair<Grid<Boolean>, String>> buildShapes(String shape) {
    return shape.startsWith("biped") ? buildBipeds() : buildWorms();
  }

}