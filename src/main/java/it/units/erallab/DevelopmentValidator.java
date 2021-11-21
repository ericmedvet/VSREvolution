package it.units.erallab;

import it.units.erallab.hmsrobots.core.controllers.Controller;
import it.units.erallab.hmsrobots.core.objects.Robot;
import it.units.erallab.hmsrobots.core.objects.SensingVoxel;
import it.units.erallab.hmsrobots.tasks.locomotion.Locomotion;
import it.units.erallab.hmsrobots.util.Grid;
import it.units.erallab.hmsrobots.util.SerializationUtils;
import it.units.erallab.hmsrobots.util.Utils;
import it.units.malelab.jgea.Worker;
import it.units.malelab.jgea.core.util.Pair;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVPrinter;
import org.apache.commons.csv.CSVRecord;
import org.dyn4j.dynamics.Settings;

import java.io.FileReader;
import java.io.IOException;
import java.io.PrintStream;
import java.io.Reader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static it.units.malelab.jgea.core.util.Args.i;

public class DevelopmentValidator extends Worker {

  public static void main(String[] args) {
    new DevelopmentValidator(args);
  }

  public DevelopmentValidator(String[] args) {
    super(args);
  }

  @SuppressWarnings("unchecked")
  public void run() {

    int validationEpisodeTime = i(a("validationEpisodeTime", "30"));
    String validationTerrain = a("validationTerrain", "flat");
    int nRemovals = i(a("nRemovals", "5"));
    int[] removals = IntStream.range(1, nRemovals + 1).toArray();

    String path = a("path", "");
    String inputFileName = a("inputFile", path + "last.txt");

    String shapesFile = a("shapesFile", path + "shapes.txt");
    String descriptorsFile = a("descriptorsFile", path + "descriptors.txt");
    String voxelRemovalFile = a("voxelRemovalFile", path + "removal-speeds.txt");
    String speedsFile = a("speedsFile", path + "speeds.txt");

    List<String> colNames = List.of("seed", "devo.function", "development.schedule");
    String robotsColumn = "best.fitness→devo.robots";
    String velocitiesColumn = "best.fitness→outcomes.speeds";

    CSVPrinter shapesPrinter;
    CSVPrinter descriptorsPrinter;
    CSVPrinter voxelRemovalPrinter;
    CSVPrinter speedsPrinter;
    try {
      shapesPrinter = new CSVPrinter(new PrintStream(shapesFile), CSVFormat.DEFAULT.withDelimiter(';'));
      descriptorsPrinter = new CSVPrinter(new PrintStream(descriptorsFile), CSVFormat.DEFAULT.withDelimiter(';'));
      voxelRemovalPrinter = new CSVPrinter(new PrintStream(voxelRemovalFile), CSVFormat.DEFAULT.withDelimiter(';'));
      speedsPrinter = new CSVPrinter(new PrintStream(speedsFile), CSVFormat.DEFAULT.withDelimiter(';'));
    } catch (IOException e) {
      e.printStackTrace();
      return;
    }

    List<CSVRecord> records = readRecordsFromFile(inputFileName);
    if (records == null) {
      System.exit(-1);
    }

    List<String> shapesHeaders = new ArrayList<>(colNames);
    List<String> descriptorsHeaders = new ArrayList<>(colNames);
    List<String> voxelRemovalHeaders = new ArrayList<>(colNames);
    List<String> speedsHeaders = new ArrayList<>(colNames);

    shapesHeaders.addAll(List.of("stage", "x", "y"));
    descriptorsHeaders.addAll(List.of("stage", "compactness", "elongation"));
    voxelRemovalHeaders.addAll(List.of("n", "speed"));
    speedsHeaders.addAll(List.of("stage", "speed"));

    try {
      shapesPrinter.printRecord(shapesHeaders);
      descriptorsPrinter.printRecord(descriptorsHeaders);
      voxelRemovalPrinter.printRecord(voxelRemovalHeaders);
      speedsPrinter.printRecord(speedsHeaders);
    } catch (IOException e) {
      e.printStackTrace();
    }

    int counter = 0;
    for (CSVRecord record : records) {
      List<String> values = colNames.stream().map(record::get).collect(Collectors.toList());
      String[] serializedRobots = record.get(robotsColumn).split(",");
      List<Robot<SensingVoxel>> robots = new ArrayList<>();
      for (String serializedRobot : serializedRobots) {
        robots.add(SerializationUtils.deserialize(serializedRobot, Robot.class, SerializationUtils.Mode.GZIPPED_JSON));
      }
      List<Grid<?>> bodies = robots.stream().map(Robot::getVoxels).collect(Collectors.toList());

      // shapes
      List<List<Number>> shapeRecords = extractShapeDevelopmentStages(bodies);
      for (List<Number> shapeAddition : shapeRecords) {
        printRecord(values, shapeAddition, shapesPrinter);
      }

      // descriptors
      for (int i = 0; i < bodies.size(); i++) {
        List<Number> descriptorsAdditions = new ArrayList<>(List.of(i));
        descriptorsAdditions.addAll(computeDescriptors(bodies.get(i)));
        printRecord(values, descriptorsAdditions, descriptorsPrinter);
      }

      // voxel removal speed
      Robot<SensingVoxel> lastRobot = robots.get(robots.size() - 1);
      for (int removal : removals) {
        List<Double> velocities = getVelocitiesAfterRemoval(lastRobot, removal, validationEpisodeTime, validationTerrain);
        for (double v : velocities) {
          List<Number> removalAddition = List.of(removal, v);
          printRecord(values, removalAddition, voxelRemovalPrinter);
        }
      }

      // speeds
      String[] speeds = record.get(velocitiesColumn).split(",");
      for (int i = 0; i < speeds.length; i++) {
        List<Number> speedsAddition = List.of(i, Double.parseDouble(speeds[i]));
        printRecord(values, speedsAddition, speedsPrinter);
      }

      // print progress
      counter++;
      System.out.printf("%d/%d%n", counter, records.size());
    }

  }

  private static List<CSVRecord> readRecordsFromFile(String inputFileName) {
    //read data
    try (Reader reader = new FileReader(inputFileName)) {
      return CSVFormat.DEFAULT.withDelimiter(';').withFirstRecordAsHeader().parse(reader).getRecords();
    } catch (IOException e) {
      e.printStackTrace();
      return null;
    }
  }

  private static void printRecord(List<String> values, List<Number> additions, CSVPrinter printer) {
    List<String> descriptorsRecord = new ArrayList<>(values);
    descriptorsRecord.addAll(additions.stream().map(d -> d + "").collect(Collectors.toList()));
    try {
      printer.printRecord(descriptorsRecord);
    } catch (IOException e) {
      e.printStackTrace();
    }
  }

  private static List<Double> getVelocitiesAfterRemoval(Robot<SensingVoxel> robot, int nVoxels, int validationEpisodeTime, String validationTerrain) {
    Controller<SensingVoxel> controller = robot.getController();
    Grid<? extends SensingVoxel> body = robot.getVoxels();
    List<Grid<? extends SensingVoxel>> newBodies = getVoxelRemovalBodies(body, nVoxels);
    List<Robot<? extends SensingVoxel>> robots = newBodies.stream().parallel().map(
        b -> new Robot<>(SerializationUtils.clone(controller), b)
    ).collect(Collectors.toList());
    return robots.stream().parallel()
        .map(r -> new Locomotion(validationEpisodeTime, Locomotion.createTerrain(validationTerrain), new Settings())
            .apply(r).getVelocity())
        .collect(Collectors.toList());
  }

  private static List<Grid<? extends SensingVoxel>> getVoxelRemovalBodies(Grid<? extends SensingVoxel> body, int nVoxels) {
    List<Pair<Integer, Integer>> voxelsPositions = body.stream()
        .filter(Objects::nonNull).filter(v -> v.getValue() != null)
        .map(c -> Pair.of(c.getX(), c.getY())).collect(Collectors.toList());
    int targetNumberOfVoxels = voxelsPositions.size() - nVoxels;
    List<List<Pair<Integer, Integer>>> removalCandidates = getRemovalPositions(voxelsPositions, nVoxels);
    List<Grid<? extends SensingVoxel>> newBodies = new ArrayList<>();
    removalCandidates.stream().parallel().forEach(l -> {
      Grid<? extends SensingVoxel> newBody = SerializationUtils.clone(body);
      for (Pair<Integer, Integer> p : l) {
        newBody.set(p.first(), p.second(), null);
      }
      if (Utils.gridLargestConnected(newBody, Objects::nonNull).count(Objects::nonNull) == targetNumberOfVoxels) {
        newBodies.add(newBody);
      }
    });
    return newBodies;
  }

  private static List<Double> computeDescriptors(Grid<?> body) {
    Grid<Boolean> booleanBody = Grid.create(body, Objects::nonNull);
    double shapeCompactness = Utils.shapeCompactness(booleanBody);
    double shapeElongation = Utils.shapeElongation(booleanBody, 4);
    return List.of(shapeCompactness, shapeElongation);
  }

  private static List<List<Pair<Integer, Integer>>> getRemovalPositions(List<Pair<Integer, Integer>> coordinates, int nRemovals) {
    List<int[]> combinations = generateCombinations(coordinates.size(), nRemovals);
    return combinations.stream().map(
        list -> Arrays.stream(list).mapToObj(
            coordinates::get
        ).collect(Collectors.toList())
    ).collect(Collectors.toList());
  }

  // https://www.baeldung.com/java-combinations-algorithm
  private static List<int[]> generateCombinations(int n, int r) {
    List<int[]> combinations = new ArrayList<>();
    int[] combination = new int[r];
    // initialize with lowest lexicographic combination
    for (int i = 0; i < r; i++) {
      combination[i] = i;
    }
    while (combination[r - 1] < n) {
      combinations.add(combination.clone());
      // generate next combination in lexicographic order
      int t = r - 1;
      while (t != 0 && combination[t] == n - r + t) {
        t--;
      }
      combination[t]++;
      for (int i = t + 1; i < r; i++) {
        combination[i] = combination[i - 1] + 1;
      }
    }
    return combinations;
  }

  private static List<List<Number>> extractShapeDevelopmentStages(List<Grid<?>> grids) {
    int width = grids.get(0).getW();
    int height = grids.get(0).getH();
    List<List<Number>> lists = new ArrayList<>();
    for (int x = 0; x < width; x++) {
      for (int y = 0; y < height; y++) {
        for (int stage = 0; stage < grids.size(); stage++) {
          if (grids.get(stage).get(x, y) != null) {
            List<Number> list = new ArrayList<>();
            list.add(stage);
            list.add(x);
            list.add(y);
            lists.add(list);
            break;
          }
        }
      }
    }
    return lists;
  }

}
