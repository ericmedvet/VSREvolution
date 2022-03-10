package it.units.erallab;

import com.google.common.collect.Sets;
import it.units.erallab.hmsrobots.core.objects.Robot;
import it.units.erallab.hmsrobots.core.objects.Voxel;
import it.units.erallab.hmsrobots.tasks.locomotion.Locomotion;
import it.units.erallab.hmsrobots.tasks.locomotion.Outcome;
import it.units.erallab.hmsrobots.util.DoubleRange;
import it.units.erallab.hmsrobots.util.Grid;
import it.units.erallab.hmsrobots.util.SerializationUtils;
import it.units.erallab.locomotion.NamedFunctions;
import it.units.malelab.jgea.Worker;
import it.units.malelab.jgea.core.listener.NamedFunction;
import it.units.malelab.jgea.core.util.Pair;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVPrinter;
import org.apache.commons.csv.CSVRecord;
import org.dyn4j.dynamics.Settings;

import java.io.FileReader;
import java.io.IOException;
import java.io.PrintStream;
import java.io.Reader;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static it.units.malelab.jgea.core.util.Args.d;

public class RobotsPhysicalValidationHetero extends Worker {

  public static void main(String[] args) {
    new RobotsPhysicalValidationHetero(args);
  }

  public RobotsPhysicalValidationHetero(String[] args) {
    super(args);
  }

  public void run() {
    int spectrumSize = 10;
    double spectrumMinFreq = 0d;
    double spectrumMaxFreq = 5d;

    String inputFileName = a("inputFile", "D:\\Research\\Physical_parameters\\Active\\last-full.txt");
    String outputFileName = a("outputFile", "D:\\Research\\Physical_parameters\\Active\\validation-phys-homo.txt");

    String robotsColumn = a("robotColumn", "best→solution→serialized");
    double episodeTime = d(a("episodeTime", "10"));
    double episodeTransientTime = d(a("episodeTransientTime", "1"));

    List<String> headersToKeep = List.of("iterations", "births", "fitness.evaluations", "elapsed.seconds",
        "experiment.name", "seed", "terrain", "shape", "sensor.config", "mapper", "transformation", "evolver",
        "spring.f", "spring.d", "friction", "delta.active");

    CSVPrinter printer;
    try {
      printer = new CSVPrinter(new PrintStream(outputFileName), CSVFormat.DEFAULT.withDelimiter(';'));
    } catch (IOException e) {
      e.printStackTrace();
      return;
    }

    Pair<List<String>, List<CSVRecord>> parsedCsv = readRecordsFromFile(inputFileName);
    List<NamedFunction<Outcome, ?>> basicOutcomeFunctions = NamedFunctions.basicOutcomeFunctions();
    List<NamedFunction<Outcome, ?>> detailedOutcomeFunctions = NamedFunctions.detailedOutcomeFunctions(spectrumMinFreq, spectrumMaxFreq, spectrumSize);
    List<String> basicOutcomeFunctionsNames = basicOutcomeFunctions.stream().map(NamedFunction::getName).toList();
    List<String> detailedOutcomeFunctionsNames = detailedOutcomeFunctions.stream().map(NamedFunction::getName).toList();
    List<String> headers = new ArrayList<>(headersToKeep);
    headers.add("validation.terrain");
    headers.addAll(basicOutcomeFunctionsNames.stream().map(h -> "validation." + h).toList());
    headers.addAll(detailedOutcomeFunctionsNames.stream().map(h -> "validation." + h).toList());
    headers.addAll(List.of("validation.delta.active", "validation.spring.d", "validation.spring.f", "validation.friction"));

    Pair<String, Pair<DoubleRange, Integer>> activeDelta = Pair.of("activeDelta", Pair.of(DoubleRange.of(0.1, 0.3), 8));
    Pair<String, Pair<DoubleRange, Integer>> springD = Pair.of("springD", Pair.of(DoubleRange.of(0.1, 0.99), 8));
    Pair<String, Pair<DoubleRange, Integer>> springF = Pair.of("springF", Pair.of(DoubleRange.of(3, 10), 8));
    Pair<String, Pair<DoubleRange, Integer>> friction = Pair.of("friction", Pair.of(DoubleRange.of(0.1, 0.3), 8));

    try {
      printer.printRecord(headers);
    } catch (IOException e) {
      e.printStackTrace();
    }
    List<CSVRecord> records = parsedCsv.second();
    if (records == null) {
      System.exit(-1);
    }
    int i = 0;
    System.out.print(records.size() + "\n");
    for (CSVRecord record : records) {
      System.out.println(i);
      i++;
      List<String> oldRecord = headersToKeep.stream().map(record::get).toList();
      Robot robot = SerializationUtils.deserialize(record.get(robotsColumn), Robot.class, SerializationUtils.Mode.GZIPPED_JSON);
      String expName = record.get("experiment.name");

      List<Map<String, Double>> properties;
      if (expName.contains("active")) {
        properties = buildVoxelPropertiesCombinations(List.of(springD, springF, friction));
      } else if (expName.contains("springf")) {
        properties = buildVoxelPropertiesCombinations(List.of(springD, activeDelta, friction));
      } else if (expName.contains("springd")) {
        properties = buildVoxelPropertiesCombinations(List.of(activeDelta, springF, friction));
      } else {
        properties = buildVoxelPropertiesCombinations(List.of(springD, springF, activeDelta));
      }

      List<Robot> newRobots = properties.stream().map(params -> new Robot(
          SerializationUtils.clone(robot.getController()),
          changeRobotsPhysicalProperties(robot.getVoxels(), params)
      )).toList();

      List<List<Object>> rows = newRobots.stream().parallel()
          .map(newRobot -> {
            List<Object> cells = new ArrayList<>(oldRecord);
            cells.add("flat");
            Locomotion locomotion = new Locomotion(episodeTime, Locomotion.createTerrain("flat"), new Settings());
            Outcome outcome = locomotion.apply(SerializationUtils.clone(newRobot));
            cells.addAll(basicOutcomeFunctions.stream().map(f -> f.apply(outcome.subOutcome(episodeTransientTime, episodeTime))).toList());
            cells.addAll(detailedOutcomeFunctions.stream().map(f -> f.apply(outcome.subOutcome(episodeTransientTime, episodeTime))).toList());
            Voxel voxel = newRobot.getVoxels().stream().findFirst().get().value();
            cells.add(voxel.getDeltaActive());
            cells.add(voxel.getSpringD());
            cells.add(voxel.getSpringF());
            cells.add(voxel.getFriction());
            return cells;
          }).toList();
      rows.forEach(row -> {
        try {
          printer.printRecord(row);
        } catch (IOException e) {
          e.printStackTrace();
        }
      });

    }

  }

  private static Pair<List<String>, List<CSVRecord>> readRecordsFromFile(String inputFileName) {
    //read data
    try (Reader reader = new FileReader(inputFileName)) {
      CSVParser csvParser = CSVFormat.DEFAULT.withDelimiter(';').withFirstRecordAsHeader().parse(reader);
      List<String> headers = csvParser.getHeaderNames();
      List<CSVRecord> records = csvParser.getRecords();
      return Pair.of(headers, records);
    } catch (IOException e) {
      e.printStackTrace();
      return null;
    }
  }

  private static Grid<Voxel> changeRobotsPhysicalProperties(Grid<Voxel> originalBody, Map<String, Double> voxelProperties) {
    double springF = voxelProperties.getOrDefault("springF", Voxel.SPRING_F);
    double springD = voxelProperties.getOrDefault("springD", Voxel.SPRING_D);
    double friction = voxelProperties.getOrDefault("friction", Voxel.FRICTION);
    DoubleRange areaRatioActiveRange;
    if (voxelProperties.containsKey("activeDelta")) {
      double activeDelta = voxelProperties.get("activeDelta");
      areaRatioActiveRange = DoubleRange.of(1 - activeDelta, 1 + activeDelta);
    } else {
      areaRatioActiveRange = Voxel.AREA_RATIO_ACTIVE_RANGE;
    }
    return Grid.create(originalBody, v -> v == null ? null : new Voxel(
        Voxel.SIDE_LENGTH,
        Voxel.MASS_SIDE_LENGTH_RATIO,
        springF,
        springD,
        Voxel.MASS_LINEAR_DAMPING,
        Voxel.MASS_ANGULAR_DAMPING,
        friction,
        Voxel.RESTITUTION,
        Voxel.MASS,
        Voxel.AREA_RATIO_PASSIVE_RANGE,
        areaRatioActiveRange,
        Voxel.SPRING_SCAFFOLDINGS,
        v.getSensors()
    ));
  }

  @SuppressWarnings("unchecked")
  private List<Map<String, Double>> buildVoxelPropertiesCombinations(
      List<Pair<String, Pair<DoubleRange, Integer>>> properties
  ) {
    if (properties.size() == 0) {
      return List.of(Map.of());
    }
    List<String> propertyNames = properties.stream().map(Pair::first).toList();
    Set<Double>[] sets = properties.stream()
        .map(p -> (p.first().equals("friction")) ? FRICTION_VALUES : sampleRange(p.second().first(), p.second().second())
        )
        .toArray(Set[]::new);
    Set<List<Double>> values = Sets.cartesianProduct(sets);
    List<Map<String, Double>> propertiesCombinations = new ArrayList<>();
    for (List<Double> propertiesValues : values) {
      propertiesCombinations.add(
          IntStream.range(0, propertyNames.size()).boxed().collect(Collectors.toMap(
                  propertyNames::get, propertiesValues::get
              )
          )
      );
    }
    return propertiesCombinations;
  }

  private Set<Double> sampleRange(DoubleRange range, int nSteps) {
    if (nSteps <= 2) {
      return Set.of();
    }
    double step = range.extent() / (nSteps - 1);
    Set<Double> sample = IntStream.range(0, nSteps).mapToObj(i -> range.min() + i * step).collect(Collectors.toSet());
    return sample;
  }

  private static final Set<Double> FRICTION_VALUES = Set.of(0.05, 0.1, 0.25, 0.6, 1.5, 3d, 10d, 25d);

}
