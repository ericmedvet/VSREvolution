package it.units.erallab.utils;

import it.units.erallab.hmsrobots.core.controllers.CentralizedSensing;
import it.units.erallab.hmsrobots.core.objects.Robot;
import it.units.erallab.hmsrobots.tasks.locomotion.Locomotion;
import it.units.erallab.hmsrobots.tasks.locomotion.Outcome;
import it.units.erallab.hmsrobots.util.Parametrized;
import it.units.erallab.hmsrobots.util.SerializationUtils;
import it.units.erallab.locomotion.NamedFunctions;
import it.units.malelab.jgea.Worker;
import it.units.malelab.jgea.core.listener.NamedFunction;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVPrinter;
import org.apache.commons.csv.CSVRecord;
import org.dyn4j.dynamics.Settings;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.logging.Logger;
import java.util.stream.Collectors;

import static it.units.malelab.jgea.core.util.Args.d;

public class WeightPerturbationValidator extends Worker {

  private List<CSVRecord> records;
  private CSVPrinter printer;

  public WeightPerturbationValidator(String[] args) {
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
    int spectrumSize = 10;
    double spectrumMinFreq = 0d;
    double spectrumMaxFreq = 5d;

    Logger logger = Logger.getAnonymousLogger();
    System.setProperty("java.util.logging.SimpleFormatter.format", "[%1$tF %1$tT] [%4$-6s] %5$s %n");
    // params
    String inputFileName = a("inputFile", "C:\\Users\\giorg\\Documents\\UNITS\\LAUREA_MAGISTRALE\\TESI\\SNN\\Outcomes\\binary_weights\\last.txt");
    String serializedRobotColumnName = a("serializedRobotColumn", "best→solution→serialized");
    String outputFileName = a("outputFile", "C:\\Users\\giorg\\Documents\\UNITS\\LAUREA_MAGISTRALE\\TESI\\SNN\\Outcomes\\binary_weights\\validation-diff-weights.txt");
    double episodeTime = d(a("episodeTime", "30"));
    double episodeTransientTime = d(a("episodeTransientTime", "1"));
    double newWeightsMin = d(a("newWeightsMin", "0.1"));
    double newWeightsMax = d(a("newWeightsMax", "4"));
    SerializationUtils.Mode mode = SerializationUtils.Mode.GZIPPED_JSON;
    List<String> headersToKeep = List.of("iterations", "births", "fitness.evaluations", "elapsed.seconds", "experiment.name", "seed", "terrain",
        "shape", "sensor.config", "mapper", "transformation", "evolver", "fitness.metrics");
    List<Double> validationWeights = new ArrayList<>();
    for (double w = newWeightsMin; w <= newWeightsMax; w += 0.1) {
      validationWeights.add(w);
    }
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
    List<String> basicOutcomeFunctionsNames = basicOutcomeFunctions.stream().map(NamedFunction::getName).collect(Collectors.toList());
    List<String> headers = oldHeaders.stream().map(h -> "event." + h).collect(Collectors.toList());
    headers.addAll(List.of("event.weight", "keys.validation.weight"));
    headers.addAll(List.of("keys.validation.transformation", "keys.validation.seed", "keys.validation.terrain"));
    headers.addAll(basicOutcomeFunctionsNames.stream().map(h -> "outcome." + h).collect(Collectors.toList()));

    try {
      printer.printRecord(headers);
    } catch (IOException e) {
      e.printStackTrace();
    }
    int validationsCounter = 0;
    for (CSVRecord record : records) {
      // read robot and record
      Robot robot = SerializationUtils.deserialize(record.get(serializedRobotColumnName), Robot.class, mode);
      robot.reset();
      List<String> oldRecord = oldHeaders.stream().map(record::get).collect(Collectors.toList());
      CentralizedSensing centralizedSensing = (CentralizedSensing) robot.getController();
      Parametrized parametrized = (Parametrized) centralizedSensing.getFunction();
      double[] params = parametrized.getParams();
      double originalWeight = Arrays.stream(params).filter(p -> p != 0).findFirst().getAsDouble();
      double[] mappedParams = Arrays.stream(params).map(p -> p > 0 ? 1 : (p < 0 ? -1 : 0)).toArray();

      // validate robot with all new weights
      List<List<Object>> rows = validationWeights.stream().parallel()
          .map(weight -> {
            double[] newWeights = Arrays.stream(mappedParams).map(d -> d * weight).toArray();
            Robot newRobot = SerializationUtils.clone(robot);
            CentralizedSensing newCentralizedSensing = (CentralizedSensing) newRobot.getController();
            Parametrized newParametrized = (Parametrized) newCentralizedSensing.getFunction();
            newParametrized.setParams(newWeights);
            List<Object> cells = new ArrayList<>(oldRecord);
            cells.addAll(List.of(originalWeight, weight, "identity", 0, "flat"));
            Locomotion locomotion = new Locomotion(episodeTime, Locomotion.createTerrain("flat"), new Settings());
            Outcome outcome = locomotion.apply(newRobot);
            cells.addAll(basicOutcomeFunctions.stream().map(f -> f.apply(outcome.subOutcome(episodeTransientTime, episodeTime))).collect(Collectors.toList()));
            return cells;
          }).collect(Collectors.toList());
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
    new WeightPerturbationValidator(args);
  }

}
