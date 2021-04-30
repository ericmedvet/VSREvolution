package it.units.erallab;

import it.units.erallab.hmsrobots.core.objects.Robot;
import it.units.erallab.hmsrobots.tasks.locomotion.Locomotion;
import it.units.erallab.hmsrobots.tasks.locomotion.Outcome;
import it.units.erallab.hmsrobots.util.SerializationUtils;
import it.units.erallab.utils.Utils;
import it.units.malelab.jgea.Worker;
import it.units.malelab.jgea.core.listener.NamedFunction;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVPrinter;
import org.apache.commons.csv.CSVRecord;
import org.dyn4j.dynamics.Settings;

import java.io.*;
import java.util.ArrayList;
import java.util.List;
import java.util.logging.Logger;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static it.units.malelab.jgea.core.util.Args.d;
import static it.units.malelab.jgea.core.util.Args.l;

public class RobotsValidator extends Worker {

  private List<CSVRecord> records;
  private CSVPrinter printer;

  public RobotsValidator(String[] args) {
    super(args);
  }

  public List<String> readRecordsFromFile(String inputFileName) {
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
    String inputFileName = a("inputFile", "last.txt");
    String serializedRobotColumnName = a("serializedRobotColumn", "best→solution→serialized");
    String outputFileName = a("outputFile", "validation-redone.txt");
    double episodeTime = d(a("episodeTime", "10"));
    double episodeTransientTime = d(a("episodeTransientTime", "1"));
    List<String> terrainNames = l(a("validationTerrain", "flat," +
        "hilly-1-10-0,hilly-1-10-1,hilly-1-10-2,hilly-1-10-3,hilly-1-10-4,hilly-1-10-5," +
        "steppy-1-10-0,steppy-1-10-1,steppy-1-10-2,steppy-1-10-3,steppy-1-10-4,steppy-1-10-5," +
        "downhill-10,downhill-20,uphill-10,uphill-20"));
    SerializationUtils.Mode mode = SerializationUtils.Mode.GZIPPED_JSON;
    // create printer
    try {
      printer = new CSVPrinter(new PrintStream(outputFileName), CSVFormat.DEFAULT.withDelimiter(';'));
    } catch (IOException e) {
      e.printStackTrace();
      return;
    }
    // parse old file and print headers to new file
    List<String> oldHeaders = readRecordsFromFile(inputFileName);
    int serializedRobotColumnIndex = oldHeaders.indexOf(serializedRobotColumnName);
    oldHeaders = oldHeaders.stream().filter(x -> !x.equals(serializedRobotColumnName)).collect(Collectors.toList());
    List<NamedFunction<Outcome, ?>> basicOutcomeFunctions = Utils.basicOutcomeFunctions();
    List<NamedFunction<Outcome, ?>> detailedOutcomeFunctions = Utils.detailedOutcomeFunctions(spectrumMinFreq, spectrumMaxFreq, spectrumSize);
    List<String> basicOutcomeFunctionsNames = basicOutcomeFunctions.stream().map(NamedFunction::getName).collect(Collectors.toList());
    List<String> detailedOutcomeFunctionsNames = detailedOutcomeFunctions.stream().map(NamedFunction::getName).collect(Collectors.toList());
    List<String> headers = new ArrayList<>(oldHeaders);
    headers.add("validationTerrain");
    headers.addAll(basicOutcomeFunctionsNames);
    headers.addAll(detailedOutcomeFunctionsNames);
    try {
      printer.printRecord(headers);
    } catch (IOException e) {
      e.printStackTrace();
    }
    int validationsCounter = 0;
    for (CSVRecord record : records) {
      // read robot and record
      Robot<?> robot = SerializationUtils.deserialize(record.get(serializedRobotColumnIndex), Robot.class, mode);
      robot.reset();
      List<String> oldRecord = IntStream.range(0, record.size()).filter(index -> index != serializedRobotColumnIndex).mapToObj(
          record::get).collect(Collectors.toList());
      // validate robot on all terrains
      List<List<Object>> rows = terrainNames.stream().parallel()
          .map(terrainName -> {
            List<Object> cells = new ArrayList<>(oldRecord);
            cells.add(terrainName);
            Locomotion locomotion = new Locomotion(episodeTime, Locomotion.createTerrain(terrainName), new Settings());
            Outcome outcome = locomotion.apply(SerializationUtils.clone(robot));
            cells.addAll(basicOutcomeFunctions.stream().map(f -> f.apply(outcome.subOutcome(episodeTransientTime, episodeTime))).collect(Collectors.toList()));
            cells.addAll(detailedOutcomeFunctions.stream().map(f -> f.apply(outcome.subOutcome(episodeTransientTime, episodeTime))).collect(Collectors.toList()));
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
    new RobotsValidator(args);
  }

}
