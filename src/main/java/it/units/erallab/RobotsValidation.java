package it.units.erallab;

import it.units.erallab.hmsrobots.core.objects.Robot;
import it.units.erallab.hmsrobots.tasks.locomotion.Locomotion;
import it.units.erallab.hmsrobots.tasks.locomotion.Outcome;
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

import static it.units.malelab.jgea.core.util.Args.d;
import static it.units.malelab.jgea.core.util.Args.l;

public class RobotsValidation extends Worker {

  public static void main(String[] args) {
    new RobotsValidation(args);
  }

  public RobotsValidation(String[] args) {
    super(args);
  }

  public void run() {
    int spectrumSize = 10;
    double spectrumMinFreq = 0d;
    double spectrumMaxFreq = 5d;

    String inputFileName = a("inputFile", "D:\\Research\\Physical_parameters\\Friction-def\\last-full.txt");
    String outputFileName = a("outputFile", "D:\\Research\\Physical_parameters\\Friction-def\\validation.txt");

    String robotsColumn = a("robotColumn", "best→solution→serialized");
    double episodeTime = d(a("episodeTime", "30"));
    double episodeTransientTime = d(a("episodeTransientTime", "1"));
    List<String> validationTerrainNames = l(a("validationTerrain",
        "hilly-1-10-0,hilly-1-10-1,hilly-1-10-2,hilly-1-10-3,hilly-1-10-4,hilly-1-10-5," +
            "steppy-1-10-0,steppy-1-10-1,steppy-1-10-2,steppy-1-10-3,steppy-1-10-4,steppy-1-10-5," +
            "downhill-10,downhill-20,uphill-10,uphill-20"));

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
      List<List<Object>> rows = validationTerrainNames.stream().parallel()
          .map(terrainName -> {
            List<Object> cells = new ArrayList<>(oldRecord);
            cells.add(terrainName);
            Locomotion locomotion = new Locomotion(episodeTime, Locomotion.createTerrain(terrainName), new Settings());
            Outcome outcome = locomotion.apply(SerializationUtils.clone(robot));
            cells.addAll(basicOutcomeFunctions.stream().map(f -> f.apply(outcome.subOutcome(episodeTransientTime, episodeTime))).toList());
            cells.addAll(detailedOutcomeFunctions.stream().map(f -> f.apply(outcome.subOutcome(episodeTransientTime, episodeTime))).toList());
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

}
