package it.units.erallab;

import it.units.erallab.hmsrobots.core.controllers.CentralizedSensing;
import it.units.erallab.hmsrobots.core.controllers.MultiLayerPerceptron;
import it.units.erallab.hmsrobots.core.controllers.PruningMultiLayerPerceptron;
import it.units.erallab.hmsrobots.core.controllers.RealFunction;
import it.units.erallab.hmsrobots.core.objects.Robot;
import it.units.erallab.hmsrobots.core.objects.SensingVoxel;
import it.units.erallab.hmsrobots.tasks.locomotion.Locomotion;
import it.units.erallab.hmsrobots.tasks.locomotion.Outcome;
import it.units.erallab.hmsrobots.util.Grid;
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
import java.util.Map;
import java.util.logging.Logger;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static it.units.erallab.hmsrobots.util.Utils.params;

public class RobotsValidator extends Worker {

  private List<CSVRecord> records;
  private CSVPrinter printer;
  private Logger logger;

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
    logger = Logger.getAnonymousLogger();
    System.setProperty("java.util.logging.SimpleFormatter.format", "[%1$tF %1$tT] [%4$-6s] %5$s %n");
    // params
    String[] newMappers = {"pMLP-tanh-1200-0.125-abs_signal_mean",
        "pMLP-tanh-1200-0.125-weight",
        "pMLP-tanh-1200-0.125-random",
        "pMLP-tanh-1200-0.25-abs_signal_mean",
        "pMLP-tanh-1200-0.25-weight",
        "pMLP-tanh-1200-0.25-random",
        "pMLP-tanh-1200-0.5-abs_signal_mean",
        "pMLP-tanh-1200-0.5-weight",
        "pMLP-tanh-1200-0.5-random",
        "pMLP-tanh-1200-0.75-abs_signal_mean",
        "pMLP-tanh-1200-0.75-weight",
        "pMLP-tanh-1200-0.75-random"
    };
    String unprunedMapperEnding = "-2-tanh-1200-0.0-weight";
    String inputFileName = "final.txt";
    String serializedRobotColumnName = "best→solution→serialized";
    String mapperColumnName = "mapper";
    String outputFileName = "validation-unpruned-to-pruned.txt";
    double endTime = 60;
    double transientTime = 20;
    List<String> terrainNames = List.of("flat", "hilly-1-10-0", "hilly-1-10-1", "hilly-1-10-2", "hilly-1-10-3", "hilly-1-10-4", "hilly-1-10-5", "steppy-1-10-0", "steppy-1-10-1", "steppy-1-10-2", "steppy-1-10-3", "steppy-1-10-4", "steppy-1-10-5", "downhill-10", "downhill-20", "uphill-10", "uphill-20");
    SerializationUtils.Mode mode = SerializationUtils.Mode.GZIPPED_JSON;
    // create printer
    try {
      printer = new CSVPrinter(new PrintStream(new File(outputFileName)), CSVFormat.DEFAULT.withDelimiter(';'));
    } catch (IOException e) {
      logger.severe("Could not create printer");
      return;
    }
    // parse old file and print headers to new file
    List<String> oldHeaders = readRecordsFromFile(inputFileName);
    int serializedRobotColumnIndex = oldHeaders.indexOf(serializedRobotColumnName);
    oldHeaders = oldHeaders.stream().filter(x -> !x.equals(serializedRobotColumnName)).collect(Collectors.toList());
    List<NamedFunction<Outcome, ?>> basicOutcomeFunctions = Utils.basicOutcomeFunctions();
    List<String> newHeaders = basicOutcomeFunctions.stream().map(NamedFunction::getName).collect(Collectors.toList());
    List<String> headers = new ArrayList<>(oldHeaders);
    headers.add("validationTerrain");
    headers.add("newMapper");
    headers.addAll(newHeaders);
    try {
      printer.printRecord(headers);
    } catch (IOException e) {
      e.printStackTrace();
    }

    int validationsCounter = 0;
    logger.info(String.format("%3d/%3d", validationsCounter, records.size()));
    List<List<Object>> rows = new ArrayList<>();
    for (CSVRecord record : records) {
      if (!record.get(mapperColumnName).endsWith(unprunedMapperEnding)) {
        logger.info(String.format("%3d/%3d -> skipped", ++validationsCounter, records.size()));
        continue;
      }
      // read robot and record
      Robot<?> oldRobot = SerializationUtils.deserialize(record.get(serializedRobotColumnIndex), Robot.class, mode);
      List<String> oldRecord = IntStream.range(0, record.size()).filter(index -> index != serializedRobotColumnIndex).mapToObj(
          record::get).collect(Collectors.toList());
      for (String newMapper : newMappers) {
        Robot<?> robot = changeRobot(oldRobot, newMapper);
        // validate robot on all terrains
        rows.addAll(terrainNames.stream().parallel()
            .map(terrainName -> {
              List<Object> cells = new ArrayList<>(oldRecord);
              cells.add(terrainName);
              cells.add(newMapper);
              Locomotion locomotion = new Locomotion(endTime, Locomotion.createTerrain(terrainName), new Settings());
              Outcome outcome = locomotion.apply(SerializationUtils.clone(robot));
              cells.addAll(basicOutcomeFunctions.stream().map(f -> f.apply(outcome.subOutcome(transientTime, endTime))).collect(Collectors.toList()));
              return cells;
            })
            .collect(Collectors.toList()));
      }
      logger.info(String.format("%3d/%3d", validationsCounter++, records.size()));
    }
    rows.forEach(row -> {
      try {
        printer.printRecord(row);
      } catch (IOException e) {
        e.printStackTrace();
      }
    });
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

  // change the controller of the robot (from pruned to unpruned)
  private static Robot<?> changeRobot(Robot<?> robot, String newMapper) {
    String pruningMlp = "pMLP-(?<actFun>(sin|tanh|sigmoid|relu))-(?<nOfCalls>\\d+)-(?<pruningRate>0(\\.\\d+)?)-(?<criterion>(weight|abs_signal_mean|random))";
    Map<String, String> params = params(pruningMlp, newMapper);
    CentralizedSensing centralizedSensing = (CentralizedSensing) robot.getController();
    MultiLayerPerceptron multiLayerPerceptron = (MultiLayerPerceptron) centralizedSensing.getFunction();
    double[][][] weights = multiLayerPerceptron.getWeights();
    int[] neurons = multiLayerPerceptron.getNeurons();
    return new Robot<>(
        new CentralizedSensing(
            (Grid<SensingVoxel>) robot.getVoxels(),
            new PruningMultiLayerPerceptron(
                MultiLayerPerceptron.ActivationFunction.valueOf(params.get("actFun").toUpperCase()),
                weights,
                neurons,
                Long.parseLong(params.get("nOfCalls")),
                PruningMultiLayerPerceptron.Context.NETWORK,
                PruningMultiLayerPerceptron.Criterion.valueOf(params.get("criterion").toUpperCase()),
                Double.parseDouble(params.get("pruningRate")))
        ),
        (Grid<SensingVoxel>) robot.getVoxels()
    );
  }


}
