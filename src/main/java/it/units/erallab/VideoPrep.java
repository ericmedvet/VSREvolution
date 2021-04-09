package it.units.erallab;

import it.units.erallab.hmsrobots.core.controllers.CentralizedSensing;
import it.units.erallab.hmsrobots.core.controllers.MultiLayerPerceptron;
import it.units.erallab.hmsrobots.core.controllers.PruningMultiLayerPerceptron;
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

public class VideoPrep extends Worker {

  private List<CSVRecord> records;
  private CSVPrinter printer;
  private Logger logger;

  public VideoPrep(String[] args) {
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
    for (int i = 0; i < 3; i++) {
      logger = Logger.getAnonymousLogger();
      System.setProperty("java.util.logging.SimpleFormatter.format", "[%1$tF %1$tT] [%4$-6s] %5$s %n");
      // params
      String newMapper = "pMLP-tanh-70-0.0-weight";
      String newDesc = "No pruning";
      String weightDesc = "Weight (ρ=";
      String absSigMeanDesc = "Abs.sig.mean. (ρ=";
      String descEnd = ") pruning at t=20s";
      String inputFileName = "last-for-videos-" + i + ".txt";
      String serializedRobotColumnName = "best.solution.serialized";
      String outputFileName = "for-video-" + i + ".txt";
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
      List<String> headers = List.of("x", "y", "best.serialized.robot", "mapper", "desc");
      try {
        printer.printRecord(headers);
      } catch (IOException e) {
        e.printStackTrace();
      }

      for (CSVRecord record : records) {
        // read robot and record
        try {
          Robot<?> oldRobot = SerializationUtils.deserialize(record.get(serializedRobotColumnName), Robot.class, mode);
          String pruningMlp = "fixedCentralized<pMLP-\\d-\\d-(?<actFun>(sin|tanh|sigmoid|relu))-(?<nOfCalls>\\d+)-(?<pruningRate>0(\\.\\d+)?)-(?<criterion>(weight|abs_signal_mean|random))";
          Map<String, String> params = params(pruningMlp, record.get("mapper"));
          String oldDesc = record.get("mapper").endsWith("weight") ? weightDesc : absSigMeanDesc;
          oldDesc+=params.get("pruningRate");
          oldDesc+=descEnd;
          printer.printRecord(0, 0, SerializationUtils.serialize(oldRobot, mode), record.get("mapper"), oldDesc);
          Robot<?> robot = changeRobot(oldRobot, newMapper);
          printer.printRecord(0, 1, SerializationUtils.serialize(robot, mode), newMapper, newDesc);
        } catch (Exception e) {
          e.printStackTrace();
        }
      }
      // close printer
      try {
        printer.flush();
        printer.close();
      } catch (IOException e) {
        e.printStackTrace();
      }
    }
  }

  public static void main(String[] args) {
    new VideoPrep(args);
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
