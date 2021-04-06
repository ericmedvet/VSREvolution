package it.units.erallab;

import it.units.erallab.hmsrobots.core.controllers.CentralizedSensing;
import it.units.erallab.hmsrobots.core.controllers.MultiLayerPerceptron;
import it.units.erallab.hmsrobots.core.controllers.PruningMultiLayerPerceptron;
import it.units.erallab.hmsrobots.core.controllers.snn.MultilayerSpikingNetwork;
import it.units.erallab.hmsrobots.core.controllers.snn.SpikingFunction;
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
import java.util.*;
import java.util.logging.Logger;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;

import static it.units.erallab.hmsrobots.util.Utils.params;

public class InspectSnn extends Worker {

  private List<CSVRecord> records;
  private CSVPrinter printer;
  private Logger logger;

  public InspectSnn(String[] args) {
    super(args);
  }

  public static void main(String[] args) {
    new InspectSnn(args);
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
    String inputFileName = "snn-best.(1).txt";
    String serializedRobotColumnName = "best→solution→serialized";
    readRecordsFromFile(inputFileName);
    // parse old file and print headers to new file
    for (CSVRecord record : records) {
      // read robot and record
      Robot<?> robot = SerializationUtils.deserialize(record.get(serializedRobotColumnName), Robot.class, SerializationUtils.Mode.GZIPPED_JSON);
      getWeights(robot);
    }
  }

  // change the controller of the robot (from pruned to unpruned)
  private static void getWeights(Robot<?> robot) {
    CentralizedSensing centralizedSensing = (CentralizedSensing) robot.getController();
    MultiLayerPerceptron mlp = (MultiLayerPerceptron) centralizedSensing.getFunction();
    double[] weights = MultiLayerPerceptron.flat(mlp.getWeights(),mlp.getNeurons());
    String w = Arrays.stream(weights).sorted().mapToObj(Double::toString).collect(Collectors.joining( "," ));
    System.out.println(w);
  }

  /*private static void getWeights(Robot<?> robot) {
    CentralizedSensing centralizedSensing = (CentralizedSensing) robot.getController();
    MultilayerSpikingNetwork multilayerSpikingNetwork = (MultilayerSpikingNetwork) centralizedSensing.getFunction();
    double[] weights = MultilayerSpikingNetwork.flat(multilayerSpikingNetwork.getWeights(),multilayerSpikingNetwork.getNeurons());
    String w = Arrays.stream(weights).sorted().mapToObj(Double::toString).collect(Collectors.joining( "," ));
    System.out.println(w);
  }*/


}
