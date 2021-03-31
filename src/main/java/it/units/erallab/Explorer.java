package it.units.erallab;

import it.units.erallab.hmsrobots.core.controllers.CentralizedSensing;
import it.units.erallab.hmsrobots.core.controllers.MultiLayerPerceptron;
import it.units.erallab.hmsrobots.core.objects.Robot;
import it.units.erallab.hmsrobots.util.SerializationUtils;
import it.units.malelab.jgea.Worker;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVPrinter;
import org.apache.commons.csv.CSVRecord;

import java.io.FileReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.Reader;
import java.util.List;
import java.util.logging.Logger;

public class Explorer extends Worker {

  private List<CSVRecord> records;
  private CSVPrinter printer;
  private Logger logger;

  public Explorer(String[] args) {
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
    String inputFileName = "worm-2-2.txt";
    readRecordsFromFile(inputFileName);
    String serializedRobotColumnName = "best→solution→serialized";
    SerializationUtils.Mode mode = SerializationUtils.Mode.GZIPPED_JSON;
    // create printer
    for (CSVRecord record : records) {
      // read robot and record
      Robot<?> robot = SerializationUtils.deserialize(record.get(serializedRobotColumnName), Robot.class, mode);
      inspectRobot(robot);
      //System.out.println(robot.toString());
      break;
    }

  }

  public static void main(String[] args) {
    new Explorer(args);
  }

  private static void inspectRobot(Robot<?> robot) {
    CentralizedSensing centralizedSensing = (CentralizedSensing) robot.getController();
    //RealFunction realFunction = (MultiLayerPerceptron) centralizedSensing.getFunction();
    MultiLayerPerceptron multiLayerPerceptron =(MultiLayerPerceptron) centralizedSensing.getFunction();
    int[] neurons = multiLayerPerceptron.getNeurons();
    System.out.println(multiLayerPerceptron.flat(multiLayerPerceptron.getWeights(), multiLayerPerceptron.getNeurons()).length);
    //System.out.printf("N layers: %d\n", neurons.length);
    //System.out.printf("Input size %d\n", neurons[0]);
    //for(int i=1; i<neurons.length-1; i++){
    //  System.out.printf("Layer %d size %d\n", i, neurons[neurons.length-1]);
    //}
    //System.out.printf("Output size %d\n", neurons[neurons.length-1]);
  }


}
