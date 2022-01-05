package it.units.erallab.utils;

import it.units.erallab.hmsrobots.core.controllers.CentralizedSensing;
import it.units.erallab.hmsrobots.core.controllers.snn.learning.*;
import it.units.erallab.hmsrobots.core.controllers.snndiscr.QuantizedLearningMultilayerSpikingNetwork;
import it.units.erallab.hmsrobots.core.controllers.snndiscr.QuantizedMultilayerSpikingNetwork;
import it.units.erallab.hmsrobots.core.controllers.snndiscr.QuantizedMultilayerSpikingNetworkWithConverters;
import it.units.erallab.hmsrobots.core.objects.Robot;
import it.units.erallab.hmsrobots.util.SerializationUtils;
import it.units.malelab.jgea.Worker;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVPrinter;
import org.apache.commons.csv.CSVRecord;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.logging.Logger;
import java.util.stream.Collectors;

public class RulesParamsExtractor extends Worker {

  private List<CSVRecord> records;

  public RulesParamsExtractor(String[] args) {
    super(args);
  }

  private void readRecordsFromFile(String inputFileName) {
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
      reader.close();
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
    }
  }

  @Override
  public void run() {
    Logger logger = Logger.getAnonymousLogger();
    System.setProperty("java.util.logging.SimpleFormatter.format", "[%1$tF %1$tT] [%4$-6s] %5$s %n");
    // params
    String inputFileName = "C:\\Users\\giorg\\Documents\\UNITS\\LAUREA_MAGISTRALE\\TESI\\SNN\\Outcomes\\learning_weights_rule_no_learning\\longer\\best.txt";
    String serializedRobotColumnName = a("serializedRobotColumn", "best→solution→serialized");
    String outputFileName = "C:\\Users\\giorg\\Documents\\UNITS\\LAUREA_MAGISTRALE\\TESI\\SNN\\Outcomes\\learning_weights_rule_no_learning\\longer\\best+rules.txt";

    SerializationUtils.Mode mode = SerializationUtils.Mode.GZIPPED_JSON;
    List<String> headersToKeep = List.of("seed", "mapper", "iterations", "births", "terrain", "shape", "best→fitness→fitness", "fitness.evaluations", "evolver");
    // create printer
    CSVPrinter printer;
    try {
      printer = new CSVPrinter(new PrintStream(outputFileName), CSVFormat.DEFAULT.withDelimiter(';'));
    } catch (IOException e) {
      e.printStackTrace();
      return;
    }
    List<STDPLearningRule> existingRules = List.of(new DegenerateLearningRule(),
        new AsymmetricHebbianLearningRule(),
        new SymmetricHebbianLearningRule(),
        new AsymmetricAntiHebbianLearningRule(),
        new SymmetricAntiHebbianLearningRule());
    List<Class<?>> existingRulesClasses = existingRules.stream().map(STDPLearningRule::getClass).collect(Collectors.toList());
    List<String> existingRulesNames = existingRulesClasses.stream()
        .map(Class::getSimpleName)
        .map(s -> s.replaceAll("([a-z])([A-Z]+)", "$1.$2").toLowerCase())
        .map(s -> s.replace(".learning.", ".")).toList();
    // parse old file and print headers to new file
    readRecordsFromFile(inputFileName);
    List<String> headers = new ArrayList<>(headersToKeep);
    headers.addAll(existingRulesNames);


    try {
      printer.printRecord(headers);
    } catch (IOException e) {
      e.printStackTrace();
    }
    int validationsCounter = 0;
    for (CSVRecord record : records) {
      // read robot and record
      Robot robot = SerializationUtils.deserialize(record.get(serializedRobotColumnName), Robot.class, mode);
      CentralizedSensing centralizedSensing = (CentralizedSensing) robot.getController();
      QuantizedMultilayerSpikingNetworkWithConverters<?> qmsnConv = (QuantizedMultilayerSpikingNetworkWithConverters<?>) centralizedSensing.getFunction();
      QuantizedMultilayerSpikingNetwork qmsn = qmsnConv.getMultilayerSpikingNetwork();
      if (!(qmsn instanceof QuantizedLearningMultilayerSpikingNetwork qlmsn)) {
        continue;
      }
      STDPLearningRule[][][] rules = qlmsn.getLearningRules();
      List<String> oldRecord = headersToKeep.stream().map(record::get).toList();
      int[] rulesCounter = new int[existingRulesClasses.size()];
      for (STDPLearningRule[][] rule : rules) {
        for (STDPLearningRule[] stdpLearningRules : rule) {
          for (STDPLearningRule stdpLearningRule : stdpLearningRules) {
            for (int ruleIndex = 0; ruleIndex < rulesCounter.length; ruleIndex++) {
              Class<?> ruleClass = existingRulesClasses.get(ruleIndex);
              if (stdpLearningRule.getClass().equals(ruleClass)) {
                rulesCounter[ruleIndex] += 1;
                break;
              }
            }
          }
        }
      }

      List<String> printableRecord = new ArrayList<>(oldRecord);
      printableRecord.addAll(Arrays.stream(rulesCounter).mapToObj(i -> i + "").toList());
      try {
        printer.printRecord(printableRecord);
      } catch (IOException e) {
        e.printStackTrace();
      }

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
    new RulesParamsExtractor(args);
  }

}
