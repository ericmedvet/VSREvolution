package it.units.erallab;

import it.units.erallab.hmsrobots.core.controllers.CentralizedSensing;
import it.units.erallab.hmsrobots.core.controllers.snndiscr.QuantizedMultilayerSpikingNetworkWithConverters;
import it.units.erallab.hmsrobots.core.objects.Robot;
import it.units.erallab.hmsrobots.tasks.locomotion.Locomotion;
import it.units.erallab.hmsrobots.util.SerializationUtils;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVPrinter;
import org.apache.commons.csv.CSVRecord;
import org.dyn4j.dynamics.Settings;

import java.io.FileReader;
import java.io.IOException;
import java.io.PrintStream;
import java.util.Arrays;
import java.util.Map;

public class SNNWeightsTracker {

  public static void main(String[] args) throws IOException {
    String inputFileName = "last.txt";
    double episodeTime = 10d;
    String terrainName = "flat";
    String outputFileName = "weights.txt";

    CSVParser csvParser = CSVFormat.DEFAULT.withDelimiter(';').withFirstRecordAsHeader().parse(new FileReader(inputFileName));
    CSVRecord record = csvParser.getRecords().get(0);
    System.out.println(record.get("best→fitness→fitness"));
    Robot robot = SerializationUtils.deserialize(record.get("best→solution→serialized"), Robot.class, SerializationUtils.Mode.GZIPPED_JSON);
    CentralizedSensing centralizedSensing = (CentralizedSensing) robot.getController();
    QuantizedMultilayerSpikingNetworkWithConverters<?> quantizedMultilayerSpikingNetworkWithConverters = (QuantizedMultilayerSpikingNetworkWithConverters<?>) centralizedSensing.getFunction();
    quantizedMultilayerSpikingNetworkWithConverters.setWeightsTracker(true);
    new Locomotion(episodeTime, Locomotion.createTerrain(terrainName), new Settings()).apply(robot);
    //Map<Double, Map<Double, Double>> weightsDistributionInTime = quantizedMultilayerSpikingNetworkWithConverters.getWeightsDistributionInTime();
    Map<Double, double[]> weightsInTime = quantizedMultilayerSpikingNetworkWithConverters.getWeightsInTime();
    CSVPrinter printer = new CSVPrinter(new PrintStream(outputFileName), CSVFormat.DEFAULT.withDelimiter(';'));
    printer.printRecord("time", "weight");
    weightsInTime.forEach((time, weights) -> Arrays.stream(weights).forEach(weight -> {
      try {
        printer.printRecord(time, weight);
      } catch (IOException e) {
        e.printStackTrace();
      }
    }));
    printer.close();
  }

}
