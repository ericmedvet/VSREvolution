package it.units.erallab.utils;

import it.units.erallab.hmsrobots.core.controllers.CentralizedSensing;
import it.units.erallab.hmsrobots.core.objects.Robot;
import it.units.erallab.hmsrobots.util.Parametrized;
import it.units.erallab.hmsrobots.util.SerializationUtils;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVPrinter;
import org.apache.commons.csv.CSVRecord;

import java.io.FileReader;
import java.io.PrintStream;
import java.io.Reader;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

public class NNDistanceCalculator {


  public static void main(String[] args) throws Exception {
    String inputFileName = "C:\\Users\\giorg\\Documents\\UNITS\\LAUREA_MAGISTRALE\\TESI\\SNN\\Outcomes\\binary_weights\\last-mlp-1.txt";
    String outputFileName = "C:\\Users\\giorg\\Documents\\UNITS\\LAUREA_MAGISTRALE\\TESI\\SNN\\Outcomes\\binary_weights\\distances-mlp.txt";
    String serializedRobotColumnName = "best→solution→serialized";
    String mapperColumnName = "mapper";
    String seedColumnName = "seed";

    Reader reader = new FileReader(inputFileName);
    CSVParser csvParser = CSVFormat.DEFAULT.withDelimiter(';').withFirstRecordAsHeader().parse(reader);
    List<CSVRecord> records = csvParser.getRecords();

    Map<String, String[]> mappersAndGenotypes = new HashMap<>();

    for (CSVRecord record : records) {
      int seed = Integer.parseInt(record.get(seedColumnName));
      String mapper = record.get(mapperColumnName);
      String genotype = extractNeuralNetworkAsString(SerializationUtils.deserialize(record.get(serializedRobotColumnName), Robot.class, SerializationUtils.Mode.GZIPPED_JSON));
      String[] genotypes = mappersAndGenotypes.getOrDefault(mapper, new String[10]);
      genotypes[seed] = genotype;
      mappersAndGenotypes.put(mapper, genotypes);
    }

    CSVPrinter printer = new CSVPrinter(new PrintStream(outputFileName), CSVFormat.DEFAULT.withDelimiter(';'));
    printer.printRecord(List.of("mapper", "x1", "x2", "distance"));
    for (Map.Entry<String, String[]> entry : mappersAndGenotypes.entrySet()) {
      String mapper = entry.getKey();
      double[][] distances = computeNormalizedHammingDistanceMatrix(entry.getValue());
      for (int i = 0; i < distances.length; i++) {
        for (int j = 0; j < distances.length; j++) {
          printer.printRecord(List.of(mapper, i + "", j + "", distances[i][j] + ""));
        }
      }

    }


  }

  private static String extractNeuralNetworkAsString(Robot robot) {
    robot.reset();
    CentralizedSensing centralizedSensing = (CentralizedSensing) robot.getController();
    Parametrized parametrized = (Parametrized) centralizedSensing.getFunction();
    double[] originalWeights = parametrized.getParams();
    return Arrays.stream(originalWeights).mapToObj(p -> p > 0 ? "p" : (p < 0 ? "n" : "z")).collect(Collectors.joining(","));
  }

  private static double[][] computeNormalizedHammingDistanceMatrix(String[] strings) {
    double[][] distances = new double[strings.length][strings.length];
    for (int i = 0; i < strings.length; i++) {
      for (int j = i + 1; j < strings.length; j++) {
        distances[i][j] = distances[j][i] = computeNormalizedHammingDistance(strings[i], strings[j]);
      }
    }
    return distances;
  }

  private static double computeNormalizedHammingDistance(String s1, String s2) {
    if (s1.length() != s2.length()) {
      throw new IllegalArgumentException("The strings need to be of equal length.");
    }
    double distance = 0;
    for (int i = 0; i < s1.length(); i++) {
      if (s1.charAt(i) != s2.charAt(i)) {
        distance++;
      }
    }
    return distance / s1.length();
  }

}
