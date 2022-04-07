package it.units.erallab;

import it.units.erallab.hmsrobots.core.controllers.CentralizedSensing;
import it.units.erallab.hmsrobots.core.controllers.MultiLayerPerceptron;
import it.units.erallab.hmsrobots.core.objects.Robot;
import it.units.erallab.hmsrobots.util.SerializationUtils;
import it.units.malelab.jgea.Worker;
import it.units.malelab.jgea.core.util.Pair;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVPrinter;
import org.apache.commons.csv.CSVRecord;

import java.io.FileReader;
import java.io.IOException;
import java.io.PrintStream;
import java.io.Reader;
import java.util.*;
import java.util.stream.IntStream;

public class GenotypeExtraction extends Worker {

  public static void main(String[] args) {
    new GenotypeExtraction(args);
  }

  public GenotypeExtraction(String[] args) {
    super(args);
  }

  private record GenotypeAndInfo(double[] genotype, int seed, double fitness) {
  }

  Map<String, Map<Double, List<GenotypeAndInfo>>> genotypes;

  public void run() {
    String inputFileName = a("inputFile", "D:\\Research\\Physical_parameters\\last-springd.txt");
    String outputFileName = a("outputFile", "D:\\Research\\Physical_parameters\\dist-springd.txt");

    String shapeColumn = a("shape", "shape");
    String parameterColumn = a("parameter", "spring.d");
    String robotColumn = a("robotColumn", "best.solution.serialized");
    String fitnessColumn = a("fitnessColumn", "best.fitness.fitness");
    String seedColumn = a("seedColumn", "seed");


    CSVPrinter printer;
    try {
      printer = new CSVPrinter(new PrintStream(outputFileName), CSVFormat.DEFAULT.withDelimiter(';'));
    } catch (IOException e) {
      e.printStackTrace();
      return;
    }
    try {
      printer.printRecord("variation", "parameter", "shape", "seed1", "seed2", "geno.dist", "fitness.dist");
    } catch (IOException e) {
      e.printStackTrace();
    }


    Pair<List<String>, List<CSVRecord>> parsedCsv = readRecordsFromFile(inputFileName);

    genotypes = new HashMap<>();

    List<CSVRecord> records = parsedCsv.second();
    if (records == null) {
      System.exit(-1);
    }
    int i = 0;
    System.out.print(records.size() + "\n");
    for (CSVRecord record : records) {
      System.out.println(i);
      i++;
      String shape = record.get(shapeColumn);
      double param = Double.parseDouble(record.get(parameterColumn));
      Robot robot = SerializationUtils.deserialize(record.get(robotColumn), Robot.class, SerializationUtils.Mode.GZIPPED_JSON);
      CentralizedSensing centralizedSensing = (CentralizedSensing) robot.getController();
      MultiLayerPerceptron multiLayerPerceptron = (MultiLayerPerceptron) centralizedSensing.getFunction();
      double[] genotype = multiLayerPerceptron.getParams();
      double fitness = Double.parseDouble(record.get(fitnessColumn));
      int seed = Integer.parseInt(record.get(seedColumn));

      Map<Double, List<GenotypeAndInfo>> genotypeWithThisShape = genotypes.getOrDefault(
          shape,
          new HashMap<>()
      );
      List<GenotypeAndInfo> genotypeWithThisShapeAndParam = genotypeWithThisShape.getOrDefault(
          param,
          new ArrayList<>()
      );
      GenotypeAndInfo genotypeAndInfo = new GenotypeAndInfo(genotype, seed, fitness);
      genotypeWithThisShapeAndParam.add(genotypeAndInfo);
      genotypeWithThisShape.put(param, genotypeWithThisShapeAndParam);
      genotypes.put(shape, genotypeWithThisShape);
    }

    Set<String> shapes = genotypes.keySet();
    for (String shape : shapes) {
      Set<Double> paramValues = genotypes.get(shape).keySet();
      for (double paramValue : paramValues) {
        List<GenotypeAndInfo> genotypeAndInfos = genotypes.get(shape).get(paramValue);
        List<List<String>> recs = computeRecords(parameterColumn, shape, paramValue, genotypeAndInfos);
        try {
          printer.printRecords(recs);
        } catch (IOException e) {
          e.printStackTrace();
        }
      }
    }

  }

  private List<List<String>> computeRecords(String paramName, String shape, double param, List<GenotypeAndInfo> genotypeAndInfos) {
    List<List<String>> lists = new ArrayList<>();
    for (int i = 0; i < genotypeAndInfos.size(); i++) {
      for (int j = i + 1; j < genotypeAndInfos.size(); j++) {
        GenotypeAndInfo genotypeAndInfo1 = genotypeAndInfos.get(i);
        GenotypeAndInfo genotypeAndInfo2 = genotypeAndInfos.get(j);
        int seed1 = genotypeAndInfo1.seed;
        int seed2 = genotypeAndInfo2.seed;
        double genotypeDistance = computeDistance(genotypeAndInfo1.genotype(), genotypeAndInfo2.genotype());
        double fitnessDifference = Math.abs(genotypeAndInfo1.fitness() - genotypeAndInfo2.fitness());
        lists.add(List.of(
            paramName, param + "", shape, seed1 + "", seed2 + "", genotypeDistance + "", fitnessDifference + ""
        ));
      }
    }
    return lists;
  }

  private static double computeDistance(double[] d1, double[] d2) {
    double d = IntStream.range(0, d1.length).mapToDouble(i -> Math.pow(d1[i] - d2[i], 2)).sum();
    return Math.sqrt(d);
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
