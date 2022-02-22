package it.units.erallab;

import it.units.erallab.hmsrobots.core.objects.Robot;
import it.units.erallab.hmsrobots.core.objects.Voxel;
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
import java.util.ArrayList;
import java.util.List;

public class PhysicalParametersParser extends Worker {

  public static void main(String[] args) {
    new PhysicalParametersParser(args);
  }

  public PhysicalParametersParser(String[] args) {
    super(args);
  }

  public void run() {

    String inputFileName = "C:\\Users\\giorg\\Documents\\UNITS\\PHD\\Physical_parameters\\Active\\best.txt";
    String outputFileName = "C:\\Users\\giorg\\Documents\\UNITS\\PHD\\Physical_parameters\\Active\\best-small.txt";

    String robotsColumn = "best→solution→serialized";

    CSVPrinter printer;
    try {
      printer = new CSVPrinter(new PrintStream(outputFileName), CSVFormat.DEFAULT.withDelimiter(';'));
    } catch (IOException e) {
      e.printStackTrace();
      return;
    }

    Pair<List<String>, List<CSVRecord>> parsedCsv = readRecordsFromFile(inputFileName);
    List<String> headers = parsedCsv.first().stream().filter(s -> !s.equals(robotsColumn)).toList();
    List<String> newHeaders = new ArrayList<>(headers);
    newHeaders.addAll(List.of("spring.f", "spring.d", "friction", "delta.active"));
    try {
      printer.printRecord(newHeaders);
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
      Robot robot = SerializationUtils.deserialize(record.get(robotsColumn), Robot.class, SerializationUtils.Mode.GZIPPED_JSON);
      Voxel voxel = robot.getVoxels().stream().findFirst().get().value();
      List<String> newRecord = new ArrayList<>(headers.stream().map(record::get).toList());
      newRecord.add(voxel.getSpringF() + "");
      newRecord.add(voxel.getSpringD() + "");
      newRecord.add(voxel.getFriction() + "");
      newRecord.add(voxel.getDeltaActive() + "");

      try {
        printer.printRecord(newRecord);
      } catch (IOException e) {
        e.printStackTrace();
      }

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
