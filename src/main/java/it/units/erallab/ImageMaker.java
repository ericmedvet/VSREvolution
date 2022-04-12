package it.units.erallab;

import it.units.erallab.hmsrobots.core.objects.Ground;
import it.units.erallab.hmsrobots.core.objects.Robot;
import it.units.erallab.hmsrobots.tasks.locomotion.Locomotion;
import it.units.erallab.hmsrobots.util.SerializationUtils;
import it.units.erallab.hmsrobots.viewers.AllRobotFollower;
import it.units.erallab.hmsrobots.viewers.FramesImageBuilder;
import it.units.erallab.hmsrobots.viewers.drawers.*;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;
import org.dyn4j.dynamics.Settings;

import javax.imageio.ImageIO;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.io.Reader;
import java.util.List;
import java.util.function.Function;

public class ImageMaker {

  public static void main(String[] args) throws Exception {

    String param = "active";
    String robotFileName = "biped-" + param + ".txt";
    String robotColumn = "best.solution.serialized";
    String descColumn = "desc";
    double frameDeltaT = 1d;
    double ghostDeltaT = 0.5;

    Reader reader = new FileReader(robotFileName);
    CSVParser csvParser = CSVFormat.DEFAULT.withDelimiter(';').withFirstRecordAsHeader().parse(reader);
    List<CSVRecord> records = csvParser.getRecords();
    reader.close();

    int i = 0;
    for (CSVRecord record : records) {
      String serializedRobot = record.get(robotColumn);
      Robot robot = SerializationUtils.deserialize(serializedRobot, Robot.class, SerializationUtils.Mode.GZIPPED_JSON);
      String desc = record.get(descColumn);
      String destFile = "D:\\Research\\Physical_parameters\\Video-def\\img\\0.5-" + desc + ".png";
      makeImg(robot, desc, destFile, frameDeltaT, ghostDeltaT);
      i++;
    }


  }

  private static void makeImg(Robot robot, String description, String destFile, double frameDeltaT, double ghostDeltaT) {
    Locomotion locomotion = new Locomotion(30, Locomotion.createTerrain("flat"), new Settings());

    Function<String, Drawer> drawerProvider = s -> Drawer.of(
        Drawer.clear(),
        Drawer.transform(
            new AllRobotFollower(1.5d, 2),
            Drawer.of(
                new GhostRobotDrawer(5 * ghostDeltaT, ghostDeltaT, 0, false),
                new PolyDrawer(PolyDrawer.TEXTURE_PAINT, SubtreeDrawer.Extractor.matches(null, Ground.class, null)),
                new VoxelDrawer()
            )
        )
    );

    FramesImageBuilder framesImageBuilder = new FramesImageBuilder(
        7,
        12,
        frameDeltaT,
        800,
        600,
        FramesImageBuilder.Direction.HORIZONTAL,
        drawerProvider.apply(description)
    );
    locomotion.apply(robot, framesImageBuilder);

    try {
      ImageIO.write(framesImageBuilder.getImage(), "png", new File(destFile));
    } catch (IOException e) {
      e.printStackTrace();
    }
  }

}
