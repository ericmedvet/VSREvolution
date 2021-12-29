package it.units.erallab.builder.robot;

import it.units.erallab.builder.PrototypedFunctionBuilder;
import it.units.erallab.hmsrobots.core.controllers.DistributedSensing;
import it.units.erallab.hmsrobots.core.controllers.RealFunction;
import it.units.erallab.hmsrobots.core.controllers.TimedRealFunction;
import it.units.erallab.hmsrobots.core.objects.Robot;
import it.units.erallab.hmsrobots.core.objects.Voxel;
import it.units.erallab.hmsrobots.util.Grid;
import it.units.erallab.hmsrobots.util.SerializationUtils;

import java.util.function.Function;

/**
 * @author eric
 */
public class FixedHeteroDistributed implements PrototypedFunctionBuilder<Grid<TimedRealFunction>, Robot> {
  private final int signals;

  public FixedHeteroDistributed(int signals) {
    this.signals = signals;
  }

  @Override
  public Function<Grid<TimedRealFunction>, Robot> buildFor(Robot robot) {
    Grid<int[]> dims = getIODims(robot);
    Grid<Voxel> body = robot.getVoxels();
    return functions -> {
      //check
      if (dims.getW() != functions.getW() || dims.getH() != functions.getH()) {
        throw new IllegalArgumentException(String.format("Wrong size of functions grid: %dx%d expected, %dx%d found",
            dims.getW(),
            dims.getH(),
            functions.getW(),
            functions.getH()
        ));
      }
      for (Grid.Entry<int[]> entry : dims) {
        if (entry.value() == null) {
          continue;
        }
        if (functions.get(entry.key().x(), entry.key().y()).getInputDimension() != entry.value()[0]) {
          throw new IllegalArgumentException(String.format(
              "Wrong number of function input args at (%d,%d): %d expected, %d found",
              entry.key().x(),
              entry.key().y(),
              entry.value()[0],
              functions.get(entry.key().x(), entry.key().y()).getInputDimension()
          ));
        }
        if (functions.get(entry.key().x(), entry.key().y()).getOutputDimension() != entry.value()[1]) {
          throw new IllegalArgumentException(String.format(
              "Wrong number of function output args at (%d,%d): %d expected, %d found",
              entry.key().x(),
              entry.key().y(),
              entry.value()[1],
              functions.get(entry.key().x(), entry.key().y()).getOutputDimension()
          ));
        }
      }
      //return
      DistributedSensing controller = new DistributedSensing(body, signals);
      for (Grid.Entry<Voxel> entry : body) {
        if (entry.value() != null) {
          controller.getFunctions()
              .set(entry.key().x(), entry.key().y(), functions.get(entry.key().x(), entry.key().y()));
        }
      }
      return new Robot(controller, SerializationUtils.clone(body));
    };
  }

  @Override
  public Grid<TimedRealFunction> exampleFor(Robot robot) {
    return Grid.create(getIODims(robot), dim -> dim == null ? null : RealFunction.build(d -> d, dim[0], dim[1]));
  }

  private Grid<int[]> getIODims(Robot robot) { // TODO should return an ad hoc record
    Grid<Voxel> body = robot.getVoxels();
    return Grid.create(body,
        v -> v == null ? null : new int[]{DistributedSensing.nOfInputs(v, signals), DistributedSensing.nOfOutputs(
            v,
            signals
        )}
    );
  }

}
