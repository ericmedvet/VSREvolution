package it.units.erallab.mapper.robot;

import it.units.erallab.RealFunction;
import it.units.erallab.hmsrobots.core.controllers.DistributedSensing;
import it.units.erallab.hmsrobots.core.objects.Robot;
import it.units.erallab.hmsrobots.core.objects.SensingVoxel;
import it.units.erallab.hmsrobots.util.Grid;
import it.units.erallab.hmsrobots.util.SerializationUtils;
import it.units.erallab.mapper.PrototypedFunctionBuilder;

import java.util.function.Function;

/**
 * @author eric
 */
public class FixedHeteroDistributed implements PrototypedFunctionBuilder<Grid<RealFunction>, Robot<? extends SensingVoxel>> {
  private final int signals;

  public FixedHeteroDistributed(int signals) {
    this.signals = signals;
  }

  @Override
  public Function<Grid<RealFunction>, Robot<? extends SensingVoxel>> buildFor(Robot<? extends SensingVoxel> robot) {
    Grid<int[]> dims = getIODims(robot);
    Grid<? extends SensingVoxel> body = robot.getVoxels();
    return functions -> {
      //check
      if (dims.getW() != functions.getW() || dims.getH() != functions.getH()) {
        throw new IllegalArgumentException(String.format(
            "Wrong size of functions grid: %dx%d expected, %dx%d found",
            dims.getW(), dims.getH(),
            functions.getW(), functions.getH()
        ));
      }
      for (Grid.Entry<int[]> entry : dims) {
        if (entry.getValue() == null) {
          continue;
        }
        if (functions.get(entry.getX(), entry.getY()).getNOfInputs() != entry.getValue()[0]) {
          throw new IllegalArgumentException(String.format(
              "Wrong number of function input args at (%d,%d): %d expected, %d found",
              entry.getX(), entry.getY(),
              entry.getValue()[0],
              functions.get(entry.getX(), entry.getY()).getNOfInputs()
          ));
        }
        if (functions.get(entry.getX(), entry.getY()).getNOfInputs() != entry.getValue()[0]) {
          throw new IllegalArgumentException(String.format(
              "Wrong number of function output args at (%d,%d): %d expected, %d found",
              entry.getX(), entry.getY(),
              entry.getValue()[1],
              functions.get(entry.getX(), entry.getY()).getNOfOutputs()
          ));
        }
      }
      //return
      DistributedSensing controller = new DistributedSensing(body, signals);
      for (Grid.Entry<? extends SensingVoxel> entry : body) {
        if (entry.getValue() != null) {
          controller.getFunctions().set(entry.getX(), entry.getY(), functions.get(entry.getX(), entry.getY()));
        }
      }
      return new Robot<>(
          controller,
          SerializationUtils.clone(body)
      );
    };
  }

  @Override
  public Grid<RealFunction> exampleFor(Robot<? extends SensingVoxel> robot) {
    return Grid.create(
        getIODims(robot),
        dim -> dim == null ? null : RealFunction.from(dim[0], dim[1], d -> d)
    );
  }

  private Grid<int[]> getIODims(Robot<? extends SensingVoxel> robot) {
    Grid<? extends SensingVoxel> body = robot.getVoxels();
    return Grid.create(
        body,
        v -> v == null ? null : new int[]{
            DistributedSensing.nOfInputs(v, signals),
            DistributedSensing.nOfOutputs(v, signals)
        }
    );
  }

}
