package it.units.erallab.mapper.robot;

import it.units.erallab.RealFunction;
import it.units.erallab.hmsrobots.core.controllers.CentralizedSensing;
import it.units.erallab.hmsrobots.core.objects.Robot;
import it.units.erallab.hmsrobots.core.objects.SensingVoxel;
import it.units.erallab.hmsrobots.util.Grid;
import it.units.erallab.hmsrobots.util.SerializationUtils;
import it.units.erallab.mapper.PrototypedFunctionBuilder;

import java.util.function.Function;

/**
 * @author eric
 */
public class FixedCentralized implements PrototypedFunctionBuilder<RealFunction, Robot<?>> {

  @Override
  public Function<RealFunction, Robot<?>> buildFor(Robot<?> robot) {
    Grid<? extends SensingVoxel> body = (Grid<? extends SensingVoxel>) robot.getVoxels();
    return function -> {
      if (function.getNOfInputs() != CentralizedSensing.nOfInputs(body)) {
        throw new IllegalArgumentException(String.format(
            "Wrong number of function input args: %d expected, %d found",
            CentralizedSensing.nOfInputs(body),
            function.getNOfInputs()
        ));
      }
      if (function.getNOfOutputs() != CentralizedSensing.nOfOutputs(body)) {
        throw new IllegalArgumentException(String.format(
            "Wrong number of function output args: %d expected, %d found",
            CentralizedSensing.nOfOutputs(body),
            function.getNOfOutputs()
        ));
      }
      return new Robot<>(
          new CentralizedSensing(body, function),
          SerializationUtils.clone(body)
      );
    };
  }

  @Override
  public RealFunction exampleFor(Robot<?> robot) {
    Grid<? extends SensingVoxel> body = (Grid<? extends SensingVoxel>) robot.getVoxels();
    return RealFunction.from(
        CentralizedSensing.nOfInputs(body),
        CentralizedSensing.nOfOutputs(body),
        v -> v
    );
  }

}
