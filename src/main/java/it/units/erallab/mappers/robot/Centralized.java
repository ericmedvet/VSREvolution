package it.units.erallab.mappers.robot;

import it.units.erallab.RealFunction;
import it.units.erallab.hmsrobots.core.controllers.CentralizedSensing;
import it.units.erallab.hmsrobots.core.objects.Robot;
import it.units.erallab.hmsrobots.core.objects.SensingVoxel;
import it.units.erallab.hmsrobots.util.Grid;
import it.units.erallab.hmsrobots.util.SerializationUtils;

import java.util.function.Function;

/**
 * @author eric
 * @created 2020/12/07
 * @project VSREvolution
 */
public class Centralized implements RobotMapper<RealFunction> {
  @Override
  public Function<RealFunction, Robot<?>> apply(Grid<? extends SensingVoxel> body) {
    return function -> {
      if (function.getInputDim() != CentralizedSensing.nOfInputs(body)) {
        throw new IllegalArgumentException(String.format(
            "Wrong number of function input args: %d expected, %d found",
            CentralizedSensing.nOfInputs(body),
            function.getInputDim()
        ));
      }
      if (function.getOutputDim() != CentralizedSensing.nOfOutputs(body)) {
        throw new IllegalArgumentException(String.format(
            "Wrong number of function output args: %d expected, %d found",
            CentralizedSensing.nOfOutputs(body),
            function.getOutputDim()
        ));
      }
      return new Robot<>(
          new CentralizedSensing(body, function),
          SerializationUtils.clone(body)
      );
    };
  }

  @Override
  public RealFunction example(Grid<? extends SensingVoxel> body) {
    return RealFunction.from(
        CentralizedSensing.nOfInputs(body),
        CentralizedSensing.nOfOutputs(body),
        in -> new double[CentralizedSensing.nOfOutputs(body)]
    );
  }
}
