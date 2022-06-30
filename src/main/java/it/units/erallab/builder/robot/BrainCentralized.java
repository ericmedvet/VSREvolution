package it.units.erallab.builder.robot;

import it.units.erallab.builder.NamedProvider;
import it.units.erallab.builder.PrototypedFunctionBuilder;
import it.units.erallab.hmsrobots.core.controllers.CentralizedSensing;
import it.units.erallab.hmsrobots.core.controllers.RealFunction;
import it.units.erallab.hmsrobots.core.controllers.TimedRealFunction;
import it.units.erallab.hmsrobots.core.objects.Robot;
import it.units.erallab.hmsrobots.core.objects.Voxel;
import it.units.erallab.hmsrobots.util.Grid;
import it.units.erallab.hmsrobots.util.SerializationUtils;

import java.util.Map;
import java.util.function.Function;

/**
 * @author eric
 */
public class BrainCentralized implements NamedProvider<PrototypedFunctionBuilder<TimedRealFunction, Robot>> {

  @Override
  public PrototypedFunctionBuilder<TimedRealFunction, Robot> build(Map<String, String> params) {
    return new PrototypedFunctionBuilder<>() {
      @Override
      public Function<TimedRealFunction, Robot> buildFor(Robot robot) {
        Grid<Voxel> body = robot.getVoxels();
        return function -> {
          if (function.getInputDimension() != CentralizedSensing.nOfInputs(body)) {
            throw new IllegalArgumentException(String.format(
                "Wrong number of function input args: %d expected, %d found",
                CentralizedSensing.nOfInputs(body),
                function.getInputDimension()
            ));
          }
          if (function.getOutputDimension() != CentralizedSensing.nOfOutputs(body)) {
            throw new IllegalArgumentException(String.format(
                "Wrong number of function output args: %d expected, %d found",
                CentralizedSensing.nOfOutputs(body),
                function.getOutputDimension()
            ));
          }
          return new Robot(
              new CentralizedSensing(body, function),
              SerializationUtils.clone(body)
          );
        };
      }

      @Override
      public TimedRealFunction exampleFor(Robot robot) {
        Grid<Voxel> body = robot.getVoxels();
        return RealFunction.build(
            d -> d,
            CentralizedSensing.nOfInputs(body),
            CentralizedSensing.nOfOutputs(body)
        );
      }
    };
  }


}
