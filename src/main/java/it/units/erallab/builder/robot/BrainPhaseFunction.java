package it.units.erallab.builder.robot;

import it.units.erallab.builder.NamedProvider;
import it.units.erallab.builder.PrototypedFunctionBuilder;
import it.units.erallab.hmsrobots.core.controllers.PhaseSin;
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
public class BrainPhaseFunction implements NamedProvider<PrototypedFunctionBuilder<TimedRealFunction, Robot>> {

  @Override
  public PrototypedFunctionBuilder<TimedRealFunction, Robot> build(Map<String, String> params) {
    double frequency = Double.parseDouble(params.get("f"));
    double amplitude = Double.parseDouble(params.get("a"));
    return new PrototypedFunctionBuilder<>() {
      @Override
      public Function<TimedRealFunction, Robot> buildFor(Robot robot) {
        Grid<Voxel> body = robot.getVoxels();
        return function -> {
          if (function.getInputDimension() != 2) {
            throw new IllegalArgumentException(String.format(
                "Wrong number of function input args: 2 expected, %d found",
                function.getInputDimension()
            ));
          }
          return new Robot(new PhaseSin(frequency, amplitude, Grid.create(
              body.getW(),
              body.getH(),
              (x, y) -> function.apply(
                  0d,
                  new double[]{(double) x / (double) body.getW(), (double) y / (double) body.getW()}
              )[0]
          )), SerializationUtils.clone(body));
        };
      }

      @Override
      public TimedRealFunction exampleFor(Robot robot) {
        return RealFunction.build(d -> d, 2, 1);
      }
    };
  }


}
